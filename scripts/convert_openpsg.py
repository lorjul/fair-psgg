# convert results.pkl file to my super duper output file format
# For this repo we want to make sure that we get evaluation right and no information during training leaks into evaluation.
# Therefore, a model stores all the outputs in a file, and then only this file is used for evaluation.
# This enables a strict separation of training and final evaluation.
# It has another benefit! Other repos can write the results in our file format and use the evaluation script with ease

# The file is basically a Python pickle file. In this file we store a list of dicts. We make sure that no data types in the
# pickle file require any external dependencies. Therefore, it should work with any framework out there.
# For the exact format, check the convert() function.

import pickle
import json
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0

    return float(intersection / union)


def rgb2id(color):
    """Converts a given color to the internal segmentation id
    Adapted from https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L73
    """
    if color.dtype == np.uint8:
        color = color.astype(np.int32)
    return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]


def load_gt_seg_mask(seg_path, segments_info):
    arr = np.asarray(Image.open(seg_path))
    arr = rgb2id(arr)
    seg_ids = np.array([info["id"] for info in segments_info])[:, None, None]
    output_ids = np.full(arr.shape, fill_value=-1, dtype=int)
    # assign index to masks
    # pixels that have no mask get max_idx+1 assigned
    for i, s in enumerate(seg_ids):
        output_ids[arr == s] = i
    return output_ids


def mask_merge(
    masks: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.5,
    not_matched_threshold: float = 1.0,
):
    # do an NMS-style reduction of the masks and re-assign pairs afterwards
    mask_id_by_confidence = confidences.argsort()[::-1]

    ious2 = {}

    def _get_iou(i, k, masks):
        if (i, k) not in ious2:
            iou = calculate_iou(masks[i], masks[k])
            ious2[(i, k)] = iou
            ious2[(k, i)] = iou
            return iou
        return ious2[(i, k)]

    pan_mask2 = np.full(masks[0].shape, fill_value=-1, dtype=int)
    remapping = {}
    core_remap = {}
    # remap mask to other mask with IoU > 0.5 that has higher confidence
    for i, conf_i in enumerate(mask_id_by_confidence.tolist()):
        if conf_i in remapping:
            continue

        # assign pixels from this mask to the pan mask
        untouched = pan_mask2 == -1
        new_area = untouched & masks[conf_i]
        # check if the area of the mask is already covered by other masks
        # in that case, this mask has to be ignored or consumed by another mask
        if new_area.sum() == 0:
            continue

        pan_mask2[new_area] = len(core_remap)
        core_remap[conf_i] = len(core_remap)

        # for all other masks: assign to the current mask if IoU > 0.5 and not already assigned
        # all other masks are all masks with a lower confidence ([i+1:])
        for k in mask_id_by_confidence[i + 1 :].tolist():
            if k not in remapping and _get_iou(conf_i, k, masks) >= threshold:
                remapping[k] = conf_i

    big_remap = dict(core_remap)
    for orig, better in remapping.items():
        big_remap[orig] = core_remap[better]

    # not matched
    not_matched = set(range(len(masks))) - set(big_remap.keys())
    for idx in not_matched:
        best_iou = 0
        best_idx = None
        for core in core_remap:
            iou = _get_iou(idx, core, masks)
            if iou > best_iou:
                best_iou = iou
                best_idx = core
        # assign to best_idx
        # assert best_idx is not None
        if best_iou >= not_matched_threshold:
            big_remap[idx] = core_remap[core]

    return pan_mask2, big_remap, core_remap


def convert(anno, results, merge_mode, mask_dir=None, nomerge=False):
    assert merge_mode in ("2stage", "hilo", "psgtr")
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
    # verify that result and annotation file match
    test_annos = []
    for a in anno["data"]:
        # psg.json contains some files without relation information in the test set
        # we cannot evaluate on those and therefore, we skip them (as do the PSG models during evaluation)
        if a["image_id"] in anno["test_image_ids"] and len(a["relations"]) > 0:
            test_annos.append(a)

    assert len(results) == len(test_annos), (len(results), len(test_annos))

    outputs = []
    for r, a in zip(tqdm(results, dynamic_ncols=True), test_annos):
        assert (r.labels > 0).all()
        # different repos store their results differently, therefore we have different conversion modes
        if merge_mode == "2stage":
            assert mask_dir is not None
            # take the masks from the annotation file
            # they have the same ordering as the bboxes
            np.testing.assert_almost_equal(
                r.bboxes, np.array([b["bbox"] for b in a["annotations"]]), decimal=1
            )
            masks = load_gt_seg_mask(
                mask_dir / a["pan_seg_file_name"], a["segments_info"]
            )
            bboxes = r.bboxes
            box_label = r.labels
            pairs = r.rel_pair_idxes
            rel_scores = r.rel_dists
            rel_rank = r.triplet_scores
        elif merge_mode == "hilo":
            masks, full_remap, core_remap = mask_merge(r.masks, r.refine_bboxes[:, -1])
            # remap boxes (and labels)
            bboxes = np.empty(
                (len(core_remap), r.refine_bboxes.shape[1]), dtype=r.refine_bboxes.dtype
            )
            box_label = np.empty((len(core_remap),), dtype=r.labels.dtype)
            for orig, new in core_remap.items():
                bboxes[new] = r.refine_bboxes[orig]
                box_label[new] = r.labels[orig]
            # remap pairs (remove all pairs that refer to nodes that could not be matched)
            pairs = []
            rel_scores = []
            for (sbj, obj), rs in zip(r.rel_pair_idxes, r.rel_dists):
                sbj = int(sbj)
                obj = int(obj)
                if sbj in full_remap and obj in full_remap:
                    pairs.append((full_remap[sbj], full_remap[obj]))
                    rel_scores.append(rs)
            pairs = np.array(pairs)
            rel_scores = np.stack(rel_scores)
            # hilo already sorts the outputs in inverse order
            rel_rank = np.arange(len(pairs))[::-1]
        elif merge_mode == "psgtr":
            if len(r.rel_pair_idxes) == 0:
                continue
            masks = r.masks.argmax(0)
            masks[r.masks.sum(0) == 0] = -1
            # remap boxes (and labels)
            bboxes = None
            box_label = r.labels
            if nomerge:
                # do not filter unique pairs
                pairs = r.rel_pair_idxes
                rel_scores = r.rel_dists
            else:
                rel_scores = []
                pairs = []
                for p in np.unique(r.rel_pair_idxes, axis=0):
                    group = (r.rel_pair_idxes[:, 0] == p[0]) & (
                        r.rel_pair_idxes[:, 1] == p[1]
                    )
                    best_row = r.rel_dists[group, 1:].max(1).argmax()
                    rel_scores.append(r.rel_dists[group][best_row])
                    pairs.append(p)
                pairs = np.stack(pairs)
                rel_scores = np.stack(rel_scores)
            # sort by max predicate score per relation
            rel_rank = rel_scores[:, 1:].max(1)
        else:
            raise RuntimeError("You must specify --mode")
        outputs.append(
            {
                "img_id": a["image_id"],
                "bboxes": bboxes,
                "mask": masks,
                # box labels should start at 0
                "box_label": box_label - 1,
                "pairs": pairs,
                "rel_scores": rel_scores,
                "rel_rank": rel_rank,  # highest rank means most important
            }
        )

    return outputs


def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "anno",
        help="Path to annotation file that was used to generate the model output. "
        + "This is only required to determine the image_id for each result output and path to the segmentation masks if --masks is specified.",
    )
    parser.add_argument(
        "results",
        help="Results file from OpenPSG. In OpenPSG, run tools/test.py with the --out flag to get these files.",
    )
    parser.add_argument(
        "output", help="Path where the converted file should be saved to."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["2stage", "hilo", "psgtr"],
        help="Different repos store their results differently, therefore we have different conversion modes.",
    )
    parser.add_argument(
        "--masks",
        default=None,
        help="If specified, the masks from the annotation file and this path are included in the converted output. Useful for 2-stage methods.",
    )
    parser.add_argument(
        "--nomerge",
        default=False,
        action="store_true",
        help="Skip merging. Usually, you want merging so only use this option if you know what you're doing.",
    )
    args = parser.parse_args()

    with open(args.anno) as f:
        anno = json.load(f)

    with open(args.results, "rb") as f:
        results = pickle.load(f)

    converted = convert(
        anno, results, merge_mode=args.mode, mask_dir=args.masks, nomerge=args.nomerge
    )

    with open(args.output, "wb") as f:
        pickle.dump(converted, f)


if __name__ == "__main__":
    cli()
