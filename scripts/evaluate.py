from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import json
import pickle
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    fbeta_score,
)
from tqdm import tqdm
import pandas as pd

# TODO: torch and torchvision dependency is only required to calculate box_iou
# we could also use a custom implementation and remove the dependency
# then, only numpy would be an external dependency for this script
import torch
from torchvision.ops import box_iou


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def sampled_acc_auc(gt, pred, samples=100):
    # TODO: sampled_acc should be reproducible (use a fixed sampling scheme and not random)
    neg = gt == 0
    pos = gt == 1
    assert neg.sum() + pos.sum() == len(gt)
    if neg.sum() < pos.sum():
        neg, pos = pos, neg

    neg = neg.nonzero()[0]
    accs = []
    for _ in range(samples):
        sample = np.random.permutation(len(neg))[: pos.sum()]
        new_neg = np.zeros(len(gt), dtype=bool)
        new_neg[neg[sample]] = True
        both_sel = new_neg | pos
        assert both_sel.sum() > 0
        for thresh in np.linspace(start=0.1, stop=0.9, num=17):
            accs.append((gt[both_sel] == (pred[both_sel] > thresh)).mean())

    return np.mean(accs)


def sampled_acc(gt, pred, samples=100):
    # TODO: sampled_acc should be reproducible (use a fixed sampling scheme and not random)
    neg = gt == 0
    pos = gt == 1
    assert neg.sum() + pos.sum() == len(gt)
    if neg.sum() < pos.sum():
        neg, pos = pos, neg

    neg = neg.nonzero()[0]
    accs = []
    for _ in range(samples):
        sample = np.random.permutation(len(neg))[: pos.sum()]
        new_neg = np.zeros(len(gt), dtype=bool)
        new_neg[neg[sample]] = True
        both_sel = new_neg | pos
        accs.append((gt[both_sel] == (pred[both_sel] > 0.5)).mean())

    return np.mean(accs)


def sampled_pauc(gt, pred, samples=100):
    # TODO: sampled_pauc should be reproducible (use a fixed sampling scheme and not random)
    neg = gt == 0
    pos = gt == 1
    assert neg.sum() + pos.sum() == len(gt)
    if neg.sum() < pos.sum():
        neg, pos = pos, neg

    neg = neg.nonzero()[0]
    scores = []
    for _ in range(samples):
        sample = np.random.permutation(len(neg))[: pos.sum()]
        new_neg = np.zeros(len(gt), dtype=bool)
        new_neg[neg[sample]] = True
        both_sel = new_neg | pos
        scores.append(roc_auc_score(gt[both_sel], pred[both_sel]))

    return np.mean(scores)


def match_boxes(gt_boxes, out_boxes, gt_labels=None, out_labels=None):
    ious = box_iou(out_boxes, gt_boxes).numpy()
    if gt_labels is not None and out_labels is not None:
        lbl_mismatch = out_labels[:, None] != gt_labels[None, :]
        ious[lbl_mismatch] = 0

    # PSG uses a threshold of 0.5 IoU
    min_thresh = 0.5
    gt_assign = defaultdict(list)
    for det_i, det in enumerate(ious):
        if (det > min_thresh).any():
            gt_assign[int(det.argmax())].append(det_i)

    matching = {}
    for g, dets in gt_assign.items():
        best_det = dets[ious[dets, g].argmax()]
        matching[best_det] = g

    return matching


def match_masks(
    gt_masks: np.ndarray, out_masks: np.ndarray, gt_labels=None, out_labels=None
):
    # background class is -1 and can be ignored for matching
    num_gt = gt_masks.max() + 1
    num_out = out_masks.max() + 1

    assert num_gt == len(gt_labels)
    assert num_out == len(out_labels), (num_out, len(out_labels))

    ious = np.zeros((num_out, num_gt))
    for i in range(num_out):
        for j in range(num_gt):
            ious[i, j] = mask_iou(out_masks == i, gt_masks == j)

    if gt_labels is not None and out_labels is not None:
        lbl_mismatch = out_labels[:, None] != gt_labels[None, :]
        ious[lbl_mismatch] = 0

    # PSG uses a threshold of 0.5 IoU
    min_thresh = 0.5
    gt_assign = defaultdict(list)
    for det_i, det in enumerate(ious):
        if (det > min_thresh).any():
            gt_assign[int(det.argmax())].append(det_i)

    matching = {}
    for g, dets in gt_assign.items():
        best_det = dets[ious[dets, g].argmax()]
        matching[best_det] = g

    return matching


def dedup_rels_ng_style(item: dict):
    """If there are duplicate relations, aggregate them into a single one by
    choosing the highest predicate score among all relation candidates.
    """
    old_pairs = item["pairs"]
    old_scores = item["rel_scores"]
    old_rel_rank = item.get("rel_rank")
    if old_rel_rank is None:
        old_rel_rank = np.arange(len(old_pairs))[::-1]
    new_pairs = []
    new_scores = []
    new_rel_ranks = []
    for p in np.unique(old_pairs, axis=0):
        group = (old_pairs[:, 0] == p[0]) & (old_pairs[:, 1] == p[1])
        # TODO: max for each predicate, store in new score array
        # use highest rel_rank, if available
        new_s = old_scores[group].max(0)
        # treat no-relation score differently?
        new_s[0] = old_scores[group, 0].mean()

        new_pairs.append(p)
        new_scores.append(new_s)
        new_rel_ranks.append(old_rel_rank[group].max())

    assert len(new_pairs) == len(np.unique(new_pairs, axis=0))
    assert len(new_pairs) == len(np.unique(old_pairs, axis=0))

    item["pairs"] = np.stack(new_pairs)
    item["rel_scores"] = np.stack(new_scores)
    item["rel_rank"] = np.stack(new_rel_ranks)


def dedup_rels_maxpred(item: dict):
    """If there are duplicate relations, choose the one with the highest predicate score.
    Discard the remaining relations.
    """
    old_pairs = item["pairs"]
    old_scores = item["rel_scores"]
    pair_idxes = np.zeros(len(old_pairs), dtype=bool)
    for p in np.unique(old_pairs, axis=0):
        group = (old_pairs[:, 0] == p[0]) & (old_pairs[:, 1] == p[1])
        best_row = old_scores[group, 1:].max(1).argmax()
        sel = np.zeros((group.sum(),), dtype=bool)
        sel[best_row] = True
        sel2 = group.copy()
        sel2[group] = sel
        pair_idxes[sel2] = True

    assert pair_idxes.sum() == len(np.unique(old_pairs, axis=0))

    item["pairs"] = item["pairs"][pair_idxes]
    item["rel_scores"] = item["rel_scores"][pair_idxes]
    if item.get("rel_rank") is not None:
        item["rel_rank"] = item["rel_rank"][pair_idxes]

    assert len(item["pairs"]) == len(np.unique(item["pairs"], axis=0))
    assert len(item["pairs"]) == len(np.unique(old_pairs, axis=0))


def dedup_rels_first(item: dict, k: int):
    """If there are duplicate relations, choose the one that comes first in the array.
    Discard the remaining relations.
    Most existing code bases sort the relations by importance.
    """

    num_pairs_before_dedup = len(item["pairs"])
    # remove duplicates (introduced e.g. by HiLo after mask merging)
    _, pair_idxes = np.unique(item["pairs"], axis=0, return_index=True)
    pair_idxes.sort()
    item["pairs"] = item["pairs"][pair_idxes]
    num_pairs_after_dedup = len(item["pairs"])
    if num_pairs_before_dedup >= k and num_pairs_after_dedup < k:
        print("Less than k after deduplication")
    item["rel_scores"] = item["rel_scores"][pair_idxes]

    if "rel_rank" in item and item["rel_rank"] is not None:
        item["rel_rank"] = item["rel_rank"][pair_idxes]


# calc stats for a single image (can be easily parallelized)
def calc_single(
    gt,
    item,
    ks,
    ignore_box_labels=True,
    match="mask",
    dedup_mode="first",
    is_ours=True,
):
    assert dedup_mode in ("none", "first", "max", "ng", "fail")
    if match == "box":
        gt_boxes = torch.tensor([b["bbox"] for b in gt["annotations"]])
        # in case the output contains an additional confidence value, just select the first 4 coefficients
        pred_bboxes = torch.tensor(item["bboxes"])[:, :4]

        if ignore_box_labels:
            matching = match_boxes(gt_boxes, pred_bboxes)
        else:
            gt_lbls = np.array([b["category_id"] for b in gt["annotations"]])
            matching = match_boxes(
                gt_boxes,
                pred_bboxes,
                gt_labels=gt_lbls,
                out_labels=item["box_label"],
            )
    elif match == "mask":
        pred_masks = item["mask"]
        gt_masks = gt["pan_mask"]

        # if len(pred_masks.shape) == 3:
        #     assert pred_masks.dtype == "bool", pred_masks.dtype
        #     bg = pred_masks.sum(0) == 0
        #     pred_masks = pred_masks.argmax(0)
        #     pred_masks[bg] = -1
        assert len(pred_masks.shape) == 2, pred_masks.shape

        if ignore_box_labels:
            matching = match_masks(gt_masks, pred_masks)
        else:
            gt_lbls = np.array([b["category_id"] for b in gt["annotations"]])
            matching = match_masks(
                gt_masks,
                pred_masks,
                gt_labels=gt_lbls,
                out_labels=item["box_label"],
            )
    else:
        raise NotImplementedError(match)
    # Matching complete, now we can check how many ground truth annotations were correctly identified.

    matched_gt_boxes = set(matching.values())

    gt_rels = defaultdict(list)
    gt_count = np.zeros(56, dtype=int)
    required_gt_boxes = set()
    for sbj, obj, rel in gt["relations"]:
        gt_rels[(sbj, obj)].append(rel)
        required_gt_boxes.add(sbj)
        required_gt_boxes.add(obj)

        gt_count[rel] += 1

    matched_ratio = len(required_gt_boxes.intersection(matched_gt_boxes)) / len(
        required_gt_boxes
    )

    ks_hit_counts = []
    ks_nogc_hit_counts = []

    for k in ks:
        # for R@gt
        if k == -1:
            # the model can choose the same number of relations as there are in the ground truth
            k = len(gt_rels)

        if dedup_mode == "none":
            pass
        elif dedup_mode == "fail":
            if len(item["pairs"]) != len(np.unique(item["pairs"], axis=0)):
                raise RuntimeError("Contains duplicate relations!")
        elif dedup_mode == "ng":
            dedup_rels_ng_style(item=item)
        elif dedup_mode == "first":
            dedup_rels_first(item=item, k=k)
        elif dedup_mode == "max":
            dedup_rels_maxpred(item)
        else:
            raise RuntimeError()

        if item.get("rel_rank") is None:
            # use the ordering of the rows as rel_rank (this is the default by OpenPSG and others)
            item["rel_rank"] = np.arange(len(item["pairs"]))[::-1]

        # Select the top k candidates based on the `rel_rank`
        sel = item["rel_rank"].argsort()[-k:]
        pairs = item["pairs"][sel]
        scores = item["rel_scores"][sel]

        # calculate normal recall hits
        hit_count = np.zeros(56, dtype=int)
        for (sbj, obj), rel_pred in zip(pairs, scores[:, 1:].argmax(-1)):
            if sbj not in matching or obj not in matching:
                continue
            g_sbj = matching[sbj]
            g_obj = matching[obj]
            if (g_sbj, g_obj) in gt_rels:
                g_rel_list = gt_rels[(g_sbj, g_obj)]
                for g_rel in g_rel_list:
                    if g_rel == rel_pred:
                        hit_count[g_rel] += 1

        # calculate nogc recall hits
        nogc_hit_count = np.zeros(56, dtype=int)
        if is_ours:
            fg_score = (1 - item["rel_scores"][:, 0])[:, None]
            ngc_scores = item["rel_scores"][:, 1:] * fg_score
        else:
            # just use the scores directly for nogc metrics
            ngc_scores = item["rel_scores"][:, 1:]
        ordering = ngc_scores.flatten().argsort()[-k:]
        pair_ids = np.tile(np.arange(len(item["pairs"]))[:, None], (1, 56)).flatten()[
            ordering
        ]
        pred_ids = np.tile(np.arange(56), len(item["rel_scores"]))
        for (sbj, obj), rel_pred in zip(item["pairs"][pair_ids], pred_ids[ordering]):
            if sbj not in matching or obj not in matching:
                continue
            g_sbj = matching[sbj]
            g_obj = matching[obj]
            if (g_sbj, g_obj) in gt_rels:
                g_rel_list = gt_rels[(g_sbj, g_obj)]
                for g_rel in g_rel_list:
                    if g_rel == rel_pred:
                        nogc_hit_count[g_rel] += 1

        ks_hit_counts.append(hit_count)
        ks_nogc_hit_counts.append(nogc_hit_count)

    # collect data for non-recall metrics
    pauc_gt = []
    pauc_scores = []
    for (sbj, obj), score_row in zip(pairs, scores):
        if sbj not in matching or obj not in matching:
            # TODO: do they also count for the final metric?
            continue
        g_sbj = matching[sbj]
        g_obj = matching[obj]
        ohe = np.zeros(len(score_row))
        for gr in gt_rels.get((g_sbj, g_obj), [-1]):
            ohe[gr + 1] = 1
        pauc_gt.append(ohe)
        pauc_scores.append(score_row)

    return (
        gt_count,
        ks_hit_counts,
        ks_nogc_hit_counts,
        matched_ratio,
        # TODO: handle the case when box matching fails
        np.stack(pauc_gt) if pauc_gt else None,
        np.stack(pauc_scores) if pauc_scores else None,
    )


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


def _nan0_mean(values):
    values = np.array(values)
    values[np.isnan(values)] = 0
    return values.mean()


def _calc_predicate_metrics(gt, scores, predicate_names):
    paucs = []
    f1s = []
    f2s = []
    f05s = []
    bal_accs = []
    full_accs = []
    accs = []
    acc_aucs = []

    assert len(gt) == len(scores) == len(predicate_names) + 1

    nan = float("nan")
    for g, s in zip(gt, scores):
        if g.sum() == 0:
            accs.append(nan)
            bal_accs.append(nan)
            acc_aucs.append(nan)
            full_accs.append(nan)
            f1s.append(nan)
            f2s.append(nan)
            f05s.append(nan)
            paucs.append(nan)
        else:
            accs.append(sampled_acc(g, s, samples=100))
            bal_accs.append(balanced_accuracy_score(g, s > 0.5))
            acc_aucs.append(sampled_acc_auc(g, s, samples=100))
            full_accs.append((g == (s > 0.5)).mean())
            f1s.append(f1_score(g, s > 0.5))
            f2s.append(fbeta_score(g, s > 0.5, beta=2))
            f05s.append(fbeta_score(g, s > 0.5, beta=0.5))
            paucs.append(roc_auc_score(g, s))

    # calculate mean values and insert in first row
    accs.insert(0, _nan0_mean(accs))
    bal_accs.insert(0, _nan0_mean(bal_accs))
    acc_aucs.insert(0, _nan0_mean(acc_aucs))
    full_accs.insert(0, _nan0_mean(full_accs))
    f1s.insert(0, _nan0_mean(f1s))
    f2s.insert(0, _nan0_mean(f2s))
    f05s.insert(0, _nan0_mean(accs))
    paucs.insert(0, _nan0_mean(paucs))

    return pd.DataFrame(
        {
            "acc": accs,
            "bal_acc": bal_accs,
            "acc_auc": acc_aucs,
            "full_acc": full_accs,
            "f1": f1s,
            "f2": f2s,
            "f05": f05s,
            "pauc": paucs,
        },
        index=["mean", "NONE"] + predicate_names,
    )


def _make_recall_table(predicate_scores, predicate_names):
    values = {m: s.tolist() for m, s in predicate_scores.items()}
    for m in values:
        values[m].insert(0, np.mean(values[m]))
        # TODO
        values[m].insert(1, pd.NA)
    return pd.DataFrame(values, index=["mean", "NONE"] + predicate_names)


def calc_metrics(
    psg,
    outputs,
    csv_path,
    ignore_box_labels: bool,
    seg_dir=None,
    dedup_mode="first",
    improved_ngR=True,
):
    if seg_dir is None:
        match_method = "box"
    else:
        match_method = "mask"
    predicate_names = psg["predicate_classes"]
    num_rel = len(predicate_names)
    byid = {}
    for d in psg["data"]:
        if d["image_id"] not in psg["test_image_ids"]:
            continue
        byid[d["image_id"]] = d

    ks = [10, 20, 50, -1]

    processed = set()
    ms = []
    gt_counts = []
    hit_counts = defaultdict(list)
    nogc_hit_counts = defaultdict(list)
    pauc_gt = []
    pauc_scores = []
    for x in tqdm(outputs):
        gt_item = dict(byid[x["img_id"]])
        if seg_dir is not None:
            gt_item["pan_mask"] = load_gt_seg_mask(
                seg_dir / gt_item["pan_seg_file_name"], gt_item["segments_info"]
            )
        g, h, nh, m, pg, ps = calc_single(
            gt_item,
            x,
            ks=ks,
            ignore_box_labels=ignore_box_labels,
            match=match_method,
            dedup_mode=dedup_mode,
            is_ours=improved_ngR,
        )
        gt_counts.append(g)
        for k, _h, _nh in zip(ks, h, nh):
            hit_counts[k].append(_h)
            nogc_hit_counts[k].append(_nh)
        ms.append(m)
        if pg is not None:
            pauc_gt.append(pg)
        if ps is not None:
            pauc_scores.append(ps)
        processed.add(x["img_id"])

    # don't forget about the images where no ground truth file exists (not processed by the segmentation model)
    for img_id, gt in byid.items():
        g = np.zeros(num_rel, dtype=int)
        h = np.zeros(num_rel, dtype=int)
        nh = np.zeros(num_rel, dtype=int)
        if img_id not in processed:
            for _, _, rel in gt["relations"]:
                g[rel] += 1
        gt_counts.append(g)
        for k in ks:
            hit_counts[k].append(h)
            nogc_hit_counts[k].append(nh)

    gt_counts = np.stack(gt_counts)
    hit_counts = {k: np.stack(v) for k, v in hit_counts.items()}
    nogc_hit_counts = {k: np.stack(v) for k, v in nogc_hit_counts.items()}
    pauc_gt = np.concatenate(pauc_gt)
    pauc_scores = np.concatenate(pauc_scores)

    predicate_metrics = _calc_predicate_metrics(
        pauc_gt.T, pauc_scores.T, predicate_names
    )

    metrics = {}
    # sum all counts, then average over all images

    for k in ks:
        metrics[f"all_mR@{k}"] = hit_counts[k].sum(0) / gt_counts.sum(0)
        metrics[f"all_mNgR@{k}"] = nogc_hit_counts[k].sum(0) / gt_counts.sum(0)

        # ignore division by zero errors (we use it to return NaNs that are skipped when averaging)
        with np.errstate(invalid="ignore"):
            # calculate score for each image, then average over all images
            metrics[f"img_mR@{k}"] = np.nanmean(hit_counts[k] / gt_counts, axis=0)
            metrics[f"img_mNgR@{k}"] = np.nanmean(
                nogc_hit_counts[k] / gt_counts, axis=0
            )

    recall_metrics = _make_recall_table(metrics, predicate_names)

    def _nanpad(v):
        return [v] + [pd.NA] * (num_rel + 1)

    for k in ks:
        recall_metrics[f"all_R@{k}"] = _nanpad(hit_counts[k].sum() / gt_counts.sum())
        recall_metrics[f"all_NgR@{k}"] = _nanpad(
            nogc_hit_counts[k].sum() / gt_counts.sum()
        )

        with np.errstate(invalid="ignore"):
            recall_metrics[f"img_R@{k}"] = _nanpad(
                np.nanmean(hit_counts[k].sum(1) / gt_counts.sum(1))
            )
            recall_metrics[f"img_NgR@{k}"] = _nanpad(
                np.nanmean(nogc_hit_counts[k].sum(1) / gt_counts.sum(1))
            )

    df = pd.concat(
        (predicate_metrics, recall_metrics), axis="columns", ignore_index=False
    )
    df.index.name = "predicate"
    df.to_csv(csv_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("psg", help="OpenPSG annotation ground truth")
    parser.add_argument("results", help="Results file, generated with inference script")
    parser.add_argument("save", help="CSV file to write the calculated metrics to")
    parser.add_argument(
        "--ignore-lbl",
        default=False,
        action="store_true",
        help="Ignore class labels during matching",
    )
    parser.add_argument(
        "--seg",
        default=None,
        help="Folder that contains the ground truth segmentation masks. "
        "If you omit this option, bounding boxes will be used instead of segmentation masks.",
    )
    parser.add_argument(
        "--dedup",
        default="first",
        choices=("none", "first", "max", "ng", "fail"),
        help="""If duplicate relations exist. How should they be aggregated?
    [none]  Keep all duplicates;
    [first] Choose the first relation from the list;
    [max]   Choose the relation that has the highest predicate score;
    [ng]    Use the maximum predicate score over all relations. The no-relation score is averaged;
    [fail]  Throw an error if duplicates exist
    """,
    )
    parser.add_argument(
        "--no-ng-improve",
        default=False,
        action="store_true",
        help="Whether to disable our improved predicate ranking for mNgR@k.",
    )
    args = parser.parse_args()

    with open(args.psg) as f:
        psg = json.load(f)
    with open(args.results, "rb") as f:
        outputs = pickle.load(f)

    calc_metrics(
        psg,
        outputs,
        csv_path=args.save,
        ignore_box_labels=args.ignore_lbl,
        seg_dir=None if args.seg is None else Path(args.seg),
        dedup_mode=args.dedup,
        improved_ngR=not args.no_ng_improve,
    )


if __name__ == "__main__":
    main()
