from pathlib import Path
from typing import Literal
import json
import numpy as np
from torchvision.ops import box_convert
from collections import defaultdict


def load_psg_entries(anno_path, split: Literal["train", "val", "test", "all"]):
    """Loads entries, node_names, rel_names for the `SGDataset`.
    For semi-supervised learning, you probably want to create a new entry loader funciton.
    """
    entries = []

    with open(anno_path) as f:
        anno = json.load(f)

    if split == "all":
        test_ids = set()
        # val_ids = set()
    else:
        test_ids = set(anno["test_image_ids"])
        # val_ids = set(anno["test_image_ids"])
    for img in anno["data"]:
        img = dict(img)

        if len(img["annotations"]) == 0:
            continue

        # skip self-relations and increase rel index (0 is reserved for no-rel)
        if "relations" in img:
            if len(img["relations"]) == 0:
                # don't allow empty relations in trainin set
                continue

            new_relations = []
            for sbj, obj, rel in img["relations"]:
                if sbj != obj:
                    new_relations.append((sbj, obj, rel + 1))
            if len(img["relations"]) > 0 and len(new_relations) == 0:
                # the only relation was a self-relation -> skip
                continue

            img["relations"] = new_relations
        else:
            # no relation annotation is only allowed for all/test split
            assert split in ("all", "test"), split

        img_id = img["image_id"]
        in_test = img_id in test_ids
        # in_val = img_id in val_ids
        if split == "all":
            entries.append(img)
        elif split == "train" and not in_test:
            entries.append(img)
        elif split == "val" and in_test:
            entries.append(img)
        elif split == "test" and in_test:
            entries.append(img)

    assert len(entries) > 0, f"Empty dataset: {split}"

    node_names = anno["thing_classes"] + anno["stuff_classes"]
    rel_names = anno["predicate_classes"]
    return entries, node_names, rel_names


def load_objects365_entries(anno_path, img_dir):
    """`img_dir` will be used to check if the image files exist"""
    img_dir = Path(img_dir)
    with open(anno_path) as f:
        raw_annos = json.load(f)

    assert all((i == x["id"] - 1 for i, x in enumerate(raw_annos["categories"])))
    node_names = [x["name"] for x in raw_annos["categories"]]
    rel_names = None

    anno_by_img = defaultdict(list)
    for a in raw_annos["annotations"]:
        anno_by_img[a["image_id"]].append(a)

    entries = []
    num_missing = 0
    for img in raw_annos["images"]:
        # check if file exists. TODO: download the missing files
        file_name = Path(img["file_name"]).name
        if not (img_dir / file_name).exists():
            num_missing += 1
            continue
        annos = []
        for a in anno_by_img[img["id"]]:
            x1, y1, w, h = a["bbox"]
            annos.append(
                {
                    "iscrowd": a["iscrowd"],
                    "bbox": (x1, y1, x1 + w, y1 + h),
                    "category_id": a["category_id"] - 1,
                }
            )
        entries.append(
            {
                "image_id": img["id"],
                "file_name": file_name,
                "annotations": annos,
            }
        )

    if num_missing > 0:
        print("Missing images:", num_missing)

    return entries, node_names, rel_names


def load_psgcoco_entries(anno_path, img_prefix, seg_prefix):
    """Loads all files from the coco folder of PSG. Even those that are not listed in the PSG annotation file"""
    with open(anno_path) as f:
        raw_annos = json.load(f)

    assert all((i == x["id"] - 1 for i, x in enumerate(raw_annos["categories"])))
    node_names = [x["name"] for x in raw_annos["categories"]]
    rel_names = None

    anno_by_img = {}
    for a in raw_annos["annotations"]:
        anno_by_img[a["image_id"]] = a

    entries = []
    for img in raw_annos["images"]:
        file_name = img["file_name"]
        pan_seg_file_name = anno_by_img[img["id"]]["file_name"]
        annos = []
        seg_info = []
        for a in anno_by_img[img["id"]]["segments_info"]:
            x1, y1, w, h = a["bbox"]
            annos.append(
                {
                    "iscrowd": a["iscrowd"],
                    "bbox": (x1, y1, x1 + w, y1 + h),
                    # starts at 1 in annotation file
                    "category_id": a["category_id"] - 1,
                }
            )
            seg_info.append({"id": a["id"]})
        entries.append(
            {
                "image_id": img["id"],
                "file_name": f"{img_prefix}/{file_name}",
                "pan_seg_file_name": f"{seg_prefix}/{pan_seg_file_name}",
                "annotations": annos,
                "segments_info": seg_info,
                # no relation information
            }
        )

    return entries, node_names, rel_names


def load_visual_genome_entries(
    img_data_path, meta_path, h5_path, split: Literal["train", "val", "test", "all"]
):
    # h5py dependency is only required for visual genome
    try:
        import h5py
    except ImportError:
        raise ImportError("Loading Visual Genome requires h5py to be installed")

    # load metadata from JSON file
    with open(meta_path) as f:
        metadata = json.load(f)
    node_names = [
        metadata["idx_to_label"][str(i + 1)]
        for i in range(len(metadata["idx_to_label"]))
    ]
    rel_names = [
        metadata["idx_to_predicate"][str(i + 1)]
        for i in range(len(metadata["idx_to_predicate"]))
    ]

    with open(img_data_path) as f:
        image_data = json.load(f)
    image_ids = []
    image_sizes = []
    for d in image_data:
        image_ids.append(d["image_id"])
        image_sizes.append(max(d["width"], d["height"]))
    image_ids = np.array(image_ids)
    image_sizes = np.array(image_sizes)

    # load annotations
    entries = []
    with h5py.File(h5_path) as f:
        if split == "train":
            group = f["split"] == 2
        elif split == "val":
            group = f["split"] == 1
        elif split == "test":
            group = f["split"] == 0
        elif split == "all":
            group = np.ones(len(f["split"]), dtype=bool)
        else:
            raise RuntimeError()
        for img_id, img_size, first_box, last_box, first_rel, last_rel in zip(
            image_ids[group],
            image_sizes[group],
            f["img_to_first_box"][group],
            f["img_to_last_box"][group],
            f["img_to_first_rel"][group],
            f["img_to_last_rel"][group],
        ):
            boxes = []
            coords = f["boxes_1024"][first_box, last_box + 1] / 1024 * img_size
            # convert to xyxy format
            coords = box_convert(coords, in_fmt="cxcywh", out_fmt="xyxy")
            # box labels start from 1 in the .h5 file
            categs = f["labels"][first_box, last_box + 1] - 1
            for coord, categ in zip(coords, categs):
                boxes.append(
                    {
                        # convert to xyxy format
                        "bbox": coord.tolist(),
                        "category_id": int(categ),
                    }
                )

            relations = []
            pairs = f["relationships"][first_rel : last_rel + 1]
            # predicate labels start from 1 in the .h5 file
            predicates = f["relationships"][first_rel : last_rel + 1] - 1
            for pair, rel in zip(pairs, predicates):
                relations.append((int(pair[0]), int(pair[1]), int(rel)))

            entries.append(
                {
                    "image_id": int(img_id),
                    "file_name": f"VG_100K/{img_id}.jpg",
                    "annotations": boxes,
                    "relations": relations,
                    # no segmentation information available
                    "pan_seg_file_name": None,
                    "segments_info": None,
                }
            )

    return entries, node_names, rel_names
