from argparse import ArgumentParser
from pathlib import Path
import json
import torch
from torchvision.ops import box_iou


def bboxes(x):
    return torch.tensor([box["bbox"] for box in x])


def cats(x):
    return torch.tensor([box["category_id"] for box in x])


def calc_stats(original, test):
    box_recalls = []
    train_recalls = []
    val_recalls = []
    box_recalls_igncls = []
    train_recalls_igncls = []
    val_recalls_igncls = []

    orig_byid = {}
    for x in original["data"]:
        if len(x["relations"]) > 0:
            orig_byid[x["image_id"]] = x
    test_byid = {}
    for x in test["data"]:
        test_byid[x["image_id"]] = x

    num_train_matched = 0
    num_val_matched = 0

    for idx, orig_item in orig_byid.items():
        if idx in test_byid:
            test_item = test_byid[idx]
            orig_boxes = bboxes(orig_item["annotations"])
            test_boxes = bboxes(test_item["annotations"])
            orig_cats = cats(orig_item["annotations"])
            test_cats = cats(test_item["annotations"])
            same_category = orig_cats[:, None] == test_cats[None]
            found = ((box_iou(orig_boxes, test_boxes) >= 0.5) & same_category).any(
                dim=1
            )
            box_recalls.append(found.sum() / len(orig_boxes))
            found_igncls = (box_iou(orig_boxes, test_boxes) >= 0.5).any(dim=1)
            box_recalls_igncls.append(found_igncls.sum() / len(orig_boxes))

            relations = torch.tensor(orig_item["relations"])
            found_rels = found[relations[:, 0]] & found[relations[:, 1]]
            r = found_rels.float().mean()
            found_rels_igncls = (
                found_igncls[relations[:, 0]] & found_igncls[relations[:, 1]]
            )
            r_ign = found_rels_igncls.float().mean()

            if idx in original["test_image_ids"]:
                num_val_matched += 1
            else:
                num_train_matched += 1
        else:
            box_recalls.append(0.0)
            box_recalls_igncls.append(0.0)
            r = 0.0
            r_ign = 0.0

        if idx in original["test_image_ids"]:
            val_recalls.append(r)
            val_recalls_igncls.append(r_ign)
        else:
            train_recalls.append(r)
            train_recalls_igncls.append(r_ign)

    return {
        "img_match_train": num_train_matched,
        "img_match_val": num_val_matched,
        "box_recall": torch.tensor(box_recalls).mean().item(),
        "box_recall_ign": torch.tensor(box_recalls_igncls).mean().item(),
        "mR@inf_train": torch.tensor(train_recalls).mean().item(),
        "mR@inf_val": torch.tensor(val_recalls).mean().item(),
        "mR@inf_train_ign": torch.tensor(train_recalls_igncls).mean().item(),
        "mR@inf_val_ign": torch.tensor(val_recalls_igncls).mean().item(),
    }


def cli():
    parser = ArgumentParser()
    parser.add_argument("original")
    parser.add_argument("test", nargs="+")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    with open(args.original) as f:
        original = json.load(f)

    if args.csv is not None:
        import warnings

        warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
        import pandas as pd

        rows = []
        for t in args.test:
            with open(t) as f:
                row = calc_stats(original, json.load(f))
                row["method"] = Path(t).parent.stem
                rows.append(row)
        df = pd.DataFrame(rows).set_index("method")
        df.to_csv(args.csv, index=True)
    else:
        for t in args.test:
            with open(t) as f:
                scores = calc_stats(original, json.load(f))
            for k, v in scores.items():
                print(k, float(v), sep=": ")


if __name__ == "__main__":
    cli()
