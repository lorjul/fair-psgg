from argparse import ArgumentParser
import json
from torchvision.ops import box_iou
import torch
from collections import defaultdict


def match_gt(original_entry, md_entry):
    orig_boxes = torch.tensor([x["bbox"] for x in original_entry["annotations"]])
    md_boxes = torch.tensor([x["bbox"] for x in md_entry["annotations"]])
    ious = box_iou(orig_boxes, md_boxes)
    assign = defaultdict(list)
    for i, gt in enumerate(ious.argmax(dim=0)):
        assign[gt.item()].append(i)

    box_mapping = {}
    for gt, preds in assign.items():
        box_mapping[gt] = preds[ious[gt, preds].argmax()]

    new_relations = []
    for sbj, obj, rel in original_entry["relations"]:
        if sbj in box_mapping and obj in box_mapping:
            new_relations.append((box_mapping[sbj], box_mapping[obj], rel))

    new_entry = dict(md_entry)
    new_entry["relations"] = new_relations
    # has to match with test_image_ids
    new_entry["image_id"] = original_entry["image_id"]
    return new_entry


def cli():
    parser = ArgumentParser()
    parser.add_argument("psg", help="Path to original OpenPSG annotation file")
    parser.add_argument("custom", help="Path to inferred annotation file")
    parser.add_argument("output", help="Path to matched output JSON annotation file")
    parser.add_argument(
        "--enforce-rel",
        default=False,
        action="store_true",
        help="Whether to make sure that the matched annotation file contains only images with relations",
    )
    args = parser.parse_args()

    with open(args.custom) as f:
        custom_anno = json.load(f)

    with open(args.psg) as f:
        psg = json.load(f)

    custom_by_filename = {}
    for item in custom_anno["data"]:
        custom_by_filename[item["file_name"]] = item

    new_data = []
    for item in psg["data"]:
        if item["file_name"] not in custom_by_filename:
            continue
        m = custom_by_filename[item["file_name"]]
        new_entry = match_gt(item, m)
        if not args.enforce_rel or len(new_entry["relations"]) > 0:
            new_data.append(new_entry)

    to_write = dict(psg)
    to_write["data"] = new_data

    print("Lengths:")
    print(
        "PSG:",
        len(psg["data"]),
        "Custom:",
        len(custom_anno["data"]),
        "Matched:",
        len(new_data),
    )

    with open(args.output, "w") as f:
        json.dump(to_write, f)


if __name__ == "__main__":
    cli()
