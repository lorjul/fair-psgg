# create an inference dump file
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import pickle
import torch
from tqdm import tqdm
from ..data import get_loader
from ..data.split_batch import split_batch_iter
from ..trainer import prepare_batch
from ..config import Config
from .. import from_config


def cat0(tensors, dim=0):
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


@torch.inference_mode()
def inference2(
    model_folder,
    output_path,
    anno_path,
    img_dir,
    seg_dir,
    batch_size: int,
    num_workers: int,
    split="val",
    apply_sigmoid=True,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_folder = Path(model_folder)
    config = Config.from_file(model_folder / "config.json")

    loader = get_loader(
        anno_path=anno_path,
        split=split,
        img_dir=img_dir,
        seg_dir=seg_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentations=from_config.get_augmentations(config, split="test"),
        # for testing/inference, we want to look at all possible combinations
        allow_overlapping_negatives=True,
    )

    model = from_config.get_model(
        config,
        num_node_outputs=len(loader.dataset.node_names),
        num_rel_outputs=len(loader.dataset.rel_names),
    )

    checkpoint = torch.load(model_folder / "best_state.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    results = defaultdict(
        lambda: {
            "pairs": [],
            "rel_scores": [],
            "rel_rank": [],
        }
    )
    for batch in split_batch_iter(tqdm(loader, unit="batch"), max_relations=512):
        # get pair ids
        model_input, sbj_target, obj_target, rel_target = prepare_batch(batch, device)
        sbj_out, obj_out, rel_out, rel_ranking = model(model_input)

        if apply_sigmoid:
            rel_out = rel_out.sigmoid()

        batch_pairs = batch["sampled_relations"][:, :2]
        rel_out = rel_out.cpu().clone()
        bboxes = batch["bboxes"]
        box_labels = batch["box_categories"]

        rel_img_ids = torch.repeat_interleave(batch["image_id"], batch["num_relations"])
        box_img_ids = torch.repeat_interleave(batch["image_id"], batch["num_boxes"])
        for img_id, data_idx, in_img in zip(
            batch["image_id"], batch["idx"], batch["img"]
        ):
            key = str(int(img_id))

            raw_entry = loader.dataset.entries[data_idx]
            scale_x = raw_entry["width"] / in_img.size(2)
            scale_y = raw_entry["height"] / in_img.size(1)
            scale = max(scale_x, scale_y)

            # bboxes
            results[key]["bboxes"] = bboxes[box_img_ids == img_id] * scale
            results[key]["raw_bboxes"] = torch.tensor(
                [b["bbox"] for b in raw_entry["annotations"]]
            )
            results[key]["box_label"] = box_labels[box_img_ids == img_id]
            # masks
            seg_mask = loader.dataset._load_seg(
                raw_entry["pan_seg_file_name"], raw_entry["segments_info"]
            )
            seg_mask_ids = seg_mask.long().argmax(0)
            # where no mask is present, set to background (-1)
            seg_mask_ids[seg_mask.sum(0) == 0] = -1
            results[key]["raw_mask"] = seg_mask_ids
            # pairs
            results[key]["pairs"].append(batch_pairs[rel_img_ids == img_id])
            # relation scores
            results[key]["rel_rank"].append(1 - rel_out[rel_img_ids == img_id, 0])
            results[key]["rel_scores"].append(rel_out[rel_img_ids == img_id])

    output_data = []
    for img_id, values in results.items():
        output_data.append(
            {
                "img_id": img_id,
                "bboxes": values["raw_bboxes"].numpy(),
                "mask": values["raw_mask"].numpy(),
                "box_label": values["box_label"].numpy(),
                "pairs": cat0(values["pairs"]).numpy(),
                "rel_scores": cat0(values["rel_scores"]).numpy(),
                "rel_rank": cat0(values["rel_rank"]).numpy(),
            }
        )

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)


def cli():
    parser = ArgumentParser(
        description="Given a trained DSFormer model, runs inference on a given dataset. "
        "Note that this script only runs inference and does not evaluate the resulting file. "
        "Check the README for more information."
    )
    parser.add_argument("anno", help="Path to OpenPSG annotation file")
    parser.add_argument("img", help="Folder that contains images")
    parser.add_argument("seg", help="Folder that contains segmentation masks")
    parser.add_argument(
        "model",
        help="Path to model folder that contains config.json and best_state.pth",
    )
    parser.add_argument(
        "output", help="Output file where the inference results should be written to"
    )
    parser.add_argument("--bs", default=32, type=int, help="Batch size. Default: 32")
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        help="Number of workers for the data loader. Default: 0",
    )
    parser.add_argument(
        "--split", default="all", choices=("all", "val", "test", "train")
    )
    parser.add_argument(
        "--no-sigmoid",
        default=False,
        action="store_true",
        help="Don't apply sigmoid to the model outputs",
    )
    args = parser.parse_args()

    if not Path(args.model).is_dir():
        print("The MODEL argument must be path to a folder")
        exit(1)

    inference2(
        model_folder=args.model,
        output_path=args.output,
        anno_path=args.anno,
        img_dir=args.img,
        seg_dir=args.seg,
        batch_size=args.bs,
        num_workers=args.workers,
        split=args.split,
        apply_sigmoid=not args.no_sigmoid,
    )


if __name__ == "__main__":
    cli()
