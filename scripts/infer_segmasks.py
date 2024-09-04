import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForUniversalSegmentation,
    OneFormerProcessor,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


MAX_SEG_ID = 256**3 - 1


class InfDataset(Dataset):
    def __init__(self, img_dir, processor):
        self.paths = sorted(list(Path(img_dir).glob("*.jpg")))
        self.processor = processor
        self.processor_kwargs = {"return_tensors": "pt"}
        if isinstance(processor, OneFormerProcessor):
            self.processor_kwargs["task_inputs"] = ["panoptic"]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        inputs = self.processor(images=img, **self.processor_kwargs)
        inputs["pixel_values"] = inputs["pixel_values"][0]
        if "task_inputs" in inputs:
            inputs["task_inputs"] = inputs["task_inputs"][0]
        inputs["pixel_mask"] = inputs["pixel_mask"][0]
        return inputs, img.height, img.width, str(p)


def _id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def mask_to_boxes(mask):
    y, x = torch.where(mask != 0)
    return torch.stack([torch.min(x), torch.min(y), torch.max(x), torch.max(y)])


class NoInstancesError(Exception):
    pass


def topsg(result, filename, height, width):
    filename = Path(filename)
    fileno = int(filename.stem)

    seg_mask = result["segmentation"].cpu()
    seg_info = result["segments_info"]
    if len(seg_info) == 0:
        raise NoInstancesError(str(filename))

    seg_id_scale = MAX_SEG_ID // len(seg_info)

    out_mask = _id2rgb(seg_mask.numpy() * seg_id_scale)
    pan_seg_file_name = f"seg_{filename.stem}.png"
    seg_img = Image.fromarray(out_mask)

    out_seg_info = []
    out_boxes = []
    for info in seg_info:
        seg_id = info["id"] * seg_id_scale
        assert seg_id <= MAX_SEG_ID
        bin_mask = seg_mask == info["id"]
        out_seg_info.append(
            dict(
                id=seg_id,
                category_id=info["label_id"],
                iscrowd=0,
                isthing=1 if info["label_id"] < 80 else 0,
                # attribute_ids=[],
                area=bin_mask.sum().item(),
                score=info["score"],
            )
        )

        # create bounding boxes from the masks
        out_boxes.append(
            dict(
                bbox=mask_to_boxes(bin_mask).tolist(),
                bbox_mode=0,
                category_id=info["label_id"],
                score=info["score"],
            )
        )

    return (
        dict(
            file_name=f"{filename.parent.name}/{filename.name}",
            height=height,
            width=width,
            image_id=fileno,
            pan_seg_file_name=pan_seg_file_name,
            segments_info=out_seg_info,
            # location="",
            # weather="",
            annotations=out_boxes,
            # coco_image_id=None,
        ),
        seg_img,
    )


def main(img_dir, out_dir, psg_path, model: str, num_workers: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(psg_path) as f:
        psg_metadata = {
            k: v
            for k, v in json.load(f).items()
            if k in ("thing_classes", "stuff_classes", "predicate_classes")
        }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    processor = AutoProcessor.from_pretrained(model)
    model = AutoModelForUniversalSegmentation.from_pretrained(model)
    model.to(device)

    print("Using:", type(model).__name__)

    # make sure that the model outputs the same classes as PSG
    class_labels = psg_metadata["thing_classes"] + psg_metadata["stuff_classes"]
    assert len(model.config.id2label) == len(class_labels), len(model.config.id2label)
    for i, n in enumerate(class_labels):
        assert model.config.id2label[i] == n, (model.config.id2label[i], n)

    loader = DataLoader(
        dataset=InfDataset(img_dir, processor),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    new_data = []
    skipped = []
    assert loader.batch_size == 1
    with torch.no_grad():
        for inputs, height, width, abs_path in tqdm(
            loader, unit="file", dynamic_ncols=True
        ):
            # batch size is 1
            height = height[0].item()
            width = width[0].item()
            abs_path = abs_path[0]

            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            if "task_inputs" in inputs:
                inputs["task_inputs"] = inputs["task_inputs"].to(device)

            # actual inference
            outputs = model(**inputs)

            # estimate confidence scores, restructure model output
            # label_ids_to_fuse is set to an empty set to suppress warning messages
            result = processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[(height, width)], label_ids_to_fuse=set()
            )[0]

            try:
                a, seg_img = topsg(
                    result=result,
                    filename=abs_path,
                    height=height,
                    width=width,
                )
            except NoInstancesError as err:
                skipped.append(str(err))
                continue

            seg_img.save(out_dir / a["pan_seg_file_name"])
            a["pan_seg_file_name"] = out_dir.name + "/" + a["pan_seg_file_name"]
            new_data.append(a)

    with open(out_dir / "anno.json", "w") as f:
        json.dump(dict(data=new_data, skipped=skipped, **psg_metadata), f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("img")
    parser.add_argument("output")
    parser.add_argument("psg")
    parser.add_argument(
        "--model", default="facebook/mask2former-swin-large-coco-panoptic"
    )
    parser.add_argument("--workers", default=6, type=int)
    args = parser.parse_args()
    main(
        img_dir=args.img,
        out_dir=args.output,
        psg_path=args.psg,
        model=args.model,
        num_workers=args.workers,
    )
