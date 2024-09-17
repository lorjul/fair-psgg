import json
from pathlib import Path
from PIL import Image
from project_paths import project_paths
from .data import get_loader
from . import from_config


def get_psg(split: str = "val", batch_size: int = 4, neg_ratio: float = None):
    if neg_ratio is None and split == "train":
        neg_ratio = 2.0
    return get_loader(
        anno_path=project_paths.psg_annotation_dir,
        split=split,
        img_dir=project_paths.psg_img_dir,
        seg_dir=project_paths.psg_seg_dir,
        batch_size=batch_size,
        num_workers=0,
        augmentations=from_config.get_augmentations(config=None, split=split),
        neg_ratio=neg_ratio,
    )


def load_psg_json():
    with open(project_paths.psg_annotation_dir) as f:
        return json.load(f)


def load_psg_image(item):
    if isinstance(item, str):
        file_name = item
    elif isinstance(item, dict):
        file_name = item["file_name"]
    else:
        raise RuntimeError()
    return Image.open(Path(project_paths.psg_img_dir) / file_name).convert("RGB")
