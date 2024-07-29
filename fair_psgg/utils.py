import subprocess as sp
from typing import Union, Tuple
import numpy as np
import torch
from PIL import Image
import os


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_git_commit():
    proc = sp.Popen(["git", "rev-parse", "HEAD"], stdout=sp.PIPE, encoding="ascii")
    proc.wait()
    if proc.returncode == 0:
        return proc.stdout.read().strip()

    return None


def get_git_changes():
    proc = sp.Popen(["git", "diff"], stdout=sp.PIPE, encoding="utf-8")
    proc.wait()
    if proc.returncode == 0:
        return proc.stdout.read()

    return None


def detect_task_spooler():
    return "TS_SOCKET" in os.environ


def box_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor):
    # taken from https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html
    assert boxes1.dim() == boxes2.dim() == 2
    assert boxes1.size(1) == boxes2.size(1) == 4
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    return wh[:, :, 0] * wh[:, :, 1]  # [N,M]


def rgb2id(color: Union[np.ndarray, Tuple[int, int, int]]):
    """Converts a given color to the internal segmentation id
    Adapted from https://github.com/cocodataset/panopticapi/blob/7bb4655548f98f3fedc07bf37e9040a992b054b0/panopticapi/utils.py#L73
    """
    if isinstance(color, np.ndarray):
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[..., 0] + 256 * color[..., 1] + 256 * 256 * color[..., 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map: Union[np.ndarray, int]):
    if isinstance(id_map, np.ndarray):
        assert id_map.max() <= 256 * 256 * 256 - 1, id_map.max()
        rgb_map = np.empty((*id_map.shape, 3), dtype=np.uint8)
        rgb_map[..., 0] = id_map % 256
        rgb_map[..., 1] = (id_map // 256) % 256
        rgb_map[..., 2] = (id_map // 256 // 256) % 256
        return rgb_map
    color = [id_map % 256, (id_map // 256) % 256, (id_map // 256 // 256) % 256]
    return color


def open_segmask(path):
    with Image.open(path) as img:
        arr = np.asarray(img)
        return rgb2id(arr)


def get_num_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
