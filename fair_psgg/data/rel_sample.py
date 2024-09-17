from typing import Union
import torch
from torchvision.ops import box_iou
import random


def idx_to_sbjobj(idx: torch.Tensor, num_boxes: int):
    tmp_idx = idx + (idx // num_boxes) + 1
    sbj = tmp_idx // num_boxes
    obj = tmp_idx - (sbj * num_boxes)
    return torch.stack((sbj, obj), dim=1)


def _sbjobj_to_idx(sbj: torch.Tensor, obj: torch.Tensor, num_boxes: int):
    """Self-relations are not allowed!"""
    assert (sbj != obj).all()
    return sbj * (num_boxes - 1) + obj - (sbj < obj).long()


def _get_num_boxes(boxes: Union[int, torch.Tensor]):
    if isinstance(boxes, int):
        return boxes
    if isinstance(boxes, torch.Tensor):
        return len(boxes)
    raise ValueError(boxes)


def _get_all_negative_pair_ids(
    boxes: Union[int, torch.Tensor], rel_targets: torch.Tensor
):
    num_boxes = _get_num_boxes(boxes)

    # select all pairs
    neg_pair_mask = torch.ones((num_boxes * (num_boxes - 1),), dtype=torch.bool)

    num_positives = len(rel_targets)
    # positive pairs must not be sampled
    if num_positives > 0:
        sbj = rel_targets[:, 0]
        obj = rel_targets[:, 1]
        pos_ids = _sbjobj_to_idx(sbj, obj, num_boxes)
        neg_pair_mask[pos_ids] = False

    # the remaining entries are valid pairs for negative sampling
    return torch.nonzero(neg_pair_mask)[:, 0]


def sample_one_negative_pair(
    boxes: Union[int, torch.Tensor], rel_targets: torch.Tensor
):
    """Samples one single negative pair"""
    num_boxes = _get_num_boxes(boxes)
    neg_pair_ids = _get_all_negative_pair_ids(boxes, rel_targets)
    idx = random.choice(neg_pair_ids).unsqueeze(0)
    return idx_to_sbjobj(idx, num_boxes)[0]


def sample_negatives(
    boxes: Union[int, torch.Tensor], rel_targets: torch.Tensor, neg_ratio: float
) -> torch.Tensor:
    """
    :param num_boxes: Number of boxes for the processed image
    :param rel_targets: List of relations as they come out of LoadAnnotation
    (sbj, obj, rel0, rel1, ...)
    :param neg_ratio: The sampling ratio
    """
    neg_pair_ids = _get_all_negative_pair_ids(boxes, rel_targets)
    num_boxes = _get_num_boxes(boxes)

    num_pos = len(rel_targets)
    assert num_pos > 0, num_pos
    num_neg = round(neg_ratio * num_pos)
    num_neg = min(num_neg, len(neg_pair_ids))
    neg_ids = neg_pair_ids[torch.randperm(len(neg_pair_ids))[:num_neg]]

    # convert ids back to sbj/obj
    neg_sbjobj = idx_to_sbjobj(neg_ids, num_boxes)

    neg_targets = torch.cat(
        (
            neg_sbjobj,
            torch.zeros((num_neg, rel_targets.shape[1] - 2), dtype=rel_targets.dtype),
        ),
        dim=1,
    )
    return torch.cat((rel_targets, neg_targets))


def sample_all(num_boxes: int, num_outputs: int, rel_targets=None):
    neg_targets = torch.zeros(
        (num_boxes * (num_boxes - 1), num_outputs), dtype=torch.long
    )
    all_sbjobj = idx_to_sbjobj(torch.arange(num_boxes * (num_boxes - 1)), num_boxes)

    if rel_targets is not None and len(rel_targets) > 0:
        # write existing values back
        pos_sbj = rel_targets[:, 0]
        pos_obj = rel_targets[:, 1]
        neg_targets[_sbjobj_to_idx(pos_sbj, pos_obj, num_boxes)] = rel_targets[:, 2:]

    return torch.cat((all_sbjobj, neg_targets), dim=1)
