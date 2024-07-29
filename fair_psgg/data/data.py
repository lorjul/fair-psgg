from typing import Iterable, Sequence
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import torch
from PIL import Image
from collections import defaultdict

from ..utils import open_segmask
from .rel_sample import sample_all, sample_negatives
from .common import get_generic_loader
from .load_entries import load_psg_entries


def make_multilabel_target(singlelabel_target, num_classes: int):
    if len(singlelabel_target) == 0:
        return torch.zeros((0, 2 + num_classes), dtype=torch.long)

    # convert to multi label
    multi_dict = defaultdict(list)
    for sbj, obj, rel in singlelabel_target:
        # revert the 1-offset (we don't have a NONE class anymore)
        multi_dict[(sbj, obj)].append(rel - 1)

    # build a normal tensor out of it
    sbj_obj = []
    multi_rels = []
    for (sbj, obj), rels in multi_dict.items():
        sbj_obj.append((sbj, obj))
        one_hot = torch.zeros((num_classes,), dtype=torch.long)
        for r in rels:
            one_hot[r] = 1
        multi_rels.append(one_hot)

    return torch.cat((torch.tensor(sbj_obj), torch.stack(multi_rels)), dim=1)


def count_predicates(entries, num_rel_outputs: int):
    counts = torch.zeros(num_rel_outputs - 1, dtype=torch.long)
    for entry in entries:
        for _, _, rel in entry["relations"]:
            assert rel > 0, rel
            counts[rel - 1] += 1
    return counts


def count_nodes(entries, num_node_outputs: int):
    counts = torch.zeros(num_node_outputs, dtype=torch.long)
    for entry in entries:
        for box in entry["annotations"]:
            counts[box["category_id"]] += 1
    return counts


def get_predicate_neg_ratio(entries, num_rel_outputs: int, neg_ratio):
    pos = torch.zeros(num_rel_outputs, dtype=torch.long)
    neg = torch.zeros(num_rel_outputs, dtype=torch.long)
    for entry in entries:
        this_rel = torch.zeros(num_rel_outputs, dtype=torch.bool)
        for _, _, rel in entry["relations"]:
            assert rel > 0
            this_rel[rel - 1] = 1

        pos += this_rel
        neg += ~this_rel

    ratio = pos.float() / neg.float()

    assert neg_ratio is not None
    # explicit none is more of a hack currently
    return torch.cat((torch.tensor((neg_ratio,)), ratio[:-1]))


class SGConcatDataset(ConcatDataset):
    """Subclass of `ConcatDataset` that also contains the `node_names` and `rel_names` properties.
    This dataset class is intended for semi-supervised learning.
    If you want to access the `count_nodes()`, `count_predicates()`, or `get_predicate_neg_ratio()` functions,
    call them directly on one of the child datasets.
    """

    def __init__(self, datasets: Iterable["SGDataset"]):
        super().__init__(datasets)
        nn = self.datasets[0].node_names
        rn = self.datasets[0].rel_names
        for d in self.datasets:
            assert d.node_names == nn
            assert d.rel_names == rn

    @property
    def node_names(self):
        return self.datasets[0].node_names

    @property
    def rel_names(self):
        return self.datasets[0].rel_names


class SGDataset(Dataset):
    def __init__(
        self,
        is_train: bool,
        entries: Sequence[dict],
        node_names: Sequence[str],
        rel_names: Sequence[str],
        img_dir,
        seg_dir,
        augmentations: Sequence,
        neg_ratio=None,
        allow_overlapping_negatives=True,
    ):
        """Creates a new SGDataset object
        :param is_train: Whether the dataset is for training data or not.
            In training mode, different augmentation and sampling is used.
        :param entries: Sequence of dictionaries that follow the PSG file format (data key).
            Per entry dict, the following keys are required: file_name, annotations, image_id.
        :param node_names: Class names of the node categories.
            For PSG, you get them from `thing_classes` and `stuff_classes`.
        :param rel_names: Sequence of predicate class names. This must not include a no-relation
            class name (will be added in the dataset).
        :param img_dir: Path to image directory such that `{img_dir}/{entry["file_name"]}` works.
        :param seg_dir: Path to segmentation masks directory such that `{seg_dir}/{entry["pan_seg_file_name"]}` works.
        :param neg_ratio: Negative ratio for random sampling. Only applies when `is_train == True`.
        """
        self.img_dir = Path(img_dir)
        self.seg_dir = Path(seg_dir)
        self.node_names = node_names
        self.rel_names = ["NONE"] + rel_names
        self.entries = entries
        self.allow_overlapping_negatives = allow_overlapping_negatives

        assert len(self.entries) > 0, "Empty dataset"

        self.tfm = augmentations

        if is_train:
            assert isinstance(neg_ratio, (int, float)), neg_ratio
            self.neg_ratio = neg_ratio
        else:
            self.neg_ratio = None

    def __len__(self):
        return len(self.entries)

    def _open_img(self, file_name):
        # this function is separate because RndDataset has to override it (for testing)
        return Image.open(self.img_dir / file_name).convert("RGB")

    def _load_seg(self, pan_seg_file_name, segments_info):
        seg_classes = torch.from_numpy(open_segmask(self.seg_dir / pan_seg_file_name))
        seg_ids = torch.tensor([info["id"] for info in segments_info])
        seg_masks = seg_classes == seg_ids[:, None, None]
        return seg_masks

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = self._open_img(entry["file_name"])
        bboxes = torch.tensor([b["bbox"] for b in entry["annotations"]])
        seg_masks = self._load_seg(entry["pan_seg_file_name"], entry["segments_info"])

        for tfm in self.tfm:
            img, seg_masks, bboxes = tfm(img, seg_masks, bboxes)

        raw_relations = entry.get("relations", [])

        num_outputs = len(self.rel_names) - 1
        multi_targets = make_multilabel_target(raw_relations, num_classes=num_outputs)

        if self.neg_ratio is None:
            sampled_targets = sample_all(
                num_boxes=len(bboxes),
                num_outputs=num_outputs,
                rel_targets=multi_targets,
            )
        else:
            sampled_targets = sample_negatives(
                boxes=bboxes,
                rel_targets=multi_targets,
                neg_ratio=self.neg_ratio,
                allow_overlapping_negatives=self.allow_overlapping_negatives,
            )

        # add the explicit label
        expl_none_label = torch.zeros(
            (sampled_targets.size(0), 1), dtype=sampled_targets.dtype
        )
        expl_none_label[(sampled_targets[:, 2:] == 0).all(-1)] = 1
        sampled_targets = torch.cat(
            (sampled_targets[:, :2], expl_none_label, sampled_targets[:, 2:]),
            dim=1,
        )

        return {
            "idx": idx,
            "image_id": int(entry["image_id"]),
            "img": img,
            "bboxes": bboxes,
            "segmentation": seg_masks,
            "box_categories": torch.tensor(
                [b["category_id"] for b in entry["annotations"]]
            ),
            "sampled_relations": sampled_targets,
        }

    def __add__(self, other):
        if isinstance(other, SGDataset):
            return SGConcatDataset([self, other])
        return super().__add__(other)

    def count_nodes(self):
        return count_nodes(self.entries, len(self.node_names))

    def count_predicates(self):
        return count_predicates(self.entries, len(self.rel_names))

    def get_predicate_neg_ratio(self):
        return get_predicate_neg_ratio(
            self.entries, len(self.rel_names), self.neg_ratio
        )


def get_rel_loader(
    entries,
    node_names,
    rel_names,
    img_dir,
    seg_dir,
    is_train,
    augmentations,
    batch_size,
    num_workers,
    neg_ratio=None,
    allow_overlapping_negatives=True,
):
    dataset = SGDataset(
        is_train=is_train,
        entries=entries,
        node_names=node_names,
        rel_names=rel_names,
        img_dir=img_dir,
        seg_dir=seg_dir,
        augmentations=augmentations,
        neg_ratio=neg_ratio,
        allow_overlapping_negatives=allow_overlapping_negatives,
    )
    return get_generic_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=is_train,
    )


def get_loader(
    anno_path,
    split,
    img_dir,
    seg_dir,
    batch_size: int,
    num_workers: int,
    augmentations,
    neg_ratio=None,
    allow_overlapping_negatives=True,
):
    entries, node_names, rel_names = load_psg_entries(anno_path=anno_path, split=split)
    is_train = split == "train"
    dataset = SGDataset(
        is_train=is_train,
        entries=entries,
        node_names=node_names,
        rel_names=rel_names,
        img_dir=img_dir,
        seg_dir=seg_dir,
        augmentations=augmentations,
        neg_ratio=neg_ratio,
        allow_overlapping_negatives=allow_overlapping_negatives,
    )
    return get_generic_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=is_train,
    )
