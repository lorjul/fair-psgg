from typing import Sequence
from PIL import Image
import torch
from torch.nn.functional import one_hot
from .data import SGDataset
from .rel_sample import sample_one_negative_pair
from .common import get_generic_loader


class SingleRelDataset(SGDataset):
    def __init__(
        self,
        is_train: bool,
        entries: Sequence[dict],
        node_names: Sequence[str],
        rel_names: Sequence[str],
        img_dir,
        seg_dir,
        augmentations: Sequence,
        neg_ratio,
        allow_overlapping_negatives=True,
    ):
        super().__init__(
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

        num_rels = []
        has_rels_left = []
        for entry in entries:
            r = len(entry["relations"])
            b = len(entry["annotations"])
            num_rels.append(r)
            has_rels_left.append(b * (b - 1) > r)
        num_rels = torch.tensor(num_rels)
        has_rels_left = torch.tensor(has_rels_left)
        # there are two options:
        # 1) sample the same number of negatives from every image
        # num_rels += round(num_rels.mean() * neg_ratio)
        # 2) sample the number of negatives relative to the number of positives
        num_rels[has_rels_left] += (num_rels[has_rels_left] * neg_ratio).round().long()
        self._cumlen = num_rels.cumsum(0)

    def __len__(self):
        return self._cumlen[-1]

    def __getitem__(self, idx):
        img_idx = (idx < self._cumlen).nonzero().min()
        if img_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self._cumlen[img_idx - 1]

        entry = self.entries[img_idx]
        img = self._open_img(entry["file_name"])
        seg = self._load_seg(entry["pan_seg_file_name"], entry["segments_info"])
        bboxes = torch.tensor([b["bbox"] for b in entry["annotations"]])

        for tfm in self.tfm:
            img, seg, bboxes = tfm(img, seg, bboxes)

        num_rel_outputs = len(self.rel_names)
        if local_idx >= len(entry["relations"]):
            # find a negative one
            pair_part = sample_one_negative_pair(
                boxes=bboxes,
                rel_targets=torch.tensor(entry["relations"]),
                allow_overlapping_negatives=self.allow_overlapping_negatives,
            )
            rel_part = torch.zeros((num_rel_outputs,), dtype=torch.long)
        else:
            sbj, obj, rel = entry["relations"][local_idx]
            pair_part = torch.tensor((sbj, obj))
            rel_part = one_hot(torch.tensor(rel), num_classes=num_rel_outputs)
        relation = torch.cat((pair_part, rel_part))

        return {
            "image_id": int(entry["image_id"]),
            "img": img,
            "bboxes": bboxes,
            "segmentation": seg,
            "box_categories": torch.tensor(
                [b["category_id"] for b in entry["annotations"]]
            ),
            "sampled_relations": relation.unsqueeze(0),
        }


def get_single_rel_loader(
    entries,
    node_names,
    rel_names,
    is_train,
    img_dir,
    seg_dir,
    batch_size,
    num_workers,
    augmentations,
    neg_ratio,
    allow_overlapping_negatives: bool,
):
    return get_generic_loader(
        dataset=SingleRelDataset(
            is_train=is_train,
            entries=entries,
            node_names=node_names,
            rel_names=rel_names,
            img_dir=img_dir,
            seg_dir=seg_dir,
            augmentations=augmentations,
            neg_ratio=neg_ratio,
            allow_overlapping_negatives=allow_overlapping_negatives,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=is_train,
    )
