from typing import Literal, Optional
import torch
from torch import Tensor
import torch.nn as nn


def get_node_criterion(num_samples: torch.Tensor):
    inv = 1 / num_samples.float()
    class_weights = inv
    class_weights = inv / inv.mean()
    return nn.CrossEntropyLoss(weight=class_weights)


def get_multi_rel_criterion(pos_neg_ratios: torch.Tensor):
    pos_weights = 1 / pos_neg_ratios
    assert (pos_weights >= 0).all()
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


class BCEWithLogitsIgnoreNegative(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        pos_weight: Optional[Tensor] = None,
        collect_stats: Optional[int] = None,
    ) -> None:
        """
        :param collect_stats: Set this parameter to the number of target classes to count the
        number of positives/negatives until .reset_stats() is called.
        """
        assert reduction in ("none", "mean", "sum")
        super().__init__(weight=weight, reduction="none", pos_weight=pos_weight)
        self.my_reduction = reduction
        if collect_stats is not None:
            self.pos_count_stats = torch.zeros(collect_stats, dtype=torch.long)
            self.neg_count_stats = torch.zeros(collect_stats, dtype=torch.long)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # negative values indicate that they should be skipped
        if self.pos_count_stats is not None:
            assert target.size(1) == self.pos_count_stats.size(0)
            self.pos_count_stats += (target == 1).sum(0)
            self.neg_count_stats += (target == 0).sum(0)
        mask = target >= 0
        loss = super().forward(input, target.float()) * mask
        if self.my_reduction == "none":
            return loss
        if self.my_reduction == "sum":
            return loss.sum()
        if self.my_reduction == "mean":
            return loss.sum() / mask.sum()
        raise RuntimeError()

    def reset_stats(self):
        if self.pos_count_stats is not None:
            self.pos_count_stats[:] = 0
            self.neg_count_stats[:] = 0
