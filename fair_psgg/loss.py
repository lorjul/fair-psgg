from typing import Literal, Optional
import torch
from torch import Tensor
import torch.nn as nn


def get_node_criterion(num_samples: torch.Tensor):
    inv = 1 / num_samples.float()
    class_weights = inv
    class_weights = inv / inv.mean()
    return nn.CrossEntropyLoss(weight=class_weights)


def get_multi_rel_criterion(pos_neg_ratios: torch.Tensor, log=False):
    if log:
        inv = (1 / pos_neg_ratios + 1).log()
    else:
        inv = 1 / pos_neg_ratios
    # pos_weights = inv / inv.mean()
    pos_weights = inv
    assert (pos_weights >= 0).all()
    return BCEWithLogitsLossInterpolate(pos_weight=pos_weights)


class BCEWithLogitsLossInterpolate(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
        self._final_pos_weight = self.pos_weight

    def interpol_weights(self, t: float):
        t = max(0, min(1, t))
        self.pos_weight = (
            (1 - t) * torch.ones_like(self._final_pos_weight)
            + t * self._final_pos_weight
        ).to(self.pos_weight.device)


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


class RelRankLoss(nn.Module):
    def forward(
        self, input: torch.Tensor, rel_target: torch.Tensor, img_id: torch.Tensor
    ):
        assert len(rel_target) == len(img_id)
        assert input.shape == rel_target.shape

        losses = []
        # TODO: get rid of for-loop (probably use scatter_reduce and convert img_id to 2D mask)
        for i in img_id.unique(sorted=False):
            group = img_id == i

            x = input[group].flatten()
            y = rel_target[group].flatten().bool()

            neg_ids = (~y).nonzero()[:, 0]
            neg_ids = torch.randperm(len(neg_ids))[: y.sum()]
            neg_mask = torch.zeros_like(y)
            neg_mask[neg_ids] = True
            neg_mask[y] = True

            if y.sum() == 0:
                losses.append(0 * input.sum())
                continue

            pos = x[y]
            both = x[neg_mask]

            l = -(pos.exp().sum() / both.exp().sum()).log()
            losses.append(l)

        if torch.isnan(torch.stack(losses).mean()):
            torch.save(
                {
                    "in": input,
                    "t": rel_target,
                    "id": img_id,
                    "l": torch.stack(losses).mean(),
                },
                "/tmp/losses.pth",
            )
            raise RuntimeError()

        return torch.stack(losses).mean()
