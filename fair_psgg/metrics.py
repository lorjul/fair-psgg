from typing import Sequence
import torch


def _recall_k_counts(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    """A variant of the mean Recall@k metric for our model.
    To select the top-k predictions, our model can use the no-relation class output.
    This is different to other methods and therefore, we have to calculate the metric slightly different.
    """

    num_classes = output_list[0].size(1) - 1
    # count all annotations per class here (except no-relation class)
    gt_counts = torch.zeros(num_classes, dtype=torch.long)
    # count all hits per class (except no-relation class)
    hit_counts = torch.zeros(num_classes, dtype=torch.long)
    for y, x in zip(gt_list, output_list):
        gt_counts += y[:, 1:].sum(dim=0).long()

        # select the top-k predictions based on the no-relation class
        # low score means high likelihood of an annotated relation (torch.argsort is lowest first)
        ordering = torch.argsort(x[:, 0])[:k]
        # use argmax to select the highest score for the given relation (not ideal :/)
        topk_pred = x[ordering, 1:].argmax(dim=-1)
        # in the ground truth (which is one-hot-encoded), retrieve the annotations (either 0 or 1)
        is_correct = y[ordering, topk_pred + 1].long()
        # increase the number of hits per class (topk_pred chooses the predicate class index, is_correct if it's a hit or a miss)
        hit_counts.scatter_add_(0, topk_pred, is_correct)

    return gt_counts, hit_counts


def mean_recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    per_class_recall = hit_counts / gt_counts
    assert not torch.isnan(per_class_recall).any(), per_class_recall
    return per_class_recall.mean().item()


def recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    return hit_counts.sum() / gt_counts.sum()


def _nogc_recall_k_counts(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    """A variant of the mean Recall@k with no graph constraint for out model"""
    num_classes = output_list[0].size(1) - 1
    gt_counts = torch.zeros(num_classes, dtype=torch.long)
    hit_counts = torch.zeros(num_classes, dtype=torch.long)
    for y, x in zip(gt_list, output_list):
        gt_counts += y[:, 1:].sum(dim=0).long()

        # TODO: how do I obtain a good predicate score here?
        fg_score = (1 - x[:, 0])[:, None]
        modx = fg_score * x[:, 1:]

        ordering = modx.flatten().argsort()[-k:]

        class_ids = torch.tile(torch.arange(len(gt_counts)), (len(y),))
        is_correct = y[:, 1:].flatten()[ordering].long()
        hit_counts.scatter_add_(0, class_ids[ordering], is_correct)

    return gt_counts, hit_counts


def nogc_recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _nogc_recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    return hit_counts.sum() / gt_counts.sum()


def mean_nogc_recall_k(
    k: int, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    gt_counts, hit_counts = _nogc_recall_k_counts(
        k=k, gt_list=gt_list, output_list=output_list
    )
    per_class_recall = hit_counts / gt_counts
    assert not torch.isnan(per_class_recall).any(), per_class_recall
    return per_class_recall.mean().item()


def _per_class(gts: torch.Tensor, hits: torch.Tensor):
    per_class = hits / gts
    assert not torch.isnan(per_class).any(), per_class
    return per_class


def build_rel_metrics_dict(
    rel_names, gt_list: Sequence[torch.Tensor], output_list: Sequence[torch.Tensor]
):
    r20_g, r20_o = _recall_k_counts(k=20, gt_list=gt_list, output_list=output_list)
    r50_g, r50_o = _recall_k_counts(k=50, gt_list=gt_list, output_list=output_list)
    n20_g, n20_o = _nogc_recall_k_counts(k=20, gt_list=gt_list, output_list=output_list)
    n50_g, n50_o = _nogc_recall_k_counts(k=50, gt_list=gt_list, output_list=output_list)

    recall20_classes = _per_class(r20_g, r20_o)
    recall50_classes = _per_class(r50_g, r50_o)
    nogc20_classes = _per_class(n20_g, n20_o)
    nogc50_classes = _per_class(n50_g, n50_o)

    assert len(rel_names) == len(recall20_classes)

    metrics = {
        "rel_recall/20": r20_o.sum() / r20_g.sum(),
        "rel_recall/50": r50_o.sum() / r50_g.sum(),
        "rel_mean_recall/20": recall20_classes.mean(),
        "rel_mean_recall/50": recall50_classes.mean(),
        "rel_nogc_recall/20": n20_o.sum() / n20_g.sum(),
        "rel_nogc_recall/50": n50_o.sum() / n50_g.sum(),
        "rel_mean_nogc_recall/20": nogc20_classes.mean(),
        "rel_mean_nogc_recall/50": nogc50_classes.mean(),
    }

    for name, v in zip(rel_names, recall50_classes):
        metrics[f"rel_class_recall/50-{name}"] = v
    for name, v in zip(rel_names, nogc50_classes):
        metrics[f"rel_class_nogc_recall/50-{name}"] = v

    return {k: v.item() for k, v in metrics.items()}
