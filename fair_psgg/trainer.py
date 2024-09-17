from pathlib import Path
from typing import Tuple
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .data.data import get_rel_loader
from .data.split_batch import split_batch_iter
from .metrics import build_rel_metrics_dict
from .config import Config
from . import from_config
from .utils import get_device, get_git_changes, get_git_commit
from .loss import get_node_criterion, get_multi_rel_criterion


class NoTensorboard:
    def add_scalar(self, *args, **kwargs):
        pass


def prepare_batch(
    batch: dict, device: torch.device
) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Takes a batch from data loader and returns the input to a model and the expected targets
    :param batch: The batch that comes directly from the data loader.
    :return: A 4-tuple of (model_input, sbj_target, obj_target, rel_target)
    """

    pair_ids_no_offset = batch["sampled_relations"][:, :2]
    # pair ids must be shifted by the number of boxes
    shifted_num_boxes = torch.cat((torch.tensor((0,)), batch["num_boxes"][:-1]))
    pair_offset = shifted_num_boxes.cumsum(dim=0)
    pair_ids = (
        pair_ids_no_offset
        + torch.repeat_interleave(pair_offset, batch["num_relations"])[:, None]
    )

    rel_target = batch["sampled_relations"][:, 2:]

    box_target = batch["box_categories"].to(device)
    sbj_target = box_target[pair_ids[:, 0]]
    obj_target = box_target[pair_ids[:, 1]]

    model_input = {
        "img": batch["img"].to(device),
        "bboxes": batch["bboxes"],
        "num_boxes": batch["num_boxes"],
        "pair_ids": pair_ids,
        "box_categories": box_target,
    }
    if "segmentation" in batch:
        model_input["segmentation"] = batch["segmentation"]

    return model_input, sbj_target, obj_target, rel_target.float()


class Trainer:
    def __init__(
        self,
        anno_path,
        img_dir,
        seg_dir,
        config: Config,
        out_dir=None,
        num_workers=0,
        dump_data=True,
        start_state_dict=None,
        hide_batch_progress=False,
    ):
        self.device = get_device()
        self.loss_weights = config.get_loss_weights()
        self.dump_data = dump_data
        self.grad_accumulate = config.grad_accumulate
        assert isinstance(self.grad_accumulate, int) and self.grad_accumulate > 0

        self.imgs_per_batch = config.batch_size
        self.rels_per_batch = config.rels_per_batch

        self._setup_loaders(config, anno_path, img_dir, seg_dir, num_workers)

        self.model = from_config.get_model(
            config,
            num_node_outputs=len(self.train_loader.dataset.node_names),
            # rel_names includes the NO-REL class
            num_rel_outputs=len(self.train_loader.dataset.rel_names),
        )
        if start_state_dict is not None:
            with torch.no_grad():
                self.model.load_state_dict(start_state_dict)
        self.model.to(self.device)

        self.node_criterion = get_node_criterion(
            self.train_loader.dataset.count_nodes()
        ).to(self.device)
        self.rel_criterion = get_multi_rel_criterion(
            self.train_loader.dataset.get_predicate_neg_ratio(),
            # only divide by log(number of classes)
            log=config.log_rel_class_weights,
        ).to(self.device)

        if config.lr_backbone is None:
            self.optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            extractor_params = []
            other_params = []
            for n, p in self.model.named_parameters():
                if "extractor" in n:
                    extractor_params.append(p)
                else:
                    other_params.append(p)

            optim_params = [{"params": other_params}]
            if config.lr_backbone > 0:
                optim_params.append(
                    {"params": extractor_params, "lr": config.lr_backbone}
                )
            self.optimizer = optim.AdamW(
                params=optim_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        self.lr_scheduler = from_config.get_lr_scheduler(config, self.optimizer)

        self.best_metric_value = None
        self.critical_metric = "rel_mean_recall/50"

        # for tensorboard
        self._global_step = 0

        if out_dir is None:
            self.out_dir = None
            self.tensorboard = NoTensorboard()
        else:
            self.out_dir = Path(out_dir)
            self.tensorboard = SummaryWriter(log_dir=out_dir)
            config.to_file(self.out_dir / "config.json")
            self.tensorboard.add_text("config", config.to_markdown())

            git_commit = get_git_commit()
            if git_commit:
                with open(self.out_dir / "commit.txt", "w") as f:
                    f.write(git_commit)
                changes = get_git_changes()
                if changes:
                    with open(self.out_dir / "changes.patch", "w") as f:
                        f.write(changes)

        self.hide_batch_progress = hide_batch_progress

    def _setup_loaders(self, config: Config, anno_path, img_dir, seg_dir, num_workers):
        train_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="train"
        )
        val_entries, node_names, rel_names = from_config.get_data_entries(
            config, anno_path=anno_path, split="val"
        )
        self.train_loader = get_rel_loader(
            entries=train_entries,
            node_names=node_names,
            rel_names=rel_names,
            img_dir=img_dir,
            seg_dir=seg_dir,
            is_train=True,
            batch_size=self.imgs_per_batch,
            num_workers=num_workers,
            augmentations=from_config.get_augmentations(config, split="train"),
            neg_ratio=config.neg_ratio,
            allow_overlapping_negatives=config.allow_overlapping_negatives,
        )

        self.val_loader = get_rel_loader(
            entries=val_entries,
            node_names=node_names,
            rel_names=rel_names,
            img_dir=img_dir,
            seg_dir=seg_dir,
            is_train=False,
            batch_size=self.imgs_per_batch,
            num_workers=num_workers,
            augmentations=from_config.get_augmentations(config, split="val"),
            # for validation, we want to look at all possible combinations
            allow_overlapping_negatives=True,
        )

    def _common_forward(self, batch):
        rel_loss_weight, node_loss_weight = self.loss_weights

        model_input, sbj_target, obj_target, rel_target = prepare_batch(
            batch, self.device
        )

        model_out = self.model(model_input)
        sbj_out, obj_out, rel_out = model_out

        # node loss
        sbj_loss = self.node_criterion(sbj_out, sbj_target)
        obj_loss = self.node_criterion(obj_out, obj_target)
        node_loss = node_loss_weight * (sbj_loss + obj_loss)

        # rel class loss
        rel_loss = rel_loss_weight * self.rel_criterion(
            rel_out, rel_target.to(self.device)
        )

        img_ids = torch.repeat_interleave(
            torch.arange(len(batch["img"])), batch["num_relations"]
        )

        loss = node_loss + rel_loss

        if torch.isnan(loss):
            if self.out_dir is not None:
                torch.save(
                    {
                        "rel_out": rel_out,
                        "sbj_out": sbj_out,
                        "obj_out": obj_out,
                        "rel_tgt": rel_target,
                        "sbj_tgt": sbj_target,
                        "obj_tgt": obj_target,
                        "img_ids": img_ids,
                    },
                    self.out_dir / "dbg-nanloss.pth",
                )
            raise RuntimeError("NaN loss")

        return {
            "loss": loss,
            "node_loss": node_loss.detach().cpu().clone(),
            "rel_loss": rel_loss.detach().cpu().clone(),
            "sbj_target": sbj_target,
            "obj_target": obj_target,
            "sbj_out": sbj_out,
            "obj_out": obj_out,
            "rel_target": rel_target,
            "rel_out": rel_out,
        }

    @torch.inference_mode()
    def evaluate(self, epoch: int):
        self.model.eval()

        node_losses = []
        rel_losses = []
        final_losses = []

        # torch.cat is required at the end
        # it can also be the case that some images appear twice in the list
        all_node_targets = []
        all_node_outputs = []

        all_rel_targets = defaultdict(list)
        all_rel_outputs = defaultdict(list)

        batch_iterator = split_batch_iter(
            tqdm(
                self.val_loader,
                leave=False,
                desc="eval",
                dynamic_ncols=True,
                disable=self.hide_batch_progress,
            ),
            max_relations=self.rels_per_batch,
        )
        for batch in batch_iterator:
            fwd = self._common_forward(batch)

            # node output
            all_node_targets.append(fwd["sbj_target"].cpu().clone())
            all_node_targets.append(fwd["obj_target"].cpu().clone())
            all_node_outputs.append(fwd["sbj_out"].argmax(dim=-1).cpu().clone())
            all_node_outputs.append(fwd["obj_out"].argmax(dim=-1).cpu().clone())

            # rel output
            img_ids = torch.repeat_interleave(
                torch.arange(len(batch["img"])), batch["num_relations"]
            )
            cpu_rel_target = fwd["rel_target"].cpu().clone()
            cpu_rel_out = fwd["rel_out"].sigmoid().cpu().clone()
            for i in range(len(batch["img"])):
                raw_img_id = batch["image_id"][i].item()
                all_rel_targets[raw_img_id].append(cpu_rel_target[img_ids == i])
                all_rel_outputs[raw_img_id].append(cpu_rel_out[img_ids == i])

            final_losses.append(fwd["loss"].cpu().clone())
            node_losses.append(fwd["node_loss"])
            rel_losses.append(fwd["rel_loss"])

        # calculate per class accuracy
        all_node_targets = torch.cat(all_node_targets)
        all_node_outputs = torch.cat(all_node_outputs)

        def _merge(keys, d):
            out = []
            for k in keys:
                v = d[k]
                if len(v) == 1:
                    out.append(v[0])
                else:
                    out.append(torch.cat(v))
            return out

        all_img_ids = list(all_rel_targets.keys())
        all_rel_targets = _merge(all_img_ids, all_rel_targets)
        all_rel_outputs = _merge(all_img_ids, all_rel_outputs)
        per_class_node_acc = confusion_matrix(
            all_node_targets,
            all_node_outputs,
            normalize="true",
        ).diagonal()

        metrics = {
            "epoch_loss/val/sum": torch.tensor(final_losses).mean().item(),
            "epoch_loss/val/node": torch.tensor(node_losses).mean().item(),
            "epoch_loss/val/rel": torch.tensor(rel_losses).mean().item(),
            "node_acc/mean": float(per_class_node_acc.mean()),
        }

        metrics.update(
            build_rel_metrics_dict(
                self.val_loader.dataset.rel_names[1:],
                gt_list=all_rel_targets,
                output_list=all_rel_outputs,
            )
        )

        for name, acc in zip(self.val_loader.dataset.node_names, per_class_node_acc):
            metrics[f"node_class_acc/{name}"] = acc

        data = {
            "node_targets": all_node_targets,
            "node_outputs": all_node_outputs,
            "rel_targets": all_rel_targets,
            "rel_outputs": all_rel_outputs,
            "rel_img_ids": all_img_ids,
        }

        return metrics, data

    def train_one_epoch(self, epoch: int):
        self.model.train()

        batch_iterator = tqdm(
            self.train_loader,
            leave=False,
            desc="train",
            dynamic_ncols=True,
            disable=self.hide_batch_progress,
        )
        batch_iterator = split_batch_iter(
            batch_iterator,
            max_relations=self.rels_per_batch,
        )

        self.optimizer.zero_grad()
        for bi, batch in enumerate(batch_iterator):
            fwd = self._common_forward(batch)

            loss = fwd["loss"] / self.grad_accumulate
            loss.backward()
            if (bi + 1) % self.grad_accumulate == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.tensorboard.add_scalar(
                "loss/train/sum",
                loss.detach().cpu().item() * self.grad_accumulate,
                global_step=self._global_step,
            )
            for sub in ("node", "rel"):
                self.tensorboard.add_scalar(
                    f"loss/train/{sub}",
                    fwd[f"{sub}_loss"],
                    global_step=self._global_step,
                )
            self._global_step += 1

        # don't really know if this helps with reducing VRAM usage but it won't harm
        del (batch, loss, fwd)

    def run(self, epochs):
        for epoch in tqdm(range(epochs)):
            metrics = self.one_epoch(epoch)

    def one_epoch(self, epoch):
        self.train_one_epoch(epoch)

        metrics, data = self.evaluate(epoch)
        for key, value in metrics.items():
            self.tensorboard.add_scalar(key, value, global_step=epoch)

        crit_value = metrics[self.critical_metric]

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            # show learning rate during training
            for i, pg in enumerate(self.optimizer.param_groups):
                self.tensorboard.add_scalar(f"_lr/{i}", pg["lr"], global_step=epoch)

        is_best_epoch = False
        if self.best_metric_value is None or crit_value > self.best_metric_value:
            self.best_metric_value = crit_value
            is_best_epoch = True

        if self.out_dir and self.dump_data:
            model_data = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "metric": crit_value,
            }

            torch.save(model_data, self.out_dir / "last_state.pth")

            if is_best_epoch:
                torch.save(model_data, self.out_dir / "best_state.pth")
                torch.save(metrics, self.out_dir / "best_metrics.pth")

        return metrics
