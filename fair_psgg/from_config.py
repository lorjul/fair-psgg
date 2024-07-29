# this file contains methods that read a config object and return a structure based on the read config
# this file exists such that config.py has no dependencies on the rest of the code
# and such that no other low-level file has a dependency on config.py
from pathlib import Path
import torch
import torch.optim.lr_scheduler as lrs
from .config import (
    Config,
    ExtractorCfg_ConvNeXt,
    ExtractorCfg_Dinov2,
    ExtractorCfg_FasterRCNN,
    ExtractorCfg_HRNet,
    ExtractorCfg_Mask2Former,
    ExtractorCfg_ResNet,
    LRScheduleStep,
)
from .models.feature_extractors import (
    FasterRCNNExtractor,
    Mask2FormerExtractor,
    MergedMask2FormerExtractor,
    FullMask2FormerExtractor,
    GenericTimmExtractor,
    Dinov2Extractor,
    ResNetExtractor,
)
from .models.daniformer import DaniFormer
from .data.augmentation import config_to_aug, get_standard_transforms
from .data.load_entries import (
    load_psg_entries,
    load_objects365_entries,
    load_psgcoco_entries,
    load_visual_genome_entries,
)


def get_extractor(extractor_cfg):
    if isinstance(extractor_cfg, ExtractorCfg_FasterRCNN):
        return FasterRCNNExtractor(
            feature_key=extractor_cfg.stage, architecture=extractor_cfg.backbone
        )
    elif isinstance(extractor_cfg, ExtractorCfg_ResNet):
        return ResNetExtractor(name=extractor_cfg.name)
    elif isinstance(extractor_cfg, ExtractorCfg_Mask2Former):
        if extractor_cfg.features == "encoder":
            return Mask2FormerExtractor(checkpoint_name=extractor_cfg.checkpoint)
        elif extractor_cfg.features == "decoder":
            return FullMask2FormerExtractor(checkpoint_name=extractor_cfg.checkpoint)
        elif extractor_cfg.features == "decoder_merged":
            return MergedMask2FormerExtractor(checkpoint_name=extractor_cfg.checkpoint)
        else:
            raise KeyError(extractor_cfg.sub_type)
    elif isinstance(extractor_cfg, ExtractorCfg_HRNet):
        return GenericTimmExtractor(
            model_name=f"hrnet_{extractor_cfg.variant}", feature_index=1
        )
    elif isinstance(extractor_cfg, ExtractorCfg_ConvNeXt):
        return GenericTimmExtractor(
            model_name=f"convnext_{extractor_cfg.variant}", feature_index=1
        )
    elif isinstance(extractor_cfg, ExtractorCfg_Dinov2):
        return Dinov2Extractor(variant=extractor_cfg.variant)
    raise KeyError(extractor_cfg)


def get_model(config: Config, num_node_outputs: int, num_rel_outputs: int):
    arch_cfg = config.architecture
    extractor = get_extractor(config.extractor)
    # TODO: don't hardcode width and height
    with torch.inference_mode():
        # TODO: don't hardcode image size
        tmp = extractor(torch.rand(1, 3, 640, 640))
    feature_shape = tmp.shape[1:]
    del tmp
    # feature_shape = (extractor.feature_channels, 160, 160)

    return DaniFormer(
        num_node_outputs=num_node_outputs,
        num_rel_outputs=num_rel_outputs,
        extractor=extractor,
        transformer_depth=arch_cfg.transformer_depth,
        embed_dim=arch_cfg.embed_dim,
        patch_size=arch_cfg.patch_size,
        feature_shape=feature_shape,
        use_semantics=arch_cfg.use_semantics,
        use_masks=arch_cfg.use_masks,
        bg_ratio_strategy=arch_cfg.bg_ratio_strategy,
        concat_mask=arch_cfg.concat_mask,
        normalize_tokens=arch_cfg.norm_tokens,
        use_patch_tokens_for_node=arch_cfg.use_patch_tokens_for_node,
        encode_coords=arch_cfg.encode_coords,
        use_classification_token=arch_cfg.classification_token,
    )


def get_augmentations(config: Config, split: str):
    if config is None or config.augmentations is None:
        return get_standard_transforms(is_train=split == "train")
    return [config_to_aug(c) for c in config.augmentations.get_list(split)]


def get_data_entries(config: Config, anno_path, split, img_dir=None):
    if config.data.source == "psg":
        return load_psg_entries(anno_path=anno_path, split=split)
    if config.data.source == "o365":
        assert split in ("train", "val")
        return load_objects365_entries(
            anno_path=f"{anno_path}{split}.json", img_dir=Path(img_dir) / split
        )
    if config.data.source == "psg-coco":
        assert split in ("train", "val")
        anno_path = Path(anno_path)
        return load_psgcoco_entries(
            anno_path=anno_path / "annotations" / f"panoptic_{split}2017.json",
            img_prefix=f"{split}2017",
            seg_prefix=f"panoptic_{split}2017",
        )
    if config.data.source == "vg-ietrans":
        anno_path = Path(anno_path)
        assert anno_path.is_dir()
        return load_visual_genome_entries(
            img_data_path=anno_path / "image_data.json",
            # we don't use the 1000 classes
            meta_path=anno_path / "50" / "VG-SGG-dicts-with-attri.json",
            h5_path=anno_path / "50" / "VG-SGG-with-attri.h5",
            split=split,
        )

    raise ValueError(f"Unsupported dataset: {config.data.source}")


def get_lr_scheduler(config: Config, optimizer):
    if config.lr_schedule is None:
        return None
    if isinstance(config.lr_schedule, LRScheduleStep):
        return lrs.StepLR(
            optimizer=optimizer,
            step_size=config.lr_schedule.step_size,
            gamma=config.lr_schedule.factor,
        )
    raise ValueError("Unsupported lr_schedule config")
