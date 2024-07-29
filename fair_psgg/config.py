import json
from typing import Literal, Optional, Sequence, Union
from pydantic import BaseModel, Field


class ExtractorCfg_FasterRCNN(BaseModel):
    type: Literal["fasterrcnn"] = "fasterrcnn"
    stage: Literal["0", "1", "2", "3", "all"] = "0"
    backbone: Literal["resnet50", "mobilenet"] = "resnet50"


class ExtractorCfg_ResNet(BaseModel):
    type: Literal["resnet"] = "resnet"
    name: str = "resnet101"


class ExtractorCfg_Mask2Former(BaseModel):
    type: Literal["mask2former"] = "mask2former"
    checkpoint: str = "facebook/mask2former-swin-base-coco-panoptic"
    features: Literal["encoder", "decoder", "decoder_merged"] = "encoder"


class ExtractorCfg_Dinov2(BaseModel):
    type: Literal["dinov2"] = "dinov2"
    variant: Literal["s", "b", "l", "g"] = "s"


class ExtractorCfg_HRNet(BaseModel):
    type: Literal["hrnet"] = "hrnet"
    variant: Literal["w32"] = "w32"


class ExtractorCfg_ConvNeXt(BaseModel):
    type: Literal["convnext"] = "convnext"
    variant: str = "large.fb_in22k_ft_in1k_384"


class _Augmentations(BaseModel):
    train: Sequence[Union[dict, str]]
    val: Sequence[Union[dict, str]]

    def get_list(self, split: str):
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        raise KeyError(split)


class ArchCfg_DaniFormer(BaseModel):
    type: Literal["daniformer"] = "daniformer"
    transformer_depth: int = 6
    embed_dim: int = 384
    patch_size: int = 8
    use_semantics: bool = False
    use_masks: bool = False
    bg_ratio_strategy: Literal["total", "indiv", "sum1", "onoff"] = "total"
    concat_mask: Union[bool, Literal["ignore"]] = False
    norm_tokens: bool = False
    use_patch_tokens_for_node: bool = False
    encode_coords: bool = False
    classification_token: bool = True


class DataCfg(BaseModel):
    source: Literal["psg", "o365", "vg-ietrans", "psg-coco"] = "psg"


class LossCfg(BaseModel):
    rank_weight: float = 0.0
    interpolate_weights: Optional[int] = None


class LRScheduleStep(BaseModel):
    type: Literal["step"] = "step"
    step_size: int = 10
    factor: float = 0.1


class Config(BaseModel):
    rel_weight: float = 0.8
    node_loss_weight: Optional[float] = None
    batch_size: int = 32
    rels_per_batch: int = 4096
    lr: float = 0.001
    # set to None to use same lr as config.lr
    lr_backbone: Optional[float] = None
    lr_schedule: Union[LRScheduleStep, None] = None
    weight_decay: float = 0.01
    neg_ratio: float = 1.0
    augmentations: Optional[_Augmentations] = None
    single_rel_data: Optional[int] = None
    allow_overlapping_negatives: bool = True
    log_rel_class_weights: bool = False
    grad_accumulate: int = 1
    loss: Optional[LossCfg] = None

    extractor: Union[
        ExtractorCfg_FasterRCNN,
        ExtractorCfg_Mask2Former,
        ExtractorCfg_Dinov2,
        ExtractorCfg_HRNet,
        ExtractorCfg_ConvNeXt,
        ExtractorCfg_ResNet,
    ] = Field(ExtractorCfg_FasterRCNN(), discriminator="type")
    architecture: ArchCfg_DaniFormer = ArchCfg_DaniFormer()

    data: DataCfg = DataCfg()

    def get_loss_weights(self):
        if self.node_loss_weight is None:
            n = 1 - self.rel_weight
        else:
            n = self.node_loss_weight
        if self.loss is None:
            rank_weight = 0.0
        else:
            rank_weight = self.loss.rank_weight
        return self.rel_weight, n, rank_weight

    def to_markdown(self):
        lines = [f"- {key}: {value}" for key, value in self.__dict__.items()]
        return "\n".join(lines)

    @staticmethod
    def from_file(path):
        with open(path) as f:
            data = json.load(f)
        return Config.model_validate(data, strict=True)

    def to_file(self, path):
        content = self.model_dump()
        with open(path, "w") as f:
            json.dump(content, f, indent=2)


def _write_schema():
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    schema_str = Config.model_json_schema()
    with open(args.output, "w") as f:
        json.dump(schema_str, f, indent=2)


if __name__ == "__main__":
    _write_schema()
