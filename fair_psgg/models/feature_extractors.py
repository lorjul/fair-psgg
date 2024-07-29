# contains different feature extractors
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from transformers import Mask2FormerForUniversalSegmentation
import timm


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._feature_channels = None

    @property
    def feature_channels(self):
        # if the number of output channels cannot be determined, make a forward pass through the model
        if self._feature_channels is None:
            # determine the number of features lazyly
            with torch.inference_mode():
                tmp = self.forward(torch.empty(1, 3, 128, 128))
                self._feature_channels = tmp.size(1)
        return self._feature_channels

    def forward(self, img: torch.Tensor):
        raise NotImplementedError()


class FasterRCNNExtractor(FeatureExtractor):
    def __init__(self, feature_key="0", architecture="resnet50"):
        super().__init__()
        if architecture == "resnet50":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            ).backbone
        elif architecture == "mobilenet":
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            ).backbone
        else:
            raise ValueError("Unknown architecture")
        self.feature_key = feature_key

        if feature_key == "all":
            self.feature_squasher = nn.Conv2d(
                in_channels=1024, out_channels=256, kernel_size=1
            )

    @property
    def feature_channels(self):
        # no need to pass a tensor through the backbone, information is available
        return self.model.out_channels

    def forward(self, img: torch.Tensor):
        features = self.model(img)

        if self.feature_key == "all":
            # resize all feature maps to the largest size
            target_shape = features["0"].shape[-2:]
            combined = torch.cat(
                [
                    features["0"],
                    F.interpolate(features["1"], target_shape),
                    F.interpolate(features["2"], target_shape),
                    F.interpolate(features["3"], target_shape),
                ],
                dim=1,
            )

            return self.feature_squasher(combined)

        return features[self.feature_key]


class ResNetExtractor(FeatureExtractor):
    def __init__(self, name: str, feature_key="0"):
        super().__init__()
        weights = {
            "resnet101": torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2,
            "resnet152": torchvision.models.resnet.ResNet152_Weights.IMAGENET1K_V2,
            "resnet50": torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
            "resnet18": torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": torchvision.models.resnet.ResNet34_Weights.IMAGENET1K_V1,
            "resnext50": torchvision.models.resnet.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
            "resnext101_64": torchvision.models.resnet.ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
            "resnext101_32": torchvision.models.resnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
        }
        self.model = resnet_fpn_backbone(backbone_name=name, weights=weights[name])
        self.feature_key = feature_key

    def forward(self, img: torch.Tensor):
        features = self.model(img)
        return features[self.feature_key]


class Mask2FormerExtractor(FeatureExtractor):
    def __init__(
        self,
        checkpoint_name="facebook/mask2former-swin-base-coco-panoptic",
        feature_key=1,
    ):
        super().__init__()
        mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_name
        )
        self.model = mask2former.model.pixel_level_module.encoder
        self.feature_key = feature_key

    def forward(self, img: torch.Tensor):
        features = self.model(img).feature_maps
        return features[self.feature_key]


class FullMask2FormerExtractor(FeatureExtractor):
    def __init__(self, checkpoint_name="facebook/mask2former-swin-base-coco-panoptic"):
        super().__init__()
        mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_name
        )
        self.model = mask2former.model.pixel_level_module

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img).decoder_last_hidden_state


class MergedMask2FormerExtractor(FeatureExtractor):
    def __init__(self, checkpoint_name="facebook/mask2former-swin-base-coco-panoptic"):
        super().__init__()
        mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint_name
        )
        self.model = mask2former.model.pixel_level_module

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        states = self.model(img).decoder_hidden_states
        target_size = states[-1].shape[-2:]
        reshaped = [F.interpolate(t, target_size) for t in states]
        return torch.stack(reshaped, dim=0).sum(dim=0)


class Dinov2Extractor(FeatureExtractor):
    def __init__(self, variant: Literal["s", "b", "l", "g"] = "s"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{variant}14")

    def forward(self, img: torch.Tensor):
        tokens: torch.Tensor = self.model.forward_features(img)["x_norm_patchtokens"]
        features = tokens.permute(0, 2, 1)
        features = features.reshape(
            features.size(0),
            features.size(1),
            img.size(2) // 14,
            img.size(3) // 14,
        )
        return features


class GenericTimmExtractor(FeatureExtractor):
    def __init__(self, model_name: str, feature_index: int):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(feature_index,),
        )

    def forward(self, img: torch.Tensor):
        # self.model returns a list of feature tensors for the batch
        # in the constructor, out_indices is set, therefore the length of the list is 1
        return self.model(img)[0]
