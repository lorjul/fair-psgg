# transformer architecture for node detection
# easier to validate if it actually works
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from .building_blocks import CoordEncoder
from .feature_extractors import FeatureExtractor
from .transformer_blocks import TransformerBlock, make_sine_position_encoding
from .frequency_bias import FreqBias
from ..utils import box_intersection


def bg_ratio_onoff_rest(sbj_ratios, obj_ratios):
    return torch.zeros_like(sbj_ratios)


def bg_ratio_total_rest(sbj_ratios, obj_ratios):
    return torch.max(torch.tensor(0.0), 1 - sbj_ratios - obj_ratios)


class SbjObjBoxEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        img_size: Tuple[int, int],
        bg_ratio_strategy: str,
        bg_token=True,
    ):
        super().__init__()
        assert bg_ratio_strategy in ("sum", "onoff")
        self.patch_size = patch_size
        self.sbj_token = nn.Parameter(torch.rand(embed_dim))
        self.obj_token = nn.Parameter(torch.rand(embed_dim))
        if bg_token:
            self.background_token = nn.Parameter(torch.rand(embed_dim))
        else:
            self.background_token = nn.Parameter(
                torch.zeros(embed_dim), requires_grad=False
            )
        if bg_ratio_strategy == "sum":
            self.get_bg_ratio = bg_ratio_total_rest
        elif bg_ratio_strategy == "onoff":
            self.get_bg_ratio = bg_ratio_onoff_rest
        else:
            raise ValueError()

        self.onoff = bg_ratio_strategy == "onoff"

        self.patch_coords = self.get_patch_boxes(img_size).reshape(-1, 4)

    def get_patch_boxes(self, img_size):
        rows, cols = torch.meshgrid(
            torch.arange(img_size[0] // self.patch_size),
            torch.arange(img_size[1] // self.patch_size),
            indexing="ij",
        )
        return torch.stack(
            (
                cols * self.patch_size,
                rows * self.patch_size,
                (cols + 1) * self.patch_size,
                (rows + 1) * self.patch_size,
            ),
            dim=-1,
        )

    def forward_ratios(self, sbj_boxes: torch.Tensor, obj_boxes: torch.Tensor):
        ratios = box_intersection(
            torch.cat((sbj_boxes, obj_boxes)), self.patch_coords
        ) / (self.patch_size * self.patch_size)
        ratios = ratios.to(self.sbj_token.device)
        sbj_ratios = ratios[: len(sbj_boxes)]
        obj_ratios = ratios[len(sbj_boxes) :]
        bg_ratios = self.get_bg_ratio(sbj_ratios, obj_ratios)
        return bg_ratios, sbj_ratios, obj_ratios

    def forward(self, sbj_boxes: torch.Tensor, obj_boxes: torch.Tensor):
        bg_ratios, sbj_ratios, obj_ratios = self.forward_ratios(sbj_boxes, obj_boxes)
        if self.onoff:
            sbj_ratios[sbj_ratios > 0] = 1
            obj_ratios[obj_ratios > 0] = 1
            bg_ratios[:] = 0.0
        return (
            bg_ratios[..., None] * normalize(self.background_token, dim=0)[None, None]
            + sbj_ratios[..., None] * normalize(self.sbj_token, dim=0)[None, None]
            + obj_ratios[..., None] * normalize(self.obj_token, dim=0)[None, None]
        )


class SbjObjMaskEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        bg_ratio_strategy: str,
        bg_token=True,
    ):
        super().__init__()
        assert bg_ratio_strategy in ("sum", "onoff")
        self.patch_size = patch_size
        self.sbj_token = nn.Parameter(torch.rand(embed_dim))
        self.obj_token = nn.Parameter(torch.rand(embed_dim))
        if bg_token:
            self.background_token = nn.Parameter(torch.rand(embed_dim))
        else:
            self.background_token = nn.Parameter(
                torch.zeros(embed_dim), requires_grad=False
            )

        if bg_ratio_strategy == "sum":
            self.get_bg_ratio = bg_ratio_total_rest
        elif bg_ratio_strategy == "onoff":
            self.get_bg_ratio = bg_ratio_onoff_rest
        else:
            raise ValueError()

        self.onoff = bg_ratio_strategy == "onoff"

    def forward_ratios(
        self,
        sbj_seg: torch.Tensor,
        obj_seg: torch.Tensor,
        feature_size: Tuple[int, int],
    ):
        scale_factor_y = sbj_seg.size(1) // feature_size[0]
        scale_factor_x = sbj_seg.size(2) // feature_size[1]
        seg = torch.cat((sbj_seg, obj_seg))
        y_split = torch.stack(
            torch.split(seg, self.patch_size * scale_factor_y, dim=1),
            dim=1,
        )
        x_split = torch.stack(
            torch.split(y_split, self.patch_size * scale_factor_x, dim=3),
            dim=2,
        )
        # calculate average pixel coverage for each mask and each patch
        ratios = x_split.float().flatten(-2).mean(-1)
        ratios = ratios.to(self.sbj_token.device)
        # move all patches to 1D
        ratios = ratios.flatten(-2)
        sbj_ratios = ratios[: len(sbj_seg)]
        obj_ratios = ratios[len(sbj_seg) :]
        bg_ratios = self.get_bg_ratio(sbj_ratios, obj_ratios)
        return bg_ratios, sbj_ratios, obj_ratios

    def forward(
        self,
        sbj_seg: torch.Tensor,
        obj_seg: torch.Tensor,
        feature_size: Tuple[int, int],
    ):
        bg_ratios, sbj_ratios, obj_ratios = self.forward_ratios(
            sbj_seg, obj_seg, feature_size
        )
        if self.onoff:
            sbj_ratios[sbj_ratios > 0] = 1
            obj_ratios[obj_ratios > 0] = 1
            bg_ratios[:] = 0.0
        return (
            bg_ratios[..., None] * normalize(self.background_token, dim=0)[None, None]
            + sbj_ratios[..., None] * normalize(self.sbj_token, dim=0)[None, None]
            + obj_ratios[..., None] * normalize(self.obj_token, dim=0)[None, None]
        )


class NodeClassifierPatchTokens(nn.Module):
    def __init__(self, embed_dim: int, num_node_outputs: int, num_patches: int):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=1)
        self.linear = nn.Linear(in_features=num_patches, out_features=num_node_outputs)

    def forward(self, tokens: torch.Tensor, ratios):
        _, sbj_ratios, obj_ratios = ratios
        patch_tokens = tokens[:, -sbj_ratios.size(1) :]
        sbj_tokens = self.conv1x1(
            (patch_tokens * sbj_ratios[..., None]).transpose(1, 2)
        )
        obj_tokens = self.conv1x1(
            (patch_tokens * obj_ratios[..., None]).transpose(1, 2)
        )

        sbj_cls = self.linear(sbj_tokens[:, 0])
        obj_cls = self.linear(obj_tokens[:, 0])

        return sbj_cls, obj_cls


class DaniFormer(nn.Module):
    def __init__(
        self,
        num_node_outputs: int,
        num_rel_outputs: int,
        extractor: FeatureExtractor,
        transformer_depth=6,
        embed_dim=384,
        patch_size=8,
        feature_shape=(256, 128, 128),
        final_hidden_dim_factor=2,
        use_semantics=False,
        use_masks=False,
        bg_ratio_strategy="total",
        encode_coords=False,
        use_patch_tokens_for_node=False,
        use_classification_token=True,
    ):
        super().__init__()
        self.extractor = extractor

        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.feature_shape = feature_shape
        self.use_masks = use_masks

        self.patch_embed = nn.Conv2d(
            in_channels=feature_shape[0],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=self.embed_dim, num_heads=8)
                for _ in range(transformer_depth)
            ]
        )

        # NOTE: usually, the hidden layer is increasing by 4
        final_hidden_dim = round(self.embed_dim * final_hidden_dim_factor)
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=final_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=final_hidden_dim, out_features=num_rel_outputs),
        )

        self.final_ranking_layers = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=final_hidden_dim),
            nn.ReLU(),
            # for ranking, we need one less output coefficient as for classification
            nn.Linear(in_features=final_hidden_dim, out_features=num_rel_outputs - 1),
        )

        if use_patch_tokens_for_node:
            self.final_node = NodeClassifierPatchTokens(
                embed_dim=self.embed_dim,
                num_node_outputs=num_node_outputs,
                num_patches=(self.feature_shape[1] // patch_size)
                * (self.feature_shape[2] // patch_size),
            )
        else:
            self.final_node = nn.Linear(
                in_features=self.embed_dim, out_features=num_node_outputs * 2
            )

        self.pos_encoding = nn.Parameter(
            make_sine_position_encoding(
                feature_shape[-2:], patch_size=self.patch_size, d_model=self.embed_dim
            ),
            requires_grad=False,
        )

        if self.use_masks:
            self.sbjobj_encoder = SbjObjMaskEncoder(
                embed_dim=self.embed_dim,
                patch_size=patch_size,
                bg_ratio_strategy=bg_ratio_strategy,
                bg_token=True,
            )
        else:
            self.sbjobj_encoder = SbjObjBoxEncoder(
                self.embed_dim,
                patch_size,
                feature_shape[1:],
                bg_ratio_strategy=bg_ratio_strategy,
                bg_token=True,
            )

        if encode_coords:
            self.coord_embed = CoordEncoder(self.embed_dim)
        else:
            self.coord_embed = None

        if use_classification_token:
            self.classification_token = nn.Parameter(torch.rand(self.embed_dim))
        else:
            self.classification_token = None

        if use_semantics:
            self.freq_bias = FreqBias(
                num_node_classes=num_node_outputs,
                num_rel_outputs=self.embed_dim,
            )
        else:
            self.freq_bias = None

    def _forward_internal(
        self,
        data: dict,
        features: torch.Tensor,
        patches: torch.Tensor,
        img_shape,
        sbj_ids: torch.Tensor,
        obj_ids: torch.Tensor,
        return_attention,
    ):
        num_boxes = data["num_boxes"]
        box_targets = data["box_categories"]

        img_ids = torch.repeat_interleave(num_boxes)[sbj_ids]
        patches_per_box = patches[img_ids]

        if self.use_masks:
            seg = data["segmentation"]
            sbjobj_tokens = self.sbjobj_encoder(
                seg[sbj_ids], seg[obj_ids], features.shape[-2:]
            )
        else:
            # scale coords to feature size
            h, w = img_shape[-2:]
            fh, fw = features.shape[1:3]
            coords = data["bboxes"]
            coords[:, 0] *= fw / w
            coords[:, 1] *= fh / h
            coords[:, 2] *= fw / w
            coords[:, 3] *= fh / h
            sbjobj_tokens = self.sbjobj_encoder(coords[sbj_ids], coords[obj_ids])

        if self.classification_token is None:
            extra_tokens = torch.empty(
                (patches_per_box.size(0), 0, self.embed_dim), device=patches.device
            )
        else:
            extra_tokens = self.classification_token[None, None].expand(
                patches_per_box.size(0), 1, -1
            )

        # add frequency bias token if requested
        if self.freq_bias is not None:
            bias = self.freq_bias(box_targets[sbj_ids], box_targets[obj_ids])
            extra_tokens = torch.cat((extra_tokens, bias[:, None]), dim=1)

        # add coord token if requested
        if self.coord_embed is not None:
            coords = data["bboxes"]
            coord_token = self.coord_embed(
                coords[sbj_ids].to(features.device),
                coords[obj_ids].to(features.device),
                img_shape[-2:],
            )
            extra_tokens = torch.cat((extra_tokens, coord_token[:, None]), dim=1)

        tokens = torch.cat((extra_tokens, patches_per_box + sbjobj_tokens), dim=1)
        # OR: add the tokens after every layer

        if return_attention is not None:
            for i, block in enumerate(self.transformer_blocks):
                if i == return_attention:
                    return block(
                        tokens, pos_encoding=self.pos_encoding, return_attention=True
                    )
                else:
                    tokens = block(tokens, pos_encoding=self.pos_encoding)
        else:
            for block in self.transformer_blocks:
                tokens = block(tokens, pos_encoding=self.pos_encoding)

        if self.classification_token is None:
            # average the output from all tokens
            final_token = tokens.mean(1)
        else:
            # use the output from the classification token
            final_token = tokens[:, 0]

        output = self.final_layers(final_token)

        if self.final_node is None:
            # this is for inference only, safe some memory and skip the node classification
            sbj_cls = None
            obj_cls = None
        elif isinstance(self.final_node, NodeClassifierPatchTokens):
            sbj_cls, obj_cls = self.final_node(
                tokens,
                self.sbjobj_encoder.forward_ratios(
                    seg[sbj_ids], seg[obj_ids], features.shape[-2:]
                ),
            )
        else:
            node_output = self.final_node(final_token)
            half_len = node_output.size(1) // 2
            sbj_cls = node_output[:, half_len:]
            obj_cls = node_output[:, :half_len]

        # relation ranking is probably useless and we don't want it to interfere with the rest of the model
        # => call .detach() on all tensors and effectively train a second small MLP (rel_ranker)
        if self.final_ranking_layers is None:
            ranking = None
        else:
            ranking = self.final_ranking_layers(final_token.detach())

        return sbj_cls, obj_cls, output, ranking

    def forward(self, data: dict, return_attention=None, max_relations=None):
        """
        Forward pass through the model

        :param return_attention: Set `return_attention` to output the attention of the respective block.
        This is only intended for debugging and model introspection.
        """
        img = data["img"]
        pair_ids = data["pair_ids"]
        sbj_ids = pair_ids[:, 0]
        obj_ids = pair_ids[:, 1]

        features = self.extractor(img)
        assert features.shape[1:] == self.feature_shape, features.shape

        patches = self.patch_embed(features).flatten(2).transpose(1, 2)

        if max_relations is not None:
            all_sc = []
            all_oc = []
            all_rel = []
            all_rank = []
            for sb, ob in zip(
                sbj_ids.split(max_relations), obj_ids.split(max_relations)
            ):
                sc, oc, rel, rank = self._forward_internal(
                    data=data,
                    features=features,
                    patches=patches,
                    img_shape=img.shape,
                    sbj_ids=sb,
                    obj_ids=ob,
                    return_attention=return_attention,
                )
                if sc is not None:
                    all_sc.append(sc)
                if oc is not None:
                    all_oc.append(oc)
                all_rel.append(rel)
                if rank is not None:
                    all_rank.append(rank)
            return (
                torch.cat(all_sc) if all_sc else None,
                torch.cat(all_oc) if all_oc else None,
                torch.cat(all_rel),
                torch.cat(all_rank) if all_rank else None,
            )

        return self._forward_internal(
            data=data,
            features=features,
            patches=patches,
            img_shape=img.shape,
            sbj_ids=sbj_ids,
            obj_ids=obj_ids,
            return_attention=return_attention,
        )
