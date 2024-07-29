import math
import torch
import torch.nn as nn


def make_sine_position_encoding(
    in_feature_size, patch_size, d_model, temperature=10000, scale=2 * math.pi
):
    # number of patches per height and width
    h, w = in_feature_size[0] // (patch_size[0]), in_feature_size[1] // (patch_size[1])
    area = torch.ones(1, h, w)  # [b, h, w]
    # 1, 2, 3, 4, 5, ... in x-direction
    y_embed = area.cumsum(1, dtype=torch.float32)
    # 1, 2, 3, 4, 5, ... in y-direction
    x_embed = area.cumsum(2, dtype=torch.float32)

    one_direction_feats = d_model // 2

    eps = 1e-6
    # equally spaced entries between 0 and scale in y-direction
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    # equally spaced entries between 0 and scale in x-direction
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
    # temperature ** (half of embed size equally spaced double entries (always to identical entries)
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / one_direction_feats
    )

    # adds embedding_size // 2 dimension in the end. smooth transition from largest value scale (pos_x[:, :, -1, 0]) to 0 (pos_x[:, :, 0, -1]) with exponential drop
    pos_x = x_embed[:, :, :, None] / dim_t
    # same, but dimension 1 and 2 swapped
    pos_y = y_embed[:, :, :, None] / dim_t
    # sine and cos wave from [:, :, 0, -1] to [:, :, -1, 0] becoming "slower" at [:, :, 0, -1]. Only every second entry taken but there are always two identical (for sine and cos). After stack and flatten, alternating sine and cosine values
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    # combination of sine and cosine waves. in the embedding direction, the first halt (up to 96) is y-direction, then x-direction. the embedding direction is shifted to dimension 1
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    # put the patches together
    pos = pos.flatten(2).permute(0, 2, 1)
    return pos


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        scale_head=True,
        attn_drop=0.0,
        proj_drop=0.0,
        kv_token_length=None,
    ):
        """
        kv_token_length: if None, the attention is calculated between all tokens. If a number is set, only these tokens are used for keys and values and all tokens for the query
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim**-0.5 if not scale_head else (dim // num_heads) ** -0.5
        self.kv_token_length = kv_token_length

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch size, number of tokens (visual and keypoints) and number of channels
        (B, N, C) = x.shape
        kv_tokens = self.kv_token_length if self.kv_token_length is not None else N
        # linear transformation of current tokens to queries, keys and values (channel dimension tripled)
        qkv = self.qkv(x)
        # channels are split into 3 dims (qkv), number of heads and remaining channels
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode="floor")
        )
        # now we have the order 3, B, heads, N, C
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k, v have the dimensions B, heads, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]
        # k, v have dimensions B, heads, N_visual, C
        k, v = (k[:, :, -kv_tokens:], v[:, :, -kv_tokens:])

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # k has dimensions B, heads, C, N
        k_transposed = k.transpose(-2, -1)
        # matrix multiplication in the last two dimensions, attn has dimensions B, heads, N, N_visual
        attn = q @ k_transposed
        # vector normalization
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # matrix multiplication of attention and values, dimensions are B, heads, N, C (all N, not only visual)
        x = attn @ v
        x = x.transpose(1, 2)  # dimensions are B, N, heads, C
        # removing heads dimension
        x = x.reshape(B, N, C)
        # linear projection from channels to channels
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        scale_head=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_patches=None,
        num_tokens=None,
        mask_sequence_repetition=1,
        kv_token_length=None,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            scale_head=scale_head,
            attn_drop=attn_drop,
            proj_drop=drop,
            kv_token_length=kv_token_length,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.non_token_indices = None
        self.mask_sequence_repetition = mask_sequence_repetition

    def forward(self, x, pos_encoding=None, return_attention=False, token_mask=None):
        """
        @param x: The input to the transformer block with visual tokens and keypoint (and thickness) tokens
        @param pos_encoding: if available, it is added before the block
        @param return_attention: return the attention result without additional mlp and norm execution
        @param token_mask: mask the preceding tokens with the given indices
        @return:
        """
        if pos_encoding is not None:
            x[:, -pos_encoding.size(1) :] += pos_encoding
        if token_mask is not None:
            if self.non_token_indices is None:
                self.non_token_indices = torch.tensor(
                    [i for i in range(x.shape[1] - self.num_patches, x.shape[1])]
                )
            if self.mask_sequence_repetition > 1:
                # more than one token correspond to each mask entry
                append_mask = []
                for i in range(1, self.mask_sequence_repetition):
                    append_mask.extend(
                        [idx + i * self.num_tokens for idx in token_mask]
                    )
                token_mask = torch.cat((token_mask, torch.tensor(append_mask)))
            token_mask = torch.cat((token_mask, self.non_token_indices))
            x = x[:, token_mask]
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
