"""Paper-derived STE + TTE + CSA future-frame prediction model.

Tran et al., TCSVT 2024, Section III-B, Fig. 2, Eqs. (6)-(12).
Input: [batch, frames, 3, height, width], normalized to [-1, 1].
Output: [batch, 3, 256, 256], matching data_utils.py's prediction target.

The author's complete model.py is absent from this checkout. This is a
reimplementation, not a recovered implementation or checkpoint-compatible copy.
Eq. (11) does not unambiguously reduce the token sequence to one vector. Here
CSA uses standard softmax(Q K^T / sqrt(d_head)) V aggregation, with the two
alignment projections and original CLS residual described in Eqs. (11)-(12).
"""

import torch
from torch import nn


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class PositionEmbs(nn.Module):
    """Learnable positions for patch/frame tokens and the leading CLS token."""

    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding)


class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(x)))))


class EncoderBlock(nn.Module):
    """Pre-norm ViT block, as used in the repository's existing encoders."""

    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1,
                 attn_dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(
            in_dim, num_heads, dropout=attn_dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        normalized = self.norm1(x)
        attended, _ = self.attn(normalized, normalized, normalized,
                                need_weights=False)
        x = x + self.dropout(attended)
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12,
                 num_heads=8, dropout_rate=0.1, attn_dropout_rate=0.0):
        super().__init__()
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(emb_dim, mlp_dim, num_heads, dropout_rate,
                         attn_dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.pos_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.norm(x)


class SpatialTransformer(nn.Module):
    """One shared frame encoder: patch embedding -> CLS -> spatial attention."""

    def __init__(self, image_size, patch_size, emb_dim, mlp_dim, num_heads,
                 num_layers, dropout_rate, attn_dropout_rate):
        super().__init__()
        h, w = image_size
        ph, pw = patch_size
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=patch_size,
                                   stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = Encoder(
            (h // ph) * (w // pw), emb_dim, mlp_dim, num_layers, num_heads,
            dropout_rate, attn_dropout_rate)

    def forward(self, frames):
        patches = self.embedding(frames).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(frames.size(0), -1, -1)
        tokens = self.transformer(torch.cat((cls_token, patches), dim=1))
        return tokens[:, 0]


class CrossAttention(nn.Module):
    """Aligned temporal CLS queries all temporal tokens; returns [B, 1, D].

    The original CLS is added exactly once, inside this module (Eq. 12).
    Keys and values include the aligned CLS itself, as in Eq. (11).
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError('dim must be divisible by a positive num_heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.align1 = nn.Linear(dim, dim)  # W_a,1
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # W_a,2
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, length, dim = x.shape
        original_cls = x[:, :1]
        aligned_cls = self.align1(original_cls)
        context = torch.cat((aligned_cls, x[:, 1:]), dim=1)
        q = self.wq(aligned_cls).reshape(
            batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(context).reshape(
            batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(context).reshape(
            batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        weights = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        aggregate = (self.attn_drop(weights) @ v).transpose(1, 2).reshape(
            batch, 1, dim)
        return original_cls + self.proj_drop(self.proj(aggregate))


class Decoder(nn.Module):
    """Eqs. (3)-(5): project a video vector, then upsample to a 256px frame."""

    def __init__(self, emb_dim=768):
        super().__init__()

        def basic(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU())

        def upsample(channels, factor):
            # Parameters match the existing repository's decoder.
            kernel, padding, output_padding = (3, 1, 1) if factor == 2 else (4, 1, 2)
            return nn.Sequential(
                nn.ConvTranspose2d(channels, channels, kernel, stride=factor,
                                   padding=padding, output_padding=output_padding),
                nn.BatchNorm2d(channels), nn.ReLU())

        self.de_dense = nn.Sequential(nn.Linear(emb_dim, 256 * 16 * 16), nn.ELU())
        self.decoder = nn.Sequential(
            basic(256, 128), upsample(128, 2),  # 16 -> 32
            basic(128, 64), upsample(64, 2), upsample(64, 4),  # 32 -> 64 -> 256
            nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1), nn.Tanh()))

    def forward(self, x):
        features = self.de_dense(x).reshape(x.size(0), 256, 16, 16)
        return self.decoder(features)


class VisionTransformer(nn.Module):
    """Full STE + TTE + CSA predictor with the training scripts' constructor.

    Defaults use the paper's 32px patches, 768 dimensions, 8 heads, and 12
    blocks in EACH encoder. Explicit constructor arguments override defaults;
    the repository's existing b16 configuration uses different hyperparameters.
    num_classes, nu and feat_dim are accepted for legacy call sites only:
    prediction uses neither a classifier nor a hypersphere objective.
    """

    def __init__(self, image_size=(384, 384), patch_size=(32, 32), emb_dim=768,
                 mlp_dim=3072, num_heads=8, num_layers=12, num_classes=1000,
                 attn_dropout_rate=0.0, dropout_rate=0.1, num_frames=4,
                 nu=0.01, feat_dim=None):
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        if len(self.image_size) != 2 or len(self.patch_size) != 2:
            raise ValueError('image_size and patch_size must be integers or pairs')
        if any(size <= 0 or patch <= 0 or size % patch
               for size, patch in zip(self.image_size, self.patch_size)):
            raise ValueError('image dimensions must be positive multiples of patch size')
        if num_frames <= 0 or num_layers <= 0 or mlp_dim <= 0 or emb_dim <= 0:
            raise ValueError('num_frames, num_layers, mlp_dim and emb_dim must be positive')
        if num_heads <= 0 or emb_dim % num_heads:
            raise ValueError('emb_dim must be divisible by a positive num_heads')
        self.num_frames = num_frames
        self.spatial_transformer = SpatialTransformer(
            self.image_size, self.patch_size, emb_dim, mlp_dim, num_heads,
            num_layers, dropout_rate, attn_dropout_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.temporal_transformer = Encoder(
            num_frames, emb_dim, mlp_dim, num_layers, num_heads,
            dropout_rate, attn_dropout_rate)
        self.cross_att = CrossAttention(
            emb_dim, num_heads=num_heads, attn_drop=attn_dropout_rate,
            proj_drop=dropout_rate)
        self.decoder = Decoder(emb_dim)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError('expected input [batch, frames, 3, height, width]')
        batch, frames, channels, height, width = x.shape
        if frames != self.num_frames or channels != 3:
            raise ValueError(f'expected {self.num_frames} frames and 3 channels; '
                             f'got {frames} frames and {channels} channels')
        if (height, width) != self.image_size:
            raise ValueError(f'expected frame size {self.image_size}; got {(height, width)}')
        # data_utils.py returns float64 arrays. Convert dtype, but leave device
        # placement to the caller so CPU, CUDA and MPS all remain usable.
        x = x.to(dtype=self.spatial_transformer.embedding.weight.dtype)
        # Batch the shared spatial encoder while preserving chronological order.
        spatial_cls = self.spatial_transformer(
            x.reshape(batch * frames, channels, height, width)).reshape(batch, frames, -1)
        temporal_input = torch.cat((self.cls_token.expand(batch, -1, -1), spatial_cls), dim=1)
        temporal_tokens = self.temporal_transformer(temporal_input)
        enhanced_cls = self.cross_att(temporal_tokens)
        return self.decoder(enhanced_cls[:, 0])
