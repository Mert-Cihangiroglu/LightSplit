"""
Dataset-specific model architectures for U-shaped Split Learning.

Each model is split into:
  Head (client) -> smashed data -> Backbone (server) -> Tail (client)

Reference: SafeSplit (NDSS'25), Table I.
Supported: ResNet18 (CIFAR-10/100), MicronNet (GTSRB), SimpleCNN (FMNIST).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model configs ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "cifar10": {
        "model": "resnet18",
        "in_channels": 3,
        "img_size": 32,
        "num_classes": 10,
        "spatial_shape": (64, 8, 8),   # head output shape (C, H, W)
        "activation_dim": 4096,         # 64 * 8 * 8
        "backbone_out_dim": 512,
        "mlp_hidden": 512,               # FR-LL liftback hidden dim (ablation winner)
    },
    "cifar100": {
        "model": "resnet18",
        "in_channels": 3,
        "img_size": 32,
        "num_classes": 100,
        "spatial_shape": (64, 8, 8),
        "activation_dim": 4096,
        "backbone_out_dim": 512,
        "mlp_hidden": 2048,              # ablation winner (harder task benefits from wider MLP)
    },
    "gtsrb": {
        "model": "micronnet",
        "in_channels": 3,
        "img_size": 32,
        "num_classes": 43,
        # Relocated first maxpool from backbone into head (was (29, 28, 28) = 22736).
        # This keeps head params unchanged but reduces smashed-data size 4x so the
        # FR-LL liftback MLP is tractable. Head still runs conv0_bn+conv1+conv2+BN+ReLU
        # before emitting, so privacy against reconstruction attacks is preserved.
        "spatial_shape": (29, 14, 14),
        "activation_dim": 5684,          # 29 * 14 * 14
        "backbone_out_dim": 300,
        "mlp_hidden": 128,               # small-backbone default (ablation winner)
    },
    "fmnist": {
        "model": "simplecnn",
        "in_channels": 1,
        "img_size": 28,
        "num_classes": 10,
        "spatial_shape": (20, 12, 12),
        "activation_dim": 2880,          # 20 * 12 * 12
        "backbone_out_dim": 512,
        "mlp_hidden": 128,               # small-backbone default (ablation winner)
    },
}


# ── Cut-layer width ────────────────────────────────────────────────────
# The cut width is the number of channels the client head emits.  Narrowing it
# shrinks the smashed tensor to d = C_cut * H * W, holding the spatial grid fixed.
# This is NOT the same as learned_1x1: there the head still emits the native width
# and a 1x1 conv squeezes it for transmission; here the head itself is narrower.
def native_cut_channels(dataset: str) -> int:
    return MODEL_CONFIGS[dataset]["spatial_shape"][0]


def cut_dims(dataset: str, cut_channels: int | None = None):
    """-> (C, H, W, d).  cut_channels=None (or the native value) reproduces the config."""
    _, h, w = MODEL_CONFIGS[dataset]["spatial_shape"]
    c = native_cut_channels(dataset) if cut_channels is None else int(cut_channels)
    if c < 1:
        raise ValueError(f"cut_channels must be >= 1, got {c}")
    return c, h, w, c * h * w


# ── BasicBlock (BatchNorm) ──────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ── ResNet18 Head (9,536 params) ────────────────────────────────────────────
class ResNet18Head(nn.Module):
    """First conv layer of ResNet18. Output: cut_channels×8×8 on 32×32 input."""

    def __init__(self, cut_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, cut_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(cut_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return x


# ── ResNet18 Deep Head (46,528 params) ─────────────────────────────────────
class ResNet18HeadDeep(nn.Module):
    """conv1 + BN + ReLU + MaxPool + BasicBlock(C,C). Output: cut_channels×8×8."""

    def __init__(self, cut_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, cut_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(cut_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block = BasicBlock(cut_channels, cut_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.block(x)
        return x


# ── ResNet18 Deep Head, NATIVE-WIDTH BODY (cut-width sweep) ────────────────
class ResNet18HeadDeepFixedBody(nn.Module):
    """conv1 + BN + ReLU + MaxPool at the NATIVE 64ch, then BasicBlock(64, C). Output: C×8×8.

    Companion to ``ResNet18HeadDeep`` for the cut-width (d) sweep. ``ResNet18HeadDeep(C)`` runs the
    whole head at width C, so a narrow cut also means a narrow feature extractor (at C=8 the very
    first conv is 3->8), which confounds "the cut is narrow" with "the client head is weak".

    This variant keeps the head body at the native width and lets ONLY the final stage decide how
    many channels cross the cut, mirroring the SDAR privacy harness, whose width family
    ``[16, 32, C]`` likewise holds the body fixed and narrows only the cut stage. BasicBlock already
    inserts a 1x1+BN shortcut when 64 != C, so both narrowing (C<64) and widening (C>64) work.

    At C=64 it is structurally identical to ``ResNet18HeadDeep(64)``, the deployed L2H head, so the
    sweep passes exactly through the deployed configuration.
    """

    def __init__(self, cut_channels: int = 64, body_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, body_channels, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(body_channels)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block = BasicBlock(body_channels, cut_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.block(x)
        return x

# ── Self-Attention replacement for layer1 ─────────────────────────────────
class _TransformerBlock(nn.Module):
    """Single transformer block: LN + MHSA + LN + MLP."""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SelfAttentionLayer1(nn.Module):
    """Drop-in replacement for layer1 that uses self-attention.
    Treats 8×8 spatial grid as 64 tokens of C-dim each.
    Input: (B, C, H, W) → Output: (B, C, H, W).  Shape-preserving."""
    def __init__(self, channels: int = 64, num_blocks: int = 2, num_heads: int = 4):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _TransformerBlock(channels, num_heads=num_heads) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens)
        # (B, H*W, C) → (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ── ViT Backbone (~10.87M params) ──────────────────────────────────────────
class ViTBackbone(nn.Module):
    """Pure-transformer backbone. Treats C×H×W input as H*W tokens of C-dim.
    Fully spatial-agnostic: no convolutions, no positional embeddings.
    Input: (B, 64, 8, 8), Output: (B, 512)."""

    def __init__(self, in_channels: int = 64, num_tokens: int = 64,
                 embed_dim: int = 384, depth: int = 6, num_heads: int = 6,
                 mlp_ratio: int = 4, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[
            _TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) → (B, H*W, C) → (B, H*W, embed_dim)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        # Prepend CLS token: (B, H*W+1, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # Transformer blocks + extract CLS
        tokens = self.blocks(tokens)
        cls_out = self.norm(tokens[:, 0])
        return self.head_proj(cls_out)


# ── ResNet18 Backbone (11,166,976 params) ───────────────────────────────────
class ResNet18Backbone(nn.Module):
    """layer1 through layer4 + adaptive avg pool. Input: in_planes×8×8, Output: 512-dim.

    in_planes < 64 (a narrowed cut) needs no adapter: layer1's first BasicBlock is
    built as BasicBlock(in_planes, 64), and BasicBlock already inserts a 1x1+BN
    shortcut whenever in_planes != planes.  layers 2-4 are untouched.
    """

    def __init__(self, attn_layer1: bool = False, in_planes: int = 64):
        super().__init__()
        if attn_layer1 and in_planes != 64:
            # SelfAttentionLayer1 is shape-preserving, so it cannot widen C_cut -> 64.
            raise ValueError(f"attn_layer1 requires the native 64-channel cut, got in_planes={in_planes}")
        self.in_planes = in_planes
        if attn_layer1:
            self.layer1 = SelfAttentionLayer1(channels=64, num_blocks=2, num_heads=4)
        else:
            self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)


# ── MicronNet Head (824 params) ────────────────────────────────────────────
class MicronNetHead(nn.Module):
    """MicronNet head for GTSRB. Output: 29×14×14 on 32×32 input.

    Relocated the first maxpool from the original backbone into the head to shrink
    smashed-data size from 22736 to 5684. Head params unchanged (~824).
    """

    def __init__(self, cut_channels: int = 29):
        super().__init__()
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, cut_channels, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(cut_channels)
        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        return self.pool(x)


# ── MicronNet Deep Head ───────────────────────────────────────────────────
class MicronNetHeadDeep(nn.Module):
    """MicronNet deep head: adds extra Conv+BN block before the final maxpool.
    Output: 29×14×14 (same as shallow)."""

    def __init__(self, cut_channels: int = 29):
        super().__init__()
        self.conv0_bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, cut_channels, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(cut_channels)
        self.extra_conv = nn.Conv2d(cut_channels, cut_channels, 3, padding=1, bias=False)
        self.extra_bn = nn.BatchNorm2d(cut_channels)
        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_bn(self.conv1(self.conv0_bn(x))))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.extra_bn(self.extra_conv(x)))
        return self.pool(x)


# ── MicronNet Backbone ────────────────────────────────────────────────────
class MicronNetBackbone(nn.Module):
    """MicronNet backbone. Input: 29×14×14 (post-head-maxpool), Output: 300-dim.

    The original first maxpool was relocated into the head.
    """

    def __init__(self, in_channels: int = 29):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, 59, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(59)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(59, 74, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(74)
        self.fc1 = nn.Linear(1184, 300)
        self.dense1_bn = nn.BatchNorm1d(300)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(self.maxpool3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dense1_bn(x)
        x = F.dropout(x, training=self.training)
        return x


# ── SimpleCNN Head (520 params) ───────────────────────────────────────────
class SimpleCNNHead(nn.Module):
    """SimpleCNN [Cao et al.] head for FMNIST. Output: 20×12×12 on 28×28 input."""

    def __init__(self, cut_channels: int = 20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cut_channels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.conv1(x)))


# ── SimpleCNN Deep Head (~4,160 params) ───────────────────────────────────
class SimpleCNNHeadDeep(nn.Module):
    """SimpleCNN head + extra Conv+BN block. Output: 20×12×12."""

    def __init__(self, cut_channels: int = 20):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cut_channels, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.extra_conv = nn.Conv2d(cut_channels, cut_channels, 3, padding=1, bias=False)
        self.extra_bn = nn.BatchNorm2d(cut_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.extra_bn(self.extra_conv(x)))
        return x


# ── SimpleCNN Backbone (435,162 params) ───────────────────────────────────
class SimpleCNNBackbone(nn.Module):
    """SimpleCNN backbone. Input: 20×12×12, Output: 512-dim."""

    def __init__(self, in_channels: int = 20):
        super().__init__()
        self.conv2 = nn.Conv2d(in_channels, 50, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


# ── Factory functions ───────────────────────────────────────────────────────
def build_head(dataset: str, deep: bool = False, cut_channels: int | None = None,
               fixed_body: bool = False) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset]
    c = native_cut_channels(dataset) if cut_channels is None else int(cut_channels)
    if cfg["model"] == "resnet18":
        if deep and fixed_body:
            # cut-width sweep: native-width body, only the cut stage varies (see the class docstring)
            return ResNet18HeadDeepFixedBody(c, body_channels=native_cut_channels(dataset))
        return ResNet18HeadDeep(c) if deep else ResNet18Head(c)
    if cfg["model"] == "micronnet":
        return MicronNetHeadDeep(c) if deep else MicronNetHead(c)
    if cfg["model"] == "simplecnn":
        return SimpleCNNHeadDeep(c) if deep else SimpleCNNHead(c)
    raise ValueError(f"No head defined for dataset {dataset}")


def build_backbone(dataset: str, backbone: str = "resnet18",
                   attn_layer1: bool = False, cut_channels: int | None = None) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset]
    c, h, w, _ = cut_dims(dataset, cut_channels)
    if backbone == "vit":
        return ViTBackbone(
            in_channels=c, num_tokens=h * w,
            embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
            out_dim=cfg["backbone_out_dim"],
        )
    if cfg["model"] == "resnet18":
        return ResNet18Backbone(attn_layer1=attn_layer1, in_planes=c)
    if cfg["model"] == "micronnet":
        return MicronNetBackbone(in_channels=c)
    if cfg["model"] == "simplecnn":
        return SimpleCNNBackbone(in_channels=c)
    raise ValueError(f"No backbone defined for dataset {dataset}")


def build_tail(dataset: str) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset]
    return nn.Linear(cfg["backbone_out_dim"], cfg["num_classes"])
