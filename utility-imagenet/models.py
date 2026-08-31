"""
ImageNet-1k U-shaped Split-Learning model + the four compression methods.

Split (same protocol as the utility harness, scaled to ImageNet):

    Head (client)  ->  Bottleneck (the cut)  ->  Backbone (server)  ->  Tail (client)

  * Head    = ResNet-50 stem (7x7 s2 conv + BN + ReLU + 3x3 s2 maxpool) followed
              by an adaptive-avg-pool to `cut_hw x cut_hw`.  The pool is what keeps
              LightSplit-F feasible on 224x224 inputs: the natural early cut is
              64 x 56 x 56 = 200,704-dim, which would make the dense projection R a
              ~20 GB matrix.  Pooling to 8x8 fixes the smashed-data dimension at
              d = 64*8*8 = 4096 -- identical to CIFAR -- so the projection/storage
              cost is *resolution-independent* (R = 8.4 MB at CR 8x, same as CIFAR).
  * Backbone = ResNet-50 layer1..layer4 + global avg-pool  (server, the heavy part).
  * Tail     = Linear(2048, num_classes)  (client; labels never leave the client).

The four bottleneck classes (Raw / TopKSparse / Learned1x1 / FixedRandom = LightSplit-F)
are ported verbatim from utility/train_split.py so the methods are identical to
the rest of the paper; only the surrounding model is scaled up to ResNet-50 @ 224.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ── ImageNet config ─────────────────────────────────────────────────────────
IMAGENET_CONFIG = {
    "num_classes": 1000,
    "cut_channels": 64,        # ResNet stem output channels (fixed by the architecture)
    "cut_hw": 8,               # client pools head output to cut_hw x cut_hw  ->  d = 64*hw*hw
    "backbone_out_dim": 2048,  # ResNet-50 final feature dim
}


def activation_dim_for(cut_hw: int) -> int:
    return IMAGENET_CONFIG["cut_channels"] * cut_hw * cut_hw


def _resnet50(weights_none: bool = True):
    """Construct a randomly-initialised torchvision ResNet-50, version-robust."""
    try:
        return tvm.resnet50(weights=None)
    except TypeError:                       # older torchvision
        return tvm.resnet50(pretrained=False)


# ── BasicBlock (for the optional deep client head) ──────────────────────────
class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock; used only as the extra client-head block (L2H)."""
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


# ============================================================================
#  Bottlenecks  (ported verbatim from utility/train_split.py)
# ============================================================================
class RawBottleneck(nn.Module):
    """No compression: transmit the full activation (CR = 1x, the utility ceiling)."""
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return h, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload


# Honest transmitted-byte accounting for sparse payloads:
# each kept coordinate costs a 4-byte float32 value + a 2-byte uint16 index.
VALUE_BYTES = 4
INDEX_BYTES = 2


class TopKSparseBottleneck(nn.Module):
    """Magnitude top-k sparsification (matched-byte compression baseline).

    Keeps the k_s largest-magnitude coordinates per sample, zeroes the rest. The
    payload stays dense so the recon path is reused unchanged; the honest
    transmitted cost (k_s*(4+2) bytes) is carried by SplitModel.payload_dim.
    Gradients flow straight through the kept coordinates (mask-multiply).
    """
    def __init__(self, k_s: int):
        super().__init__()
        self.k_s = k_s

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, dim = h.shape
        k = min(self.k_s, dim)
        _, idx = torch.topk(h.abs(), k, dim=1)
        mask = torch.zeros_like(h).scatter(1, idx, 1.0)   # 0/1 keep-mask, per sample
        return h * mask, {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload


class FixedRandomBottleneck(nn.Module):
    """LightSplit-F: fixed orthonormal random projection R (client-only).

    Client compresses h @ R  (R is d x k, orthonormal columns via QR).  Server
    lifts back with R^T.  R is a fixed buffer (never trained, never sent to the
    server); its storage is d*k floats = 8.4 MB at d=4096, CR 8x.
    """
    def __init__(self, activation_dim: int, k: int):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)
        self.k = k

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        return z, {"bottleneck_z": z}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return payload @ self.R.T


class Learned1x1Bottleneck(nn.Module):
    """Learned 1x1-conv channel compressor/decompressor (trainable baseline)."""
    def __init__(self, in_channels: int, k_channels: int):
        super().__init__()
        self.k_channels = k_channels
        self.compress = nn.Conv2d(in_channels, k_channels, 1)
        self.decompress = nn.Conv2d(k_channels, in_channels, 1)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.compress(z_spatial), {}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.decompress(payload)


class FixedRandomMLPLiftBottleneck(nn.Module):
    """FR-LL: fixed orthonormal random projection R (client) + learned MLP lift-back (server).

    Same client-side compression and transmitted payload as FixedRandomBottleneck
    (k floats via h @ R, R a fixed client-only buffer), but the server replaces the
    fixed R^T decoder with a trainable MLP (k -> hidden -> activation_dim) that has
    no access to R. Transmitted bytes are identical to fixed_random, so the
    comparison stays matched-byte; only the server-side reconstruction is learned.
    """
    def __init__(self, activation_dim: int, k: int, hidden: int = 512, depth: int = 2):
        super().__init__()
        rand_matrix = torch.randn(activation_dim, k)
        q, _ = torch.linalg.qr(rand_matrix)
        self.register_buffer("R", q)        # client-side only; server never sees it
        self.k = k
        self.hidden = hidden
        self.depth = depth
        layers = [nn.Linear(k, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):          # depth=2 keeps the original single hidden layer
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden, activation_dim)]
        self.learned_lift = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = h @ self.R
        return z, {"bottleneck_z": z}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.learned_lift(payload)


class BlockRandomBottleneck(nn.Module):
    """Block-local fixed orthonormal random projection (privacy/utility dial).

    Splits the (C, H, W) cut tensor into block x block spatial tiles, flattens each
    tile to C*block*block dims, and projects it with a fixed orthonormal random R to
    out_per_block dims. The macro grid (H/block x W/block) is preserved, so the random
    mixing is confined to a block x block receptive field -- the knob between privacy
    and utility:
        block = 1     -> per-position channel projection  (max utility, min privacy)
        block = H     -> single global dense R == FixedRandomBottleneck (max privacy)
    Total payload = n_blocks * out_per_block == k floats, so the transmitted-byte
    budget is identical to fixed_random. One R is shared across all tiles and kept as
    a client-only buffer (never trained, never sent to the server).
    """
    def __init__(self, channels: int, hw: int, block: int, k: int):
        super().__init__()
        if hw % block != 0:
            raise ValueError(f"block={block} must divide spatial size hw={hw}")
        self.channels = channels
        self.hw = hw
        self.block = block
        self.grid = hw // block                 # tiles per side
        self.n_blocks = self.grid * self.grid
        self.in_per_block = channels * block * block
        if k % self.n_blocks != 0:
            raise ValueError(f"k={k} must be divisible by n_blocks={self.n_blocks}")
        self.out_per_block = k // self.n_blocks
        self.k = k
        rand_matrix = torch.randn(self.in_per_block, self.out_per_block)
        q, _ = torch.linalg.qr(rand_matrix)     # (in_per_block, out_per_block), orthonormal cols
        self.register_buffer("R", q)            # shared across tiles; client-only

    def _to_blocks(self, z_spatial: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, n_blocks, in_per_block)
        b = z_spatial.shape[0]
        x = z_spatial.unfold(2, self.block, self.block).unfold(3, self.block, self.block)
        # x: (B, C, grid, grid, block, block)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()        # (B, grid, grid, C, block, block)
        return x.reshape(b, self.n_blocks, self.in_per_block)

    def _from_blocks(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_blocks, in_per_block) -> (B, C, H, W)
        b = x.shape[0]
        x = x.reshape(b, self.grid, self.grid, self.channels, self.block, self.block)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()        # (B, C, grid, block, grid, block)
        return x.reshape(b, self.channels, self.hw, self.hw)

    def forward(self, z_spatial: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        payload = self._to_blocks(z_spatial) @ self.R       # (B, n_blocks, out_per_block)
        return payload, {"bottleneck_z": payload.flatten(1)}

    def lift_back(self, payload: torch.Tensor, info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._from_blocks(payload @ self.R.transpose(0, 1))


# ============================================================================
#  Model pieces  (ResNet-50, split)
# ============================================================================
class ResNet50Head(nn.Module):
    """Client head: ResNet-50 stem -> adaptive-pool to cut_hw x cut_hw [-> deep block].

    Output: (B, 64, cut_hw, cut_hw).  The adaptive pool is the smashed-data-dimension
    control that makes the dense LightSplit-F projection feasible at 224x224.
    """
    def __init__(self, cut_hw: int = 8, deep: bool = False):
        super().__init__()
        net = _resnet50()
        self.conv1 = net.conv1          # 7x7 s2, 3 -> 64
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool      # 3x3 s2  -> 64 x 56 x 56 at 224 input
        self.pool = nn.AdaptiveAvgPool2d((cut_hw, cut_hw))
        self.deep = deep
        if deep:                        # L2H: one extra client-side residual block (keeps 64ch)
            self.block = BasicBlock(64, 64, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(x)
        if self.deep:
            x = self.block(x)
        return x


class ResNet50Backbone(nn.Module):
    """Server backbone: ResNet-50 layer1..layer4 + global avg-pool. Output: (B, 2048)."""
    def __init__(self):
        super().__init__()
        net = _resnet50()
        self.layer1 = net.layer1        # 64 -> 256
        self.layer2 = net.layer2        # -> 512  (s2)
        self.layer3 = net.layer3        # -> 1024 (s2)
        self.layer4 = net.layer4        # -> 2048 (s2)
        self.avgpool = net.avgpool      # adaptive avg -> 1x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)


# ============================================================================
#  Split model
# ============================================================================
class SplitModel(nn.Module):
    """U-shaped split ResNet-50 with a pluggable compression bottleneck.

    mode in {"raw", "topk_sparse", "learned_1x1", "fixed_random"}.
    k = activation_dim // compression_ratio is the LightSplit float budget; the other
    methods are matched to the same transmitted-byte budget (see payload_dim).
    """
    def __init__(self, mode: str, k: int, deep_head: bool = False,
                 num_clients: int = 1, client_heads: bool = False,
                 cut_hw: int = 8, wcc_lambda: float = 0.0, ll_hidden: int = 512, ll_depth: int = 2,
                 block_size: int = 2):
        super().__init__()
        self.mode = mode
        self.k = k
        self.cut_hw = cut_hw
        self.client_heads = client_heads
        self.activation_dim = activation_dim_for(cut_hw)
        self.spatial_shape = (IMAGENET_CONFIG["cut_channels"], cut_hw, cut_hw)
        num_classes = IMAGENET_CONFIG["num_classes"]
        out_dim = IMAGENET_CONFIG["backbone_out_dim"]

        # Client: head (shared, or per-client)
        if client_heads:
            self.heads = nn.ModuleList([
                ResNet50Head(cut_hw=cut_hw, deep=deep_head) for _ in range(num_clients)
            ])
            self.head = None
        else:
            self.head = ResNet50Head(cut_hw=cut_hw, deep=deep_head)
            self.heads = None

        # Server: backbone.  Client: tail.
        self.backbone = ResNet50Backbone()
        self.tail = nn.Linear(out_dim, num_classes)

        # Bottleneck (between head and backbone)
        in_ch = IMAGENET_CONFIG["cut_channels"]
        if mode == "raw":
            self.bottleneck = RawBottleneck()
        elif mode == "topk_sparse":
            # matched-byte: keep k_s coords so k_s*(4+2) bytes ~= k*4 bytes (LightSplit budget)
            self._k_s = max(1, (VALUE_BYTES * k) // (VALUE_BYTES + INDEX_BYTES))
            self.bottleneck = TopKSparseBottleneck(k_s=self._k_s)
        elif mode == "fixed_random":
            self.bottleneck = FixedRandomBottleneck(activation_dim=self.activation_dim, k=k)
        elif mode == "fixed_random_ll":
            self.bottleneck = FixedRandomMLPLiftBottleneck(
                activation_dim=self.activation_dim, k=k, hidden=ll_hidden, depth=ll_depth)
        elif mode == "block_random":
            self.block_size = block_size
            self.bottleneck = BlockRandomBottleneck(
                channels=in_ch, hw=cut_hw, block=block_size, k=k)
        elif mode == "learned_1x1":
            k_channels = in_ch // (self.activation_dim // k) if k < self.activation_dim else in_ch
            k_channels = max(k_channels, 1)
            self.bottleneck = Learned1x1Bottleneck(in_channels=in_ch, k_channels=k_channels)
            self._k_channels = k_channels
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @property
    def payload_dim(self) -> int:
        """Honest transmitted size in float32-equivalents (for matched-byte accounting)."""
        if self.mode == "raw":
            return self.activation_dim
        if self.mode == "learned_1x1":
            _, h, w = self.spatial_shape
            return self._k_channels * h * w
        if self.mode == "topk_sparse":
            return (self._k_s * (VALUE_BYTES + INDEX_BYTES) + VALUE_BYTES - 1) // VALUE_BYTES
        # fixed_random / fixed_random_ll (same k-float payload; only the server decoder differs)
        return self.k

    def forward(self, x: torch.Tensor, client_id: int = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Client: head
        if self.client_heads:
            assert client_id is not None, "client_id required with per-client heads"
            z_spatial = self.heads[client_id](x)
        else:
            z_spatial = self.head(x)

        # Bottleneck: learned_1x1 and block_random work on the spatial tensor; the rest flatten.
        if isinstance(self.bottleneck, (Learned1x1Bottleneck, BlockRandomBottleneck)):
            compressed, info = self.bottleneck(z_spatial)
            z_approx = self.bottleneck.lift_back(compressed, info)
        else:
            z = z_spatial.flatten(1)
            payload, info = self.bottleneck(z)
            z_approx = self.bottleneck.lift_back(payload, info)
            z_approx = z_approx.reshape_as(z_spatial)

        if self.training and self.mode != "raw":
            info["recon_loss"] = F.mse_loss(z_approx, z_spatial)

        # Server: backbone.  Client: tail.
        logits = self.tail(self.backbone(z_approx))
        return logits, info


def build_model(mode: str, compression_ratio: int, deep_head: bool, client_heads: bool,
                num_clients: int, cut_hw: int = 8, wcc_lambda: float = 0.0,
                ll_hidden: int = 512, block_size: int = 2, ll_depth: int = 2) -> SplitModel:
    """Convenience constructor: derive k from the compression ratio and build the model."""
    k = activation_dim_for(cut_hw) // compression_ratio
    return SplitModel(mode=mode, k=k, deep_head=deep_head, num_clients=num_clients,
                      block_size=block_size,
                      client_heads=client_heads, cut_hw=cut_hw, wcc_lambda=wcc_lambda,
                      ll_hidden=ll_hidden, ll_depth=ll_depth)
