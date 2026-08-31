"""
Dataset-specific model architectures for U-shaped Split Learning.

Each model is split into:
  Head (client) -> smashed data -> Backbone (server) -> Tail (client)

Reference: SafeSplit (NDSS'25), Table I.
Phase 1: ResNet18 for CIFAR-10 only.
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
    },
    "cifar100": {
        "model": "resnet18",
        "in_channels": 3,
        "img_size": 32,
        "num_classes": 100,
        "spatial_shape": (64, 8, 8),
        "activation_dim": 4096,
        "backbone_out_dim": 512,
    },
    # Deferred:
    # "mnist":   {"model": "simplecnn", "in_channels": 1, "img_size": 28, ...}
    # "fmnist":  {"model": "simplecnn", "in_channels": 1, "img_size": 28, ...}
    # "gtsrb":   {"model": "micronnet", "in_channels": 3, "img_size": 32, ...}
}


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
    """First conv layer of ResNet18. Output: 64×8×8 on 32×32 input."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return x


# ── ResNet18 Deep Head (46,528 params) ─────────────────────────────────────
class ResNet18HeadDeep(nn.Module):
    """conv1 + BN + ReLU + MaxPool + BasicBlock(64,64). Output: 64×8×8."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block = BasicBlock(64, 64, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.block(x)
        return x


# ── ResNet18 Backbone (11,166,976 params) ───────────────────────────────────
class ResNet18Backbone(nn.Module):
    """layer1 through layer4 + adaptive avg pool. Input: 64×8×8, Output: 512-dim."""

    def __init__(self):
        super().__init__()
        self.in_planes = 64
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


# ── Factory functions ───────────────────────────────────────────────────────
def build_head(dataset: str, deep: bool = False) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset]
    if cfg["model"] == "resnet18":
        return ResNet18HeadDeep() if deep else ResNet18Head()
    raise ValueError(f"No head defined for dataset {dataset}")


def build_backbone(dataset: str) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset]
    if cfg["model"] == "resnet18":
        return ResNet18Backbone()
    raise ValueError(f"No backbone defined for dataset {dataset}")


def build_tail(dataset: str) -> nn.Module:
    cfg = MODEL_CONFIGS[dataset]
    return nn.Linear(cfg["backbone_out_dim"], cfg["num_classes"])
