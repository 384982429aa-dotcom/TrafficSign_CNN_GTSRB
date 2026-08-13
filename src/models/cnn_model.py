# -*- coding: utf-8 -*-
"""
PyTorch Small-Image ResNet18 + FPN model for traffic sign classification.

Input:
    Tensor shape: (N, 3, 64, 64)

Output:
    Logits shape: (N, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_NUM_CLASSES = 43


class BasicBlock(nn.Module):
    """ResNet18 basic residual block."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class FPNFuse(nn.Module):
    """Upsample deep feature, concatenate shallow feature, then refine."""

    def __init__(self, deep_channels: int, shallow_channels: int, out_channels: int):
        super().__init__()
        in_channels = deep_channels + shallow_channels

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, deep_feature: torch.Tensor, shallow_feature: torch.Tensor) -> torch.Tensor:
        deep_feature = F.interpolate(
            deep_feature,
            size=shallow_feature.shape[-2:],
            mode="nearest",
        )
        fused = torch.cat([deep_feature, shallow_feature], dim=1)
        return self.refine(fused)


class TrafficSignResNet18FPN(nn.Module):
    """
    Small-image ResNet18 backbone with FPN-style multi-scale feature fusion.

    Feature maps:
        C1: 64x64x64
        C2: 32x32x128
        C3: 16x16x256
        C4: 8x8x512
    """

    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # Small-image stem: no 7x7 stride=2 conv and no early maxpool.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(64, 64, blocks=2, first_stride=1)
        self.stage2 = self._make_stage(64, 128, blocks=2, first_stride=2)
        self.stage3 = self._make_stage(128, 256, blocks=2, first_stride=2)
        self.stage4 = self._make_stage(256, 512, blocks=2, first_stride=2)

        self.fuse_c4_c3 = FPNFuse(512, 256, 256)
        self.fuse_f3_c2 = FPNFuse(256, 128, 128)
        self.fuse_f2_c1 = FPNFuse(128, 64, 64)

        # GAP features: F1(64) + F2(128) + F3(256) + C4(512) = 960.
        self.classifier = nn.Sequential(
            nn.Linear(64 + 128 + 256 + 512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=first_stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        f3 = self.fuse_c4_c3(c4, c3)
        f2 = self.fuse_f3_c2(f3, c2)
        f1 = self.fuse_f2_c1(f2, c1)

        p1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)
        p2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
        p3 = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        p4 = F.adaptive_avg_pool2d(c4, 1).flatten(1)

        multi_scale_feature = torch.cat([p1, p2, p3, p4], dim=1)
        logits = self.classifier(multi_scale_feature)
        return logits


def build_cnn_model(num_classes: int = DEFAULT_NUM_CLASSES) -> TrafficSignResNet18FPN:
    """Default model entry used by training code."""
    return TrafficSignResNet18FPN(num_classes=num_classes)


if __name__ == "__main__":
    model = build_cnn_model(num_classes=DEFAULT_NUM_CLASSES)
    dummy = torch.randn(2, 3, 64, 64)
    output = model(dummy)
    print(model)
    print("Output shape:", output.shape)
