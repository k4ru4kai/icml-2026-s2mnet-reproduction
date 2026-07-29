"""Exact seven-stage adapter around the pinned official PyTorch U-Mamba_Bot."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UMAMBA_PACKAGE_ROOT = PROJECT_ROOT / "baselines" / "U-Mamba" / "umamba"


def build_umamba_bot(input_size: int = 384, num_classes: int = 8):
    if input_size != 384 or num_classes != 8:
        raise ValueError("Claim 3 freezes a 384x384 input and eight outputs")
    if str(UMAMBA_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(UMAMBA_PACKAGE_ROOT))
    import torch
    from torch import nn
    from nnunetv2.nets.UMambaBot_2d import UMambaBot
    from nnunetv2.utilities.network_initialization import InitWeights_He

    model = UMambaBot(
        input_channels=3,
        n_stages=7,
        features_per_stage=[32, 64, 128, 256, 320, 320, 320],
        conv_op=nn.Conv2d,
        kernel_sizes=[(3, 3)] * 7,
        strides=[(1, 1)] + [(2, 2)] * 6,
        n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True, "negative_slope": 0.01},
        deep_supervision=False,
    )
    model.apply(InitWeights_He(1e-2))
    return model
