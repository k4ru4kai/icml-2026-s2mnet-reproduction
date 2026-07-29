"""Eight-logit adapter around the pinned official TensorFlow S2M-Net."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = PROJECT_ROOT / "official_repo"


def build_s2mnet_logits(input_size: int = 384, num_classes: int = 8):
    if num_classes != 8:
        raise ValueError("Claim 3 freezes eight output channels")
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    import tensorflow as tf
    from s2mnet.models.s2mnet import S2MNet

    released = S2MNet(
        input_size=input_size,
        num_classes=num_classes,
        filters=(24, 32, 64, 80, 128),
        use_mrfse=True,
        mrfse_kernels=(3, 5, 7),
        se_reduction=16,
        expand_ratio=6,
        use_sstm=True,
        sstm_k=32,
        sstm_ssm_dim=16,
        sstm_stages=(True, True, True, True, True),
        sstm_use_spectral=(True, True, True, True, True),
        sstm_use_ssm=(False, False, True, True, True),
        sstm_dropout=0.1,
        use_bfp=True,
        bfp_routing="soft",
        dropout=0.1,
        l2_reg=0.0,
        activation="elu",
        name="S2M-Net-Claim3",
    )
    features = released.get_layer("head_conv2").output
    logits = tf.keras.layers.Conv2D(
        num_classes,
        1,
        padding="same",
        activation=None,
        dtype="float32",
        name="claim3_logits",
    )(features)
    return tf.keras.Model(released.input, logits, name="S2M-Net-Claim3-Logits")
