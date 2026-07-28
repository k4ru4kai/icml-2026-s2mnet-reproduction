from __future__ import annotations

import json

from repro.diagnostics.verify_claim1_architecture import (
    deterministic_json_bytes,
    ledger_summary,
    official_swin_unet_ledger,
    official_transunet_ledger,
    parameter_partition,
    ratio,
    shape_numel,
    sum_parameter_shapes,
    verify_encoder_architecture,
)


class FakeWeight:
    def __init__(self, shape):
        self.shape = shape


class FakeCountModel:
    def __init__(self):
        self.trainable_weights = [
            FakeWeight((2, 3)),
            FakeWeight((4,)),
        ]
        self.non_trainable_weights = [FakeWeight((2, 2))]
        self.weights = self.trainable_weights + self.non_trainable_weights

    def count_params(self):
        return 14


class FakeLayer:
    def __init__(self, name, output_shape=(None, 1, 1, 1)):
        self.name = name
        self.output_shape = output_shape


class FakeArchitectureModel:
    def __init__(self, channels=(24, 32, 64, 80, 128)):
        resolutions = (176, 88, 44, 22, 11)
        self.layers = []
        for stage, (resolution, channel) in enumerate(
            zip(resolutions, channels, strict=True), 1
        ):
            self.layers.extend(
                [
                    FakeLayer(
                        f"enc{stage}_act",
                        (None, resolution, resolution, channel),
                    ),
                    FakeLayer(f"mrfse_stage{stage}"),
                    FakeLayer(f"sstm_stage{stage}"),
                ]
            )


def test_counting_uses_products_and_preserves_partitions():
    model = FakeCountModel()

    assert shape_numel((2, 3, 4)) == 24
    assert sum_parameter_shapes(model.weights) == 14
    assert parameter_partition(model) == {
        "total": 14,
        "trainable": 10,
        "non_trainable": 4,
    }


def test_ratio_rounding_is_stable_and_matches_claim_values():
    assert ratio(60_000_000, 4_791_544) == 12.522059695163
    assert ratio(27_168_900, 4_791_544) == 5.670176460865
    assert ratio(27_000_000, 4_700_000) == 5.744680851064


def test_architecture_check_accepts_exact_instantiated_stages():
    result = verify_encoder_architecture(FakeArchitectureModel())

    assert result["status"] == "pass"
    assert result["observed"]["stage_count"] == 5
    assert result["observed"]["channels"] == [24, 32, 64, 80, 128]
    assert all(result["checks"].values())


def test_architecture_check_rejects_wrong_channel():
    result = verify_encoder_architecture(
        FakeArchitectureModel(channels=(24, 32, 64, 96, 128))
    )

    assert result["status"] == "fail"
    assert result["checks"]["channels"] is False


def test_deterministic_output_is_independent_of_mapping_insertion_order():
    first = {"z": 1, "nested": {"b": 2, "a": [3, 4]}}
    second = {"nested": {"a": [3, 4], "b": 2}, "z": 1}

    first_bytes = deterministic_json_bytes(first)
    second_bytes = deterministic_json_bytes(second)

    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == first


def test_official_transunet_ledger_reproduces_expected_count():
    result = ledger_summary(
        official_transunet_ledger(image_size=224, num_classes=9)
    )
    components = {
        name: values["total"] for name, values in result["components"].items()
    }

    assert result["parameters"] == {
        "total": 105_277_081,
        "trainable": 105_277_081,
        "non_trainable": 0,
    }
    assert components == {
        "cup_decoder": 7_387_200,
        "final_transformer_norm": 1_536,
        "hybrid_patch_embedding": 787_200,
        "hybrid_resnet_v2": 11_894_848,
        "position_embedding": 150_528,
        "segmentation_head": 1_305,
        "transformer_blocks": 85_054_464,
    }


def test_official_swin_unet_ledger_reproduces_expected_count():
    result = ledger_summary(
        official_swin_unet_ledger(image_size=224, num_classes=9)
    )
    components = {
        name: values["total"] for name, values in result["components"].items()
    }

    assert result["parameters"] == {
        "total": 27_168_900,
        "trainable": 27_168_900,
        "non_trainable": 0,
    }
    assert result["parameter_tensor_count"] == 218
    assert components == {
        "decoder_layers": 6_219_066,
        "encoder_bottleneck_stages": 20_406_954,
        "final_decoder_norm": 192,
        "final_encoder_norm": 1_536,
        "final_patch_expand": 147_648,
        "patch_embedding": 4_896,
        "segmentation_head": 864,
        "skip_concat_projections": 387_744,
    }
