# Phase 2A — Synthetic model validation


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2a7f6d38c914", "created_at": "2026-07-21T20:37:22+00:00", "title": "Synthetic infrastructure validation"}
-->
Work in progress. Phase 2A is infrastructure validation only; no empirical reproduction results or performance conclusions are available yet.

## Verified synthetic validation

- The official Full Model was built using `train.build_model` and `official_repo/configs/retinal.yaml`.
- The verified topology contains 5 `MRF-SE`, 5 `SSTM`, and 4 `BFP` stages with `soft` routing.
- Total parameters: 4,766,008.
- Trainable parameters: 4,743,384.
- A deterministic synthetic forward pass completed successfully.
- The output and weights contained no `NaN` or `Inf` values.
- All 305 trainable variables received finite gradients.
- One diagnostic `Adam` update changed all 305 trainable variables.
- `OVERALL_STATUS=PASS`.
- TensorFlow emitted `complex64 → float32` warnings during the `SSTM` backward path, without missing or non-finite gradients.

## Scope limitations

Binary Cross-Entropy was used only as a diagnostic loss. It is not the paper's Morphology-Aware Loss. DRIVE has not been downloaded, accessed, trained on, or evaluated. This page adds no claim-level empirical evidence. The reproduction remains a work in progress.

## Reproducibility links

- [Phase 2A validation report](https://github.com/k4ru4kai/icml-2026-s2mnet-reproduction/blob/f76d4251cb518ac577946e9bf7d7e682b0c9eb7c/docs/phase2a_validation.md)
- [Synthetic forward-pass diagnostic](https://github.com/k4ru4kai/icml-2026-s2mnet-reproduction/blob/f76d4251cb518ac577946e9bf7d7e682b0c9eb7c/repro/diagnostics/phase2a_full_model_drive.py)
- [Synthetic trainability diagnostic](https://github.com/k4ru4kai/icml-2026-s2mnet-reproduction/blob/f76d4251cb518ac577946e9bf7d7e682b0c9eb7c/repro/diagnostics/phase2a_full_model_trainability.py)
- [Commit `f76d4251cb518ac577946e9bf7d7e682b0c9eb7c`](https://github.com/k4ru4kai/icml-2026-s2mnet-reproduction/commit/f76d4251cb518ac577946e9bf7d7e682b0c9eb7c)
