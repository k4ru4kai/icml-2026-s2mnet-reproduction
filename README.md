# S2M-Net DRIVE Reproduction

> **Work in progress.** This repository is being prepared for a controlled
> reproduction study. No empirical reproduction results are available yet.

## Paper

**S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for
Medical Image Segmentation**

- arXiv: [2601.01285v1](https://arxiv.org/abs/2601.01285)
- OpenReview: [eh48NIgu9z](https://openreview.net/forum?id=eh48NIgu9z)
- Official implementation:
  [sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation](https://github.com/sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation)
- Reproduction logbook:
  [Hugging Face Space K4ru4k4i/eh48NIgu9z](https://huggingface.co/spaces/K4ru4k4i/eh48NIgu9z)

## Reproduction objective

The primary experiment is a matched comparison of the released **Full Model**
and the official **No-SSTM** ablation on the DRIVE retinal vessel segmentation
dataset. Dataset split, preprocessing, augmentation, seeds, training budget,
checkpoint selection, thresholding, and evaluation code will be held constant.

The current phase covers source provenance, environment validation, model-build
diagnostics, and experiment design. No dataset has been downloaded and no
training experiment has been run.

## Official code and reproduction code

- `official_repo/` is the upstream implementation, retained as a Git submodule
  and pinned to commit
  `3ec59668ab9b438ab9b170306d29b01e9270fd5a`. It must remain unmodified.
- `repro/` contains reproduction-owned model variants, data utilities,
  evaluation code, and diagnostics.
- `configs/`, `scripts/`, and `tests/` contain reproduction-owned experiment
  configuration, entry points, and validation tests.
- `results/` is reserved for curated tables and figures; large logs,
  checkpoints, datasets, and generated artifacts are not tracked by Git.
- `.trackio/logbook/` contains the source for the public experiment logbook.

The detailed, approval-gated protocol is documented in
[`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md).

## Current status

- Official source pinned and audited.
- TensorFlow 2.15.1 environment validated with CUDA 12.2 and cuDNN 8.
- Released Full Model built successfully for the repository retinal/DRIVE
  configuration and passed one deterministic finite forward-pass diagnostic.
- No DRIVE data, checkpoints, training logs, or empirical Full Model versus
  No-SSTM results are present.

Results and conclusions will be added only after the controlled protocol has
been implemented, reviewed, and executed.
