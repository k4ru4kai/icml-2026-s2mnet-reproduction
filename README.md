# S2M-Net DRIVE Reproduction

> **Work in progress.** This repository contains a controlled reproduction study.
> The completed DRIVE results reported here are validation-only; no hidden-test
> data were accessed and no universal claim verdict is implied.

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
checkpoint selection, thresholding, and evaluation code are held constant.

## Completed DRIVE comparison

Six runs completed successfully: Full and No-SSTM for seeds 42, 7, and 123.
Every run reached 100 epochs and 24,000 optimizer steps. The reported endpoint
is the model-selected validation FOV macro hard Dice at threshold 0.5.

| Variant | Mean Dice | Sample SD |
|---|---:|---:|
| Full | 0.765522 | 0.003772 |
| No-SSTM | 0.766222 | 0.005349 |

The paired Full minus No-SSTM differences were +0.006319 (seed 42),
-0.004371 (seed 7), and -0.004049 (seed 123), for a mean paired difference of
-0.000700. Under this protocol, the three-seed validation comparison does not
show a stable Dice improvement from SSTM.

These results are limited to three paired seeds and four validation images.
Checkpoint selection and reporting use the same validation set. No test metric
or prediction was produced, and `hidden_test_accessed` remained false.

The full audit, including per-seed metrics, checkpoint selection, run
validation, aggregate calculations, warnings, timing, and exact local artifact
paths, is available in
[`docs/drive_multiseed_report.md`](docs/drive_multiseed_report.md).

## Official code and reproduction code

- `official_repo/` is the upstream implementation, retained as a Git submodule
  and pinned to commit
  `3ec59668ab9b438ab9b170306d29b01e9270fd5a`. It remains unmodified.
- `repro/` contains reproduction-owned model variants, data utilities,
  evaluation code, and diagnostics.
- `configs/`, `scripts/`, and `tests/` contain reproduction-owned experiment
  configuration, entry points, and validation tests.
- `results/` is reserved for curated tables and figures; large logs,
  checkpoints, datasets, and generated artifacts are not tracked by Git.
- `.trackio/logbook/` contains the source for the public experiment logbook.

The detailed protocol is documented in
[`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md).

## Current status

- Official source pinned and audited.
- TensorFlow 2.15.1 environment validated with CUDA 12.2 and cuDNN 8.
- Synthetic Full Model build, forward-pass, and one-step trainability
  diagnostics completed.
- DRIVE Full-versus-No-SSTM campaign completed for three predeclared seeds.
- Validation evidence does not show a stable SSTM Dice advantage under the
  adopted local protocol.
- Hidden-test evaluation has not been performed.
- The cause of the small parameter-count discrepancy between earlier synthetic
  diagnostics and the DRIVE training configuration remains to be resolved
  before finalizing the architectural parameter claim.

The next phase is the architectural evidence audit for Claim 1, followed by
any separately approved hidden-test or manuscript-faithful SSTM diagnostics.
