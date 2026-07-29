# S2M-Net Reproduction Study

> **Project status as of 2026-07-28.** Experimental completion and scientific
> claim verification are reported separately. No hidden-test data were
> accessed, and no test result is reported.

## Paper

**S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for
Medical Image Segmentation**

- arXiv: [2601.01285v1](https://arxiv.org/abs/2601.01285)
- OpenReview: [eh48NIgu9z](https://openreview.net/forum?id=eh48NIgu9z)
- Official implementation:
  [sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation](https://github.com/sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation)
- Reproduction logbook:
  [Hugging Face Space K4ru4k4i/eh48NIgu9z](https://huggingface.co/spaces/K4ru4k4i/eh48NIgu9z)

## Study scope

This repository combines controlled DRIVE experiments with executable
architecture and claim audits. The upstream implementation is pinned and kept
unchanged in `official_repo/`.

## Completed validation work

### Phase 2A

The synthetic Full Model validation completed successfully: model
construction, deterministic forward execution, finite gradients, and one
diagnostic optimizer update all passed. This was infrastructure validation,
not a dataset result.

### DRIVE Full versus No-SSTM

Six runs completed successfully: Full and No-SSTM for seeds 42, 7, and 123.
Every run used batch size 2 and reached 100 epochs and 24,000 optimizer steps.
The reported endpoint is model-selected validation supplied-FOV macro hard
Dice at threshold 0.5.

| Variant | Validation Dice, mean ± sample SD |
|---|---:|
| Full | **0.765522 ± 0.003772** |
| No-SSTM | **0.766222 ± 0.005349** |

The mean paired **Full − No-SSTM** difference is **−0.000700**. Under this
limited protocol, the three-seed comparison does not show a stable SSTM Dice
advantage.

These results are limited to three paired seeds and four validation images.
Checkpoint selection and reporting use the same validation set. No test metric
or prediction was produced; every run records
`hidden_test_accessed: false`. This result does not establish a universal
paper-claim verdict.

See the [complete DRIVE multi-seed report](docs/drive_multiseed_report.md) for
the per-seed metrics, checkpoint selection, run validation, warnings, timing,
and exact artifact provenance.

## Claim 1: architecture and parameter efficiency

The [executable Claim 1 audit](repro/diagnostics/verify_claim1_architecture.py)
passes. It verifies the canonical released S2M-Net build at 352 × 352 as
**4,791,544 parameters** and confirms five encoder stages with channels
**{24, 32, 64, 80, 128}**.

The scientific verdict remains **Partially verified**. The architecture and
exact released count are reproducible, but unresolved provenance for the
paper's TransUNet and Swin-Unet comparison configurations prevents full
verification of the parameter-efficiency ratios. Claim 1 may be revisited if
new official materials become available.

Evidence:

- [Claim 1 parameter-count investigation](docs/claim1_parameter_count_investigation.md)
- [Executable audit](repro/diagnostics/verify_claim1_architecture.py)
- [Deterministic JSON result](results/audits/claim1_architecture_parameters.json)

`Audit PASS` means the verification procedure passed; it does not mean every
part of the scientific claim was reproduced.

## Claim 2: SSTM mechanism, spectral energy, and cost

The scientific verdict is **Not verified**. The
[executable Claim 2 audit](repro/diagnostics/verify_claim2_sstm.py) passes, but
it shows that the released SSTM does not implement the centered spatial
K×K truncation described by the paper:

- the canonical configuration requests K=32, with effective values
  **{32, 32, 32, 22, 11}** across the five stages;
- `tf.signal.fft2d` is called directly on NHWC tensors, so the transformed axes
  are width and channels rather than height and width;
- the complex tensor is bilinearly resized instead of using `fftshift`,
  centered cropping, zero-padding, and `ifftshift`;
- the standard retained-subset energy ratio is therefore not defined for the
  released forward path; and
- the paper and released repository do not specify a reproducible matched
  attention baseline or cost metric for the reported 63% reduction.

A limited paper-intended diagnostic on four permitted DRIVE validation images
exceeded 95% for raw images, but it does not verify the general claim and
trained pre-SSTM features at stages 1–3 remained below 95%.

Evidence:

- [Complete Claim 2 investigation](docs/claim2_sstm_verification.md)
- [Executable audit](repro/diagnostics/verify_claim2_sstm.py)
- [Deterministic JSON result](results/audits/claim2_sstm_audit.json)

As for Claim 1, `Audit PASS` describes the reproducibility procedure, not the
scientific verdict.

## Repository guide

- `official_repo/` is the upstream implementation, retained as a Git submodule
  and pinned to commit
  `3ec59668ab9b438ab9b170306d29b01e9270fd5a`. It remains unmodified.
- `repro/` contains reproduction-owned model variants, data utilities,
  experiment configuration, evaluation code, and diagnostics.
- `scripts/` and `tests/` contain reproduction-owned entry points and
  validation tests.
- `results/` is reserved for curated tables and figures; large logs,
  checkpoints, datasets, and generated artifacts are not tracked by Git.
- `.trackio/logbook/` contains the source for the public experiment logbook.

The [historical reproduction plan](REPRODUCTION_PLAN.md) retains the original
approval-gated protocol and now records execution deviations explicitly. The
[public Trackio logbook](https://huggingface.co/spaces/K4ru4k4i/eh48NIgu9z)
provides claim-oriented summaries.

## Current status

Claims 1 and 2 now have explicit scientific verdicts and executable audits.
No work on the remaining performance claims is implied by this status.
