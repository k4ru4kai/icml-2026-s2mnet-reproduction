# Introduction and scope


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_e310a1c75fae", "created_at": "2026-07-17T20:46:36+00:00", "title": "Context and scope"}
-->
Work in progress. No empirical reproduction verdict has been reached yet.

## Paper and authoritative sources

**S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss for Medical Image Segmentation**

- [arXiv abstract](https://arxiv.org/abs/2601.01285)
- [arXiv HTML](https://arxiv.org/html/2601.01285v1)
- [arXiv PDF](https://arxiv.org/pdf/2601.01285)
- [OpenReview forum](https://openreview.net/forum?id=eh48NIgu9z)

## Official implementation

- Repository: [sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation](https://github.com/sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation)
- Pinned commit: 3ec59668ab9b438ab9b170306d29b01e9270fd5a

## Selected scope

The reproduction focuses on the SSTM truncated two-dimensional FFT mechanism and retinal vessel segmentation. The primary controlled experiment is DRIVE Full Model versus No-SSTM, with all comparison settings held constant: dataset split, seed set, preprocessing, augmentation, epochs, batch size, optimizer schedule, checkpoint selection, thresholding, and evaluation code.

## Planned DRIVE comparison

The plan uses the standard DRIVE train/test partition, defines one deterministic validation partition from training data, and runs the Full Model at K=32 against the No-SSTM ablation under the same protocol. Primary reporting is test-set Dice with paired seed-level differences; IoU, sensitivity, specificity, precision, ROC-AUC, parameter count, latency, peak GPU memory, and vessel-focused qualitative overlays are supporting outputs.

## Completed Phase 1 static audit

The challenge guide, arXiv HTML/PDF, and official source tree at the pinned commit were inspected. The audit traced the five-stage model configuration, SSTM block, spectral-analysis helper, retinal configuration and loaders, loss implementation, and ablation switches. It also recorded missing experiment assets and protocol details, including absent data, checkpoints, raw logs, and explicit split manifests, plus unpinned dependencies. A code-level risk was identified: the released channels-last SSTM passes its tensor directly to tf.signal.fft2d, which operates on the final two axes, and its spectral reduction uses complex resizing rather than the centered crop and zero-pad sequence described in the manuscript. No dataset download or training run was performed.
