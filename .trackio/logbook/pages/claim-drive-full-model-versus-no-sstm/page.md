# Claim - DRIVE Full Model versus No-SSTM


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_22edbcf62bfa", "created_at": "2026-07-17T20:46:36+00:00", "title": "Claim context and controlled comparison"}
-->
Work in progress. No empirical reproduction verdict has been reached yet.

## Claim under examination

The manuscript ablation table reports 84.83% Dice for the DRIVE Full Model and 77.78% for No-SSTM, a 7.05 percentage-point difference. This page is reserved for the matched retinal ablation requested for the reproduction.

## Controlled experiment

- Dataset: DRIVE retinal vessel segmentation using the standard public train/test partition and a recorded deterministic validation partition drawn only from training data.
- Primary Full Model arm: the official released Full Model, unchanged and pinned to repository commit 3ec59668ab9b438ab9b170306d29b01e9270fd5a.
- Primary No-SSTM arm: the official released No-SSTM ablation from the same pinned commit.
- Primary comparison controls: identical dataset split, preprocessing, seeds, training budget, and evaluation protocol. This includes the same split manifest, patch extraction, field-of-view masking, normalization, augmentation, epochs, batch size, optimizer and learning-rate schedule, loss, checkpoint rule, inference tiling, threshold selection, and metric implementation.
- Separate diagnostic arm: `paper_faithful_sstm` implements the manuscript-described spatial FFT axes, centered frequency crop and padding, and declared bottleneck. It is intended to measure the consequences of the paper/code discrepancies involving FFT axes, centered crop/padding, and the declared bottleneck. It will be reported separately and will not replace or modify the primary released Full Model versus released No-SSTM comparison.
- Repeats: at least three predeclared seeds after a one-seed smoke test and pilot.
- Primary output: test Dice for each seed and the paired Full Model minus No-SSTM difference with mean, standard deviation, and confidence interval.
- Supporting outputs: IoU, sensitivity, specificity, precision, ROC-AUC, topology-aware vessel diagnostics, parameter count, latency, peak GPU memory, learning curves, and identical-case qualitative masks with thin-vessel and connectivity overlays.

## Phase 1 audit note

The pinned source includes a No-SSTM switch and a retinal configuration, but no DRIVE data, split manifest, checkpoints, or raw run logs. The manuscript table gives 84.83 for the Full Model while the current repository README gives 84.06; both list 77.78 for No-SSTM. The manuscript table is the numeric target for this page. No retinal training or dataset acquisition has started.
