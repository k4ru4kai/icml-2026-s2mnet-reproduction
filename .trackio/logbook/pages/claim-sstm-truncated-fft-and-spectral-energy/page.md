# Claim - SSTM truncated FFT and spectral energy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_89f8d061b61f", "created_at": "2026-07-17T20:46:36+00:00", "title": "Claim context and planned tests"}
-->
Work in progress. No empirical reproduction verdict has been reached yet.

## Claim under examination

The manuscript describes SSTM as a per-channel truncated two-dimensional FFT mixer that keeps a central K by K frequency window. At K=32, it states that more than 95% of spectral energy is retained and that computational cost is reduced by 63% relative to full spatial attention.

## Planned evidence

1. Add axis-sensitive unit tests using separable sinusoidal inputs to establish that the transform is over height and width for channels-last feature maps.
2. Implement the manuscript sequence explicitly: channel-first spatial FFT, fftshift, centered K by K crop, learned complex filtering, centered zero-padding, ifftshift, and inverse FFT.
3. Measure retained energy as the squared-magnitude sum inside the centered K by K crop divided by total squared-magnitude FFT energy. Report image-level distributions on the fixed DRIVE split and stage-level distributions from matched model features.
4. Compare K=32 with K=16 and full-resolution FFT using the same inputs, with reconstruction error and downstream Dice recorded separately.
5. Compare SSTM and full spatial self-attention at matched tensor shapes and batch size. Report theoretical operation and memory terms together with synchronized latency and peak-memory measurements; the 63% figure will be assessed only for explicitly named cost measures.

## Phase 1 audit note

The paper equations specify a centered frequency crop and zero-pad reconstruction. The pinned implementation instead applies tf.signal.fft2d directly to a channels-last tensor and resizes the complex spectrum. These paths will be kept distinct in the pilot so that the manuscript mechanism and released-code behavior are not conflated. No empirical measurements are present yet.
