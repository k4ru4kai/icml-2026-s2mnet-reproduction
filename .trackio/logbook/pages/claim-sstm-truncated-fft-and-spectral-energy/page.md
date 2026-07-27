# Claim - SSTM truncated FFT and spectral energy


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_89f8d061b61f", "created_at": "2026-07-17T20:46:36+00:00", "title": "Claim context and planned tests"}
-->
## Claim under examination

> “The Spectral-Selective Token Mixer (SSTM) uses a truncated 2D FFT
> retaining K=32 frequency components, capturing more than 95% of spectral
> energy while reducing computational cost by 63% relative to full spatial
> attention.”

## Verdict

**Not verified.**

The paper-intended mechanism, the released forward implementation, the
limited local energy diagnostics, and the undocumented cost comparison must
be treated separately. The released implementation does not perform the
paper's centered spatial-frequency truncation; the energy evidence is narrow
and inconsistent across raw images and trained features; and the reported
cost reduction lacks a reproducible baseline and counting protocol.

## Evidence classification

- **Paper-intended operation:** the operation defined by the paper's equations
  and prose, irrespective of whether the released forward path implements it.
- **Released forward implementation:** behavior traced directly from the
  canonical Full configuration into the instantiated SSTM layer and verified
  with inference-only shape and FFT-axis diagnostics.
- **Limited diagnostic evidence:** measurements on the four permitted,
  non-hidden DRIVE validation images and trained pre-SSTM features from the
  three already-existing selected Full checkpoints.
- **Not reproducible from available documentation:** a reported value for
  which the paper and repository do not specify enough of the metric,
  baseline, configuration, or counting procedure to recreate it.

## Truncated spatial 2D FFT and K=32

**Component verdict: partially verified as the paper-intended specification,
but contradicted as a description of the released forward implementation.**

### Paper-intended operation

The paper defines K as the side length of a centered spatial-frequency crop:
K=32 means a **32×32 region per channel**, or 1,024 complex coefficients per
channel. Its equations describe:

1. a per-channel two-dimensional FFT over height and width;
2. `fftshift`;
3. a centered K×K crop;
4. learned K×K×C spectral filtering;
5. centered zero-padding back to H×W;
6. `ifftshift`; and
7. an inverse spatial FFT.

This evidence is in `sources/alphaxiv-2601.01285v1.txt:160-178`, with
additional K descriptions at lines 81–83, 127–130, and 189–200.

### Released forward implementation

The canonical configuration requests `sstm_k: 32`, but the released layer
uses K as the requested side length of a **bilinearly resized complex tensor
and learned real-weight grid**. It clamps that side length to the feature-map
dimensions:

```text
actual_k = min(configured K, H, W)
```

The effective values are therefore **{32, 32, 32, 22, 11}**:

| Stage | SSTM input shape | Configured K | Effective K | Released spectral operation | Actual FFT axes |
|---:|---|---:|---:|---|---|
| 1 | B×176×176×24 | 32 | 32 | Resize to B×32×32×24 | W,C |
| 2 | B×88×88×32 | 32 | 32 | Resize to B×32×32×32 | W,C |
| 3 | B×44×44×64 | 32 | 32 | Resize to B×32×32×64 | W,C |
| 4 | B×22×22×80 | 32 | 22 | B×22×22×80; no H/W reduction | W,C |
| 5 | B×11×11×128 | 32 | 11 | B×11×11×128; no H/W reduction | W,C |

The released NHWC path calls `tf.signal.fft2d` directly. TensorFlow transforms
the innermost two dimensions, so the actual axes are **width and channels
(W,C)**, not height and width (H,W). There is no transpose before the call.
An independent diagnostic matched a NumPy FFT on W,C to within
5.573143×10^-7, whereas the maximum difference from an H,W FFT was
13.180659.

The forward code contains **no centered crop, no `fftshift`, no `ifftshift`,
and no zero-padding**. It separately bilinearly resizes the real and imaginary
parts, applies a learned real k×k×C grid, resizes back, and calls the inverse
FFT on W,C. Consequently, it does not retain a discrete spectral subset.

Exact local source paths:

- `docs/claim2_sstm_verification.md`, sections 2–4;
- `repro/configs/drive_full_seed42.yaml:18-37`;
- `outputs/drive/full_seed42/config/frozen.yaml`;
- `repro/experiments/drive_train.py:490-501`;
- `official_repo/train.py:153-175`;
- `official_repo/s2mnet/models/s2mnet.py:94-127`; and
- `official_repo/s2mnet/models/blocks.py:30-194`.

## More than 95% retained spectral energy

**Component verdict: not verified for the released SSTM; supported only by a
limited four-image raw-DRIVE diagnostic under the paper-intended spatial
interpretation.**

For the paper-intended diagnostic, retained energy was defined explicitly as:

```text
sum(|centered_crop_K(fftshift(FFT_H,W(x)))|²)
------------------------------------------------
sum(|fftshift(FFT_H,W(x))|²)
```

The arithmetic mean of per-channel ratios was used, matching the released
analysis utility at `official_repo/s2mnet/utils/spectral.py:15-41`. That
utility measures the paper-intended centered spatial crop on raw images; it
does not reproduce the released SSTM forward path.

### Limited raw-image diagnostic

The four non-hidden DRIVE validation images, IDs 37–40, gave:

| Diagnostic protocol | Images | Mean retained energy |
|---|---:|---:|
| Released raw-RGB analysis protocol | 4 | **98.892700%** |
| Frozen DRIVE training preprocessing | 4 | **97.269658%** |

Both values exceed 95%, but they are limited to four validation images from
one fundus dataset. They do not reproduce the paper's undocumented
cross-dataset protocol.

### Trained pre-SSTM feature diagnostic

Using the three existing selected Full checkpoints and the same four
validation images, the paper-intended spatial interpretation gave:

| Stage | Effective K | Mean retained energy |
|---:|---:|---:|
| 1 | 32 | **52.005674%** |
| 2 | 32 | **69.157743%** |
| 3 | 32 | **91.705598%** |
| 4 | 22 | **100%** |
| 5 | 11 | **100%** |

Stages 1–3 do not reach 95%. Stages 4–5 reach 100% because effective K covers
their complete 22×22 and 11×11 spatial grids, respectively; under this
diagnostic interpretation, no spatial-frequency coefficient is excluded.

For the **released resize-based operation**, retained spectral energy is
undefined: bilinear interpolation blends the full tensor rather than selecting
a discrete set of coefficients, so the standard retained-subset numerator
does not exist.

The paper does not specify whether its 95% statement concerns raw images or
trained features, which datasets and sample counts were used, the stage,
preprocessing, channel aggregation, or formula. The repository README also
reports 94.8% overall and 96.4% for fundus images, rather than documenting a
reproducible “more than 95% across datasets” result.

Exact local source paths:

- `docs/claim2_sstm_verification.md`, section 5;
- `sources/alphaxiv-2601.01285v1.txt:73-75,127-130,369-415`;
- `official_repo/s2mnet/utils/spectral.py:15-41,74-103`;
- `official_repo/scripts/analyze_spectral.py:195-243`;
- `official_repo/README.md:18,302-314`;
- `outputs/drive/full_seed42/config/frozen.yaml`; and
- the existing selected checkpoints listed in
  `docs/claim2_sstm_verification.md`, section 5.3.

## 63% computational-cost reduction

**Component verdict: not verified and not reproducible from the available
paper or released repository.**

The paper alternates between “frequency cost,” “cost reduction,” asymptotic
complexity, and a separate reported 4.2× inference speedup. It does not define
the quantity behind 63%. No cost metric, full-spatial-attention
configuration, tensor shape, stage, token count, channel dimension, number of
heads, batch size, precision, counting convention, hardware protocol, or
executable matched baseline is specified. The released repository contains no
supporting derivation, profiler output, matched-attention module, benchmark
script, or saved result.

Programmatic arithmetic checks of plausible alternative interpretations give
reductions spanning approximately **75%–99%**, but none independently
validates 63%:

| Available comparison or interpretation | Calculated reduction |
|---|---:|
| README whole-model FLOPs: 11.2G vs. 45.2G | 75.221239% |
| README whole-model runtime: 10.1 ms vs. 42.3 ms | 76.122931% |
| Paper-reported 4.2× speedup converted to a reduction | 76.190476% |
| 32 coefficients vs. the paper's unexplained 256 | 87.500000% |
| 32×32 coefficients vs. a full 352×352 grid | 99.173554% |

These calculations compare different quantities and are audit checks, not a
like-for-like SSTM-versus-attention reproduction. Creating an arbitrary
attention module would require undocumented design choices and would not
verify the paper's result.

Exact local source paths:

- `docs/claim2_sstm_verification.md`, section 6;
- `sources/alphaxiv-2601.01285v1.txt:73-75,86-90,149-159,195-198`;
- `official_repo/README.md:83-91,348-364`; and
- `official_repo/s2mnet/models/baselines.py:99-181`, which contains a
  non-matching TransUNet bottleneck rather than the required matched
  full-spatial-attention baseline.

## Component summary

| Claim component | Conservative assessment |
|---|---|
| Truncated spatial 2D FFT with K=32 | Paper specification partially verified; released forward implementation contradicts it |
| More than 95% retained energy | Not verified for released SSTM; only limited raw-DRIVE support under the paper-intended interpretation |
| 63% lower computational cost | Not verified; baseline and cost protocol are not reproducible |

## Conclusion

Claim 2 is not verified. The paper specifies a centered 32×32 spatial-frequency crop, but the released SSTM instead bilinearly resizes the complex tensor and learned real-weight grid, with K clamped to the feature-map dimensions. Moreover, tf.signal.fft2d is applied directly to NHWC tensors, so it transforms width and channels rather than height and width. The more-than-95% energy statement is supported only by a limited four-image DRIVE diagnostic under the paper-intended spatial interpretation and is not consistently supported by trained pre-SSTM features. The reported 63% computational-cost reduction cannot be reproduced because the paper and repository do not define the baseline or cost-counting protocol.

## Authoritative evidence record

All scientific values and interpretations on this page are sourced from the
completed local investigation:
`docs/claim2_sstm_verification.md`.
