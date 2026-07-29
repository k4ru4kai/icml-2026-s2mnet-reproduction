# Claim 2 SSTM verification

- Local diagnostic investigation: 2026-07-27
- Executable reproducibility package added: 2026-07-28

Canonical Claim 2:

> “The Spectral-Selective Token Mixer (SSTM) uses a truncated 2D FFT
> retaining K=32 frequency components, capturing more than 95% of spectral
> energy while reducing computational cost by 63% relative to full spatial
> attention.”

## Scope and safeguards

This is a diagnostic investigation, not a training experiment. The scientific
verdict is kept separate from the technical result of the executable audit.

- No model was trained or fine-tuned.
- No dataset or checkpoint was downloaded.
- No hidden-test image, mask, prediction, or metric was accessed.
- Only the predeclared non-hidden DRIVE validation IDs 37–40 were read.
- Existing selected Full checkpoints were restored read-only for
  inference-only feature inspection.
- During the original 2026-07-27 data-dependent diagnostic, no project source,
  configuration, dataset, checkpoint, run, Claim 1 file, Trackio page, or
  external service was modified.
- The original 2026-07-27 diagnostic used temporary code and
  machine-readable output outside the repository at
  `/tmp/claim2_energy_diagnostics.py` and
  `/tmp/claim2_energy_diagnostics.json`.
- On 2026-07-28, the non-private and deterministic parts of the investigation
  were converted into the versioned executable audit
  `repro/diagnostics/verify_claim2_sstm.py`, its tests, and a curated JSON
  result. The original temporary files are not treated as public evidence.

The released repository was clean at pinned commit
`3ec59668ab9b438ab9b170306d29b01e9270fd5a`. The principal paper artifact was
the local nine-page arXiv v1 PDF
`sources/alphaxiv-2601.01285v1.pdf`, with extracted text at
`sources/alphaxiv-2601.01285v1.txt`. The released implementation and its
documentation are not assumed to implement the paper equations; they are
examined separately below.

The public, version-pinned source corresponding to that local artifact is
[arXiv v1 HTML](https://arxiv.org/html/2601.01285v1), especially Section 3.3
and equations 4–8. The current arXiv record states that a later v2 withdrew the
paper because it contains issues requiring revision. This audit continues to
evaluate the challenge-pinned v1 artifact; the later withdrawal is provenance
context, not a substitute for the evidence below.

## Reproducibility package

The public package now contains:

- `repro/diagnostics/verify_claim2_sstm.py`: source/AST audit of the pinned
  SSTM, effective-K trace, paper-intended energy function, optional
  explicit-image energy analysis, and arithmetic cost checks;
- `tests/test_claim2_sstm_audit.py`: six deterministic tests; and
- `results/audits/claim2_sstm_audit.json`: the checked-in result without
  private data or checkpoints.

Run the public deterministic audit with:

```bash
python repro/diagnostics/verify_claim2_sstm.py
python tests/test_claim2_sstm_audit.py -v
```

The image diagnostic accepts only explicitly named files:

```bash
python repro/diagnostics/verify_claim2_sstm.py \
  --image /path/to/permitted/image_37.tif \
  --image /path/to/permitted/image_38.tif \
  --image /path/to/permitted/image_39.tif \
  --image /path/to/permitted/image_40.tif \
  --output /tmp/claim2_drive_energy.json
```

Directories are deliberately unsupported so the audit cannot silently scan a
dataset or access hidden-test images. The checked-in JSON leaves this optional
field null because DRIVE images and checkpoints are not versioned. Therefore
the 2026-07-27 raw-image and trained-feature values remain local
data-dependent evidence; the public package independently reproduces the
mechanism trace and cost arithmetic and provides the exact raw-image
calculation for permitted inputs.

## 1. Claim decomposition

Claim 2 contains three independently testable components:

1. **Mechanism and K:** SSTM performs a truncated spatial 2D FFT and K=32
   specifies the retained frequency components.
2. **Energy:** the retained components contain more than 95% of spectral
   energy.
3. **Cost:** this mechanism reduces computational cost by 63% relative to full
   spatial attention.

These components need separate assessments because:

- a configuration can contain the integer 32 without implementing the claimed
  crop;
- an energy claim is meaningful only after the FFT axes and retained subset
  are known; and
- “cost” can denote coefficient count, FLOPs, MACs, asymptotic complexity,
  runtime, or memory.

## 2. SSTM source and configuration trace

### 2.1 Canonical construction path

The paper-consistent Full configuration at 352×352 is traced as follows:

1. `repro/configs/drive_full_seed42.yaml:18-37` and the frozen run copy
   `outputs/drive/full_seed42/config/frozen.yaml` specify:

   - `input_size: 352`;
   - `filters: [24, 32, 64, 80, 128]`;
   - `use_sstm: true`;
   - `sstm_k: 32`;
   - `sstm_ssm_dim: 16`;
   - `sstm_stages: [true, true, true, true, true]`;
   - `sstm_use_spectral: [true, true, true, true, true]`; and
   - `sstm_use_ssm: [false, false, true, true, true]`.

2. These values reproduce the released defaults in
   `official_repo/configs/default.yaml:10-25`.

3. `repro/experiments/drive_train.py:490-501` copies the frozen model
   configuration into the released builder.

4. `official_repo/train.py:153-175` passes `sstm_k` to `S2MNet`.

5. `official_repo/s2mnet/models/s2mnet.py:94-127` creates one
   `SpectralSelectiveTokenMixer` at every enabled encoder stage and passes:

   ```text
   num_frequencies = sstm_k
   ssm_state_dim   = sstm_ssm_dim
   use_spectral    = sstm_use_spectral[i]
   use_ssm         = sstm_use_ssm[i]
   ```

6. The layer implementation is
   `official_repo/s2mnet/models/blocks.py:30-194`.

The released ablation registry also calls `sstm_k` the “SSTM Truncation Size
K” and defines K=16, 24, 48, and 64 alternatives at
`official_repo/experiments/ablation_configs.py:119-132`. It contains no
self-attention replacement ablation.

### 2.2 Branch placement

All five encoder stages instantiate SSTM and enable the released spectral
path. Stages 1–2 are **spectral-only**. Stages 3–5 enable both the spectral
path and the layer called the SSM/selective path.

The selective path at `official_repo/s2mnet/models/blocks.py:152-160`:

1. flattens H×W to a token axis;
2. applies a C-to-C sigmoid gate independently at each token;
3. applies a second C-to-C dense projection; and
4. reshapes back to NHWC.

It has no recurrence, scan, state transition, or cross-token operation.
`ssm_state_dim=16` is stored but unused
(`official_repo/s2mnet/models/blocks.py:45-46,69-74,96-99,152-160`).
Consequently, it is a gated per-location channel projection, not an
implemented state-space model or the paper’s declared d=16 bottleneck.

When both paths are active, their 2C-channel concatenation is projected to C
with a dense fusion layer
(`official_repo/s2mnet/models/blocks.py:101-103,172-174`). Spectral-only
stages skip this fusion projection.

## 3. Meaning and effective value of K

### 3.1 Paper meaning

The paper’s intended meaning is unambiguous in its method equations:

- `sources/alphaxiv-2601.01285v1.txt:160-178` describes a per-channel spatial
  2D FFT, `fftshift`, a centered K×K crop, learned K×K×C weights, centered
  zero-padding to H×W, `ifftshift`, and inverse spatial FFT.
- Equations 4–7 explicitly give
  `X_crop ∈ complex^(K×K×C)`.
- `sources/alphaxiv-2601.01285v1.txt:81-83,127-130,189-200` repeatedly calls
  K=32 a K×K central-frequency region.

Thus, in the **paper-intended method**, K is the side length per spatial
frequency axis: K=32 means a 32×32 centered region per channel, or 1,024
complex coefficients per channel before learned filtering. It does not mean
32 coefficients total.

The introduction’s phrase “K=32 of 256 coefficients” at
`sources/alphaxiv-2601.01285v1.txt:73-75` is inconsistent with the equations:
32/256 is 12.5%, whereas a 32×32 crop contains 1,024 coefficients. The
available paper does not explain what “256” denotes.

### 3.2 Released-code meaning

In the released layer, `num_frequencies` is operationally a requested
**square resize side length and learned-weight-grid side length**, not a count
of retained coefficients:

```python
H, W = input_shape[1], input_shape[2]
self.actual_k = min(self.num_frequencies, H, W)
self.freq_weights.shape = (self.actual_k, self.actual_k, self.channels)
```

Source:
`official_repo/s2mnet/models/blocks.py:79-94`.

At runtime the same clamping is repeated:

```python
k = min(H, W, self.actual_k)
```

Source:
`official_repo/s2mnet/models/blocks.py:127-130`.

The code then uses `tf.image.resize(..., [k, k])`; it never slices a K×K
frequency subset. Therefore:

- K is not 32 total coefficients;
- K is not a literal retained count per axis in the released forward pass;
- K does determine an operational K×K **resampled tensor shape** and a
  K×K×C real learned-weight tensor; and
- K is fixed at the configured value 32 only while both spatial feature
  dimensions are at least 32. It is clamped to 22 and 11 at stages 4 and 5.

### 3.3 Effective stage values and branches

The following shapes were confirmed by constructing the canonical model and
running read-only inference diagnostics. “Operational spectral tensor” is
the tensor produced by complex bilinear resize. It is not a retained subset.
FFT axes are zero-based for rank-4 NHWC input.

| Stage | SSTM input feature shape | Configured K | `actual_k` | Operational resampled complex shape | Literal retained spectral shape | FFT axes | Active branches |
|---:|---|---:|---:|---|---|---|---|
| 1 | B×176×176×24 | 32 | 32 | B×32×32×24 | None; no crop | (2,3) = W,C | spectral only |
| 2 | B×88×88×32 | 32 | 32 | B×32×32×32 | None; no crop | (2,3) = W,C | spectral only |
| 3 | B×44×44×64 | 32 | 32 | B×32×32×64 | None; no crop | (2,3) = W,C | spectral + gated dense |
| 4 | B×22×22×80 | 32 | 22 | B×22×22×80 | None; no crop | (2,3) = W,C | spectral + gated dense |
| 5 | B×11×11×128 | 32 | 11 | B×11×11×128 | None; no crop | (2,3) = W,C | spectral + gated dense |

Under the paper-intended H×W crop interpretation, the nominal coefficient
fractions would be 3.305785%, 13.223140%, 52.892562%, 100%, and 100% at
stages 1–5. Stages 4–5 therefore would not be truncated even under that
interpretation.

## 4. FFT axes and tensor-layout verification

### 4.1 Actual released operation, step by step

For an input `x` with channels-last layout B×H×W×C, the released spectral path
at `official_repo/s2mnet/models/blocks.py:127-150` performs:

1. `x` is cast from real float to `complex64`.
2. `tf.signal.fft2d(x)` is called directly, with no transpose.
3. TensorFlow 2.15.1 defines `fft2d` over the **inner-most two dimensions**.
   On B×H×W×C, these are W and C, axes `(2,3)`, not spatial H and W.
4. There is no `fftshift`.
5. The real and imaginary components of the B×H×W×C result are separately
   passed to `tf.image.resize(..., [k,k], method="bilinear")`. For NHWC
   tensors, this resizes positions corresponding to H and W, producing
   B×k×k×C.
6. No frequency coefficient is selected by an index or mask. Bilinear
   resampling blends information from the full tensor; it is not a centered
   low-frequency crop.
7. A trainable **real** tensor of shape k×k×C is cast to complex and multiplied
   element-wise. It is not a general complex filter.
8. The filter initializer uses `np.fft.fftfreq` on the first two weight-grid
   dimensions without a shift and applies a radial Gaussian around normalized
   magnitude 0.25
   (`official_repo/s2mnet/models/blocks.py:111-125`). Those dimensions do not
   align with the actual W,C FFT axes after the resize.
9. The complex tensor is bilinearly resized back to H×W positions. There is no
   centered zero-padding and no `ifftshift`.
10. `tf.signal.ifft2d` again transforms the inner-most W,C dimensions.
11. The real part is retained and the imaginary part discarded.
12. Layer normalization is applied, followed later by output normalization
    and a residual addition.

Because the forward path has no discrete retained subset, “discarded
coefficients” do not exist in the paper’s mask/crop sense. Resolution is lost
through interpolation in stages 1–3; stages 4–5 do not reduce the H/W-shaped
tensor at all.

The learned filtering and interpolation need not preserve the Hermitian
symmetry required for a strictly real inverse spatial signal. The released
code simply discards the inverse transform’s imaginary component.

### 4.2 Independent axis diagnostic

An inference-only diagnostic compared TensorFlow with NumPy on a random
1×5×7×3 NHWC tensor:

| Comparison | Maximum absolute error |
|---|---:|
| `tf.signal.fft2d` vs. NumPy FFT on W,C axes `(2,3)` | 5.573143×10^-7 |
| `tf.signal.fft2d` vs. NumPy spatial FFT on H,W axes `(1,2)` | 13.180659 |

This independently confirms that the released call transforms W and C.

### 4.3 Paper equations versus released code

| Operation | Paper, equations 4–7 | Released implementation |
|---|---|---|
| Input layout | H×W×C, per-channel spatial FFT | NHWC |
| FFT dimensions | H and W per channel | W and C |
| Transpose before FFT | Implied as needed for spatial axes | None |
| `fftshift` | Required | Absent |
| Retained region | Centered K×K slice | No slice; bilinear resize |
| K=32 | 32×32 spatial-frequency region | Resize/weight-grid side, clamped by H,W |
| Discarded coefficients | Removed outside crop | No explicit discard or mask |
| Learned weights | K×K×C spectral filter | Real k×k×C tensor on resampled coordinates |
| Restore to H×W | Centered zero-padding | Bilinear resize |
| `ifftshift` | Required | Absent |
| Inverse FFT axes | Spatial H,W | W,C |
| Spatial branch | d=16 bottleneck | Two C-to-C dense layers; d=16 unused |
| Dual branches | Described as SSTM mechanism generally | Stages 1–2 spectral-only; stages 3–5 dual |

## 5. Spectral-energy protocol and results

### 5.1 What the paper and repository document

The paper states:

- K=32 preserves 95% energy
  (`sources/alphaxiv-2601.01285v1.txt:73-75`);
- a centered K×K crop retains more than 95% “across datasets”
  (`sources/alphaxiv-2601.01285v1.txt:127-130`);
- K=32 retains about 95% while K=16 retains about 85%
  (`sources/alphaxiv-2601.01285v1.txt:369-415`); and
- only about 1% of coefficients are retained
  (`sources/alphaxiv-2601.01285v1.txt:189-190`).

The paper does **not** specify:

- the datasets and sample counts in the energy calculation;
- whether raw images or encoder activations were analyzed;
- whether activations were trained or untrained;
- a particular SSTM stage;
- preprocessing;
- channel aggregation;
- per-image versus globally pooled aggregation;
- the energy formula; or
- an executable script or result table in the accessible nine-page artifact.

The text repeatedly refers to “medical images,” which weakly suggests raw
input images. It also places the claim inside the SSTM feature-map discussion,
so the exact target remains ambiguous.

The released repository supplies an executable **raw-image** protocol:

- `official_repo/s2mnet/utils/spectral.py:15-41` computes a per-channel
  spatial FFT, `fftshift`, centered K×K crop, squared-magnitude ratio, then
  averages channel ratios.
- `official_repo/s2mnet/utils/spectral.py:74-103` resizes RGB images to
  352×352 and supports at most 50 samples.
- `official_repo/scripts/analyze_spectral.py:195-243` exposes that utility.

This analysis utility implements the **paper-intended spatial crop**, not the
released SSTM forward path.

The repository README is internally more conservative than the paper:

- `official_repo/README.md:18` says K=32 captures more than 93%;
- `official_repo/README.md:302-314` reports 94.8% average across four modality
  groups and 96.4% for fundus images; and
- no raw result file, image list, seed, or derivation for that table is
  included.

Thus the repository’s own reported overall mean is below the canonical
“more than 95%” threshold, although its fundus row exceeds it.

### 5.2 Explicit formula used here

For the paper-intended diagnostic, this investigation used:

```text
retained-energy ratio
  = sum(|centered_crop_K(fftshift(FFT_H,W(x)))|²)
    / sum(|fftshift(FFT_H,W(x))|²)
```

For multichannel tensors, the primary result is the arithmetic mean of the
per-channel ratios, matching
`official_repo/s2mnet/utils/spectral.py:27-30`. A pooled ratio was also
checked; it is not substituted silently for the released utility’s
convention.

No corresponding standard ratio is calculated for the released forward path
because bilinear interpolation is not a subset operation. Comparing
`sum(|resize(F)|²)` with `sum(|F|²)` would mix coefficient energy with grid
size and interpolation scaling and would not measure retained energy.

### 5.3 Existing permitted artifacts

The frozen split at
`outputs/drive/full_seed42/config/frozen.yaml` assigns IDs 37–40 to validation,
IDs 21–36 to training, IDs 01–20 to hidden test, and records
`access_hidden_test: false`.

The following selected Full checkpoints already existed and were restored
read-only:

| Seed | Selected checkpoint | Selected epoch | Checkpoint data SHA-256 |
|---:|---|---:|---|
| 42 | `outputs/drive/full_seed42/checkpoints/best/ckpt-21` | 21 | `0a106a8f0a2385a2b74479755789bd1d6014977976c6d248ea0b7241bd4a4b45` |
| 7 | `outputs/drive/full_seed7/checkpoints/best/ckpt-14` | 14 | `205593f14cd9826a600fbe3606052047da5dbe519070aea1d90a5a20b054869f` |
| 123 | `outputs/drive/full_seed123/checkpoints/best/ckpt-24` | 24 | `eac11f3a67667cdbe503873e885fd29806065fc55a2fe9810620cb7c65e67540` |

Checkpoint selection and completion evidence is in each run’s `status.json`
and `checkpoints/best/checkpoint`. No No-SSTM checkpoint is relevant to
pre-SSTM feature-energy measurement because that architecture contains no
SSTM layers.

### 5.4 Raw-image diagnostics on non-hidden DRIVE validation data

Two explicit raw-image protocols were evaluated on validation IDs 37–40:

1. **Released analysis-script protocol:** RGB, OpenCV resize to 352×352 using
   the utility default, divide by 255, no training CLAHE or FOV masking.
2. **Frozen DRIVE training preprocessing:** green-channel CLAHE with clip
   limit 2.0 and 8×8 tiles, replicate to three channels, Lanczos4 resize to
   352×352, clip to [0,1], and apply the supplied FOV mask
   (`repro/experiments/drive_train.py:239-290`).

| Protocol | n images | Mean | Median | Minimum | Maximum | Sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Released raw-RGB analysis script | 4 | 98.892700% | 98.971160% | 98.578809% | 99.049671% | 0.214573 pp |
| Frozen DRIVE training preprocessing | 4 | 97.269658% | 97.238338% | 96.766746% | 97.835211% | 0.513367 pp |

Both limited DRIVE raw-image diagnostics exceed 95%. They do not reconstruct
the paper’s undocumented multi-dataset aggregation and cannot establish an
“across datasets” result.

### 5.5 Trained pre-SSTM feature diagnostics

For each selected Full checkpoint, an intermediate inference model exposed
the tensor immediately before every SSTM layer—after that stage’s MRF-SE
block. The four fixed validation images were evaluated without augmentation
and with `training=False`.

The table below applies the **paper-intended spatial H×W FFT and centered
K_eff×K_eff crop**, not the released W,C FFT/resize operation. The aggregate
contains 12 checkpoint-image observations per stage (3 checkpoints × 4
images). Because the same four images are repeated across checkpoints, these
12 observations are not independent and no confidence interval is claimed.

| Stage | K_eff | n checkpoint-image values | Mean | Median | Minimum | Maximum | Sample SD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 12 | 52.005674% | 51.579412% | 49.270499% | 56.231440% | 2.339273 pp |
| 2 | 32 | 12 | 69.157743% | 68.555644% | 63.857775% | 75.173468% | 3.322924 pp |
| 3 | 32 | 12 | 91.705598% | 91.436884% | 89.927323% | 93.500742% | 1.067344 pp |
| 4 | 22 | 12 | 100.000000% | 100.000000% | 100.000000% | 100.000000% | 0 |
| 5 | 11 | 12 | 100.000000% | 100.000000% | 100.000000% | 100.000000% | 0 |

Per-checkpoint means show the same pattern:

| Seed/checkpoint | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 |
|---|---:|---:|---:|---:|---:|
| 42 / epoch 21 | 53.377076% | 72.772464% | 92.885841% | 100% | 100% |
| 7 / epoch 14 | 51.349147% | 68.803374% | 91.445869% | 100% | 100% |
| 123 / epoch 24 | 51.290800% | 65.897391% | 90.785083% | 100% | 100% |

Stages 1–3 do not retain more than 95% under the intended spatial
interpretation on these trained DRIVE features. Stages 4–5 equal 100% only
because `actual_k` equals their complete H×W grids, so no spatial coefficient
would be removed.

As an axis-sensitivity check only, a hypothetical centered crop on the
released FFT axes W,C—an operation the code does **not** perform—gave aggregate
means of 66.981847%, 80.880262%, 48.817341%, 28.980997%, and 9.457528% at
stages 1–5. These numbers are not SSTM retained-energy results; they show that
changing the FFT axes materially changes the quantity.

### 5.6 Energy interpretation

The local evidence supports only this narrow statement:

> A paper-intended centered 32×32 spatial-frequency crop retains more than 95%
> on the four permitted DRIVE validation images under both tested raw-image
> preprocessing protocols.

It does not support a general or implementation-level statement because:

- the paper’s cross-dataset protocol is missing;
- the repository reports only 94.8% overall;
- the released SSTM has no retained subset;
- trained features at stages 1–3 are below 95%; and
- only four validation images from one modality are available here.

## 6. Computational-cost protocol and results

### 6.1 What “63%” denotes in the available sources

The paper does not consistently tie 63% to a named quantity:

- `sources/alphaxiv-2601.01285v1.txt:73-75` says truncation “reduces
  **frequency cost** by 63%.”
- `sources/alphaxiv-2601.01285v1.txt:86-90` abbreviates this to “63% cost
  reduction.”
- Section 3.3 compares asymptotic terms:

  ```text
  full self-attention: O((HW)^2 C)
  SSTM spectral:       O(HW C log(HW))
  SSTM spatial:        O(HW C^2)
  ```

  Source: `sources/alphaxiv-2601.01285v1.txt:149-159`.

- `sources/alphaxiv-2601.01285v1.txt:195-198` separately reports that SSTM is
  4.2× faster than self-attention with matched parameters.

The accessible paper does not give the attention tensor shape, stage, token
count, channel dimension, number of heads, head dimension, batch size,
precision, hardware, warm-up, timing protocol, FLOP/MAC convention, or
derivation behind 63%. It repeatedly refers to an Appendix, but the local
nine-page arXiv v1 artifact ends with references and contains no such
computational appendix.

### 6.2 Repository evidence

The released repository contains no profiler, FLOP counter, matched-attention
module, benchmark script, or saved profiler output for the 63% value.

`official_repo/s2mnet/models/baselines.py:99-181` has multi-head attention only
inside a distinct approximate TransUNet bottleneck. Its 256-channel bottleneck
and whole-model construction do not match the five SSTM stage shapes and
cannot be silently substituted as the claimed baseline.

The README supplies reported but non-executable values:

- SSTM spectral: 2.7 GFLOPs;
- SSTM spatial: 1.8 GFLOPs;
- whole S2M-Net: 11.2 GFLOPs;
- whole TransUNet: 45.2 GFLOPs and 42.3 ms;
- whole S2M-Net: 10.1 ms.

Sources: `official_repo/README.md:83-91,348-364`.

No formula or configuration connects these values to 63%. If the reported
4.5 GFLOP SSTM module total were exactly 63% below an attention module, that
attention module would need to cost 12.162162 GFLOPs; no such number appears
in the paper or repository.

### 6.3 Programmatic arithmetic checks

The required reduction formula is:

```text
reduction (%) = 100 × (cost_attention - cost_SSTM) / cost_attention
```

The following are audit checks, not reproductions of the undocumented
baseline:

| Available comparison or interpretation | Calculated reduction |
|---|---:|
| 32×32 coefficient count vs. full 352×352 spatial grid | 99.173554% |
| 32 coefficients vs. the paper’s unexplained 256 coefficients | 87.500000% |
| Paper-reported 4.2× inference speedup converted to reduction | 76.190476% |
| README whole-model FLOPs: 11.2G vs. TransUNet 45.2G | 75.221239% |
| README whole-model runtime: 10.1 ms vs. TransUNet 42.3 ms | 76.122931% |

None equals 63%, and none is a verified like-for-like SSTM-versus-attention
module comparison.

Using the paper’s leading asymptotic terms with unit coefficients gives the
following additional sensitivity check:

```text
attention proxy = N²C
SSTM proxy       = NC log2(N) + NC²
N                = H×W
```

| Stage | H×W | C | Unit-coefficient proxy reduction |
|---:|---:|---:|---:|
| 1 | 176×176 | 24 | 99.874358% |
| 2 | 88×88 | 32 | 99.419953% |
| 3 | 44×44 | 64 | 96.130224% |
| 4 | 22×22 | 80 | 81.628334% |
| 5 | 11×11 | 128 | -11.503193% |
| Aggregate of those proxy terms | — | — | 99.782291% |

These values demonstrate why big-O expressions cannot establish a fixed 63%:
constant factors, projection costs, FFT counting conventions, and stage
aggregation materially affect the result. They are not presented as FLOPs or
MACs.

### 6.4 Why no runtime or memory benchmark was run

No profiler measurement was performed because there is no sufficiently
specified or executable matched full-spatial-attention baseline. Constructing
one would require arbitrary choices about projections, heads, normalization,
fusion, placement, stage coverage, and parameter matching. It would not
reproduce the paper and could be tuned to produce almost any reduction.

The early SSTM feature grids contain 30,976 and 7,744 tokens. Naively
materializing full attention there would also allocate very large N×N score
matrices. Analytical counting is safer, but it becomes evidential only after
the missing baseline and counting convention are specified.

## 7. Paper-versus-code comparison

| Claim element | Paper | Released code/repository | Diagnostic finding |
|---|---|---|---|
| FFT type | Per-channel spatial 2D FFT | `fft2d` called on NHWC | Actual axes are W,C |
| K meaning | Centered K×K spatial-frequency crop | Square resize/weight-grid side | Configured 32; effective 32,32,32,22,11 |
| Shift | `fftshift` and `ifftshift` | Neither used | Direct discrepancy |
| Truncation | Crop and zero-pad | Bilinear down/up resize | No retained subset |
| Frequency weights | K×K×C learned spectral filter | Real k×k×C tensor | Not general complex filtering; coordinate mismatch |
| Selective branch | Content-gated d=16 bottleneck | C-to-C gate and projection | d=16 unused |
| Branch placement | SSTM described as dual branch | Stages 1–2 spectral-only | Direct discrepancy |
| Energy | >95% across datasets | README: 94.8% overall, 96.4% fundus | Four DRIVE raw images >95%; trained stages 1–3 <95% |
| Cost | 63%, metric ambiguous | No derivation or matched baseline | Not reproducible |
| Attention comparison | Matched parameters, 4.2× faster | No matched attention implementation | Cannot independently measure |

## 8. Discrepancies and limitations

1. **Wrong FFT dimensions in released code.** NHWC is not transposed, so the
   released 2D FFT operates over width and channels.
2. **No released truncation.** Bilinear complex-spectrum resizing is not the
   paper’s centered crop/zero-pad operation.
3. **No shift operations.** `fftshift` and `ifftshift` from equations 5 and 7
   are absent.
4. **K changes meaning.** The paper’s crop side becomes a resize and
   real-weight-grid side in code.
5. **K is clamped.** Effective K is 22 and 11 at stages 4 and 5; those stages
   do not reduce their H/W-shaped grids.
6. **The selective branch differs.** It has no state-space mechanism or d=16
   bottleneck, and stages 1–2 disable it.
7. **Energy target is ambiguous.** The paper does not resolve raw inputs versus
   trained features, stage, sample set, preprocessing, or aggregation.
8. **Paper and README disagree.** The paper says more than 95% across datasets;
   the README says 94.8% overall and more than 93% in its summary.
9. **The released utility is not the model operation.** It correctly measures
   a centered spatial crop on raw images, while the forward layer uses W,C FFT
   plus resize.
10. **Local energy evidence is narrow.** Only four non-hidden DRIVE validation
    images are available. The 12 feature observations repeat these images
    across three checkpoints and are not 12 independent cases.
11. **Cost metric and baseline are missing.** The paper does not define what
    its 63% measures or provide the baseline configuration and convention.
12. **No executable cost evidence exists.** Repository-wide searches found no
    FLOP/profiler/benchmark implementation supporting 63%.
13. **The cited appendix is unavailable.** The accessible nine-page paper
    artifact contains no appendix despite multiple references to one.
14. **No arbitrary substitute was used.** The bundled TransUNet approximation
    is not a matched SSTM replacement and was not benchmarked.

## 9. Conservative component verdicts

### 9.1 Truncated 2D FFT and K=32

**Verdict: Partially verified as a paper-intended specification; contradicted
as a description of the released forward implementation.**

The canonical configuration directly verifies a requested K value of 32 and
the layer calls a two-dimensional FFT. The paper clearly defines K as the side
length of a centered 32×32 spatial-frequency crop per channel. However, the
released code transforms W,C rather than H,W and performs bilinear resizing
rather than centered truncation. Therefore the released SSTM does not
implement the claimed truncated spatial 2D FFT.

### 9.2 More than 95% retained spectral energy

**Verdict: Not verified for the released SSTM; supported only by a limited
four-image raw-DRIVE diagnostic under the paper-intended spatial
interpretation.**

Both local raw-image protocols exceed 95%, but the paper’s cross-dataset
protocol cannot be reconstructed, its repository reports 94.8% overall, and
the released forward operation has no retained subset for which the standard
ratio is defined. On trained pre-SSTM DRIVE features, stages 1–3 retain about
52.01%, 69.16%, and 91.71%, not more than 95%; stages 4–5 are 100% only because
their complete spatial grids are used.

### 9.3 63% computational-cost reduction

**Verdict: Not verified and not reproducible from the available paper or
released repository.**

The metric, matched attention baseline, tensor configuration, counting
convention, and executable evidence are missing. Available coefficient,
whole-model, runtime, and asymptotic calculations do not recover 63%.
Constructing a new attention module would be arbitrary and was deliberately
not done.

### 9.4 Overall Claim 2

**Verdict: Not verified.**

The executable audit reports `Audit PASS`, meaning that the deterministic
verification procedure and its invariants pass. It does not mean the
scientific claim passes. The component evidence above contradicts the released
implementation-level mechanism and does not reproduce either the general
energy threshold or the 63% cost reduction.
