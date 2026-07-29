# Claim 3 — EndoVis17 matched-comparison protocol

> **PROTOCOL_STATUS=FINAL_PREDECLARED**
>
> This document freezes the protocol before implementation or training. It
> specifies an independent matched comparison; it is not a reconstruction of
> the unrecoverable experiment that reported S2M-Net 83.77 and U-Mamba 65.92.
> No Claim 3 model has been trained or evaluated.

## 1. Scientific question and scope

The experiment asks:

> Under one identical reconstructed EndoVis17 seven-instrument-type protocol,
> does S2M-Net outperform U-Mamba?

The comparison consists of six from-scratch runs: two architectures paired at
seeds 42, 7, and 123. Data, target construction, sample order and augmentation
stream within each seed, optimization budget, checkpoint criterion, and
evaluation are held fixed. The architecture is the intended independent
variable.

The original 83.77-versus-65.92 numerical claim remains not reproducible from
the released artifacts because its task details, split, evaluator, checkpoints,
and matched baseline configuration are not recoverable. This protocol does not
directly test or describe itself as a reproduction of either number.

## 2. Evidence and narrow implementation inspection

The released S2M-Net source is pinned at
`3ec59668ab9b438ab9b170306d29b01e9270fd5a`.

- `official_repo/configs/surgical.yaml:3-25` selects 512 × 512, six classes,
  batch size 2, 30 epochs, and the `core` and `boundary` morphology-aware loss
  components. These settings do not describe the eight-class task fixed here.
- `official_repo/s2mnet/models/s2mnet.py:27-52,84,153-164` constructs a
  channels-last RGB model and already parameterizes the final 1 × 1 head by
  `num_classes`. Only that head and its activation need adaptation for eight
  logits.
- `official_repo/s2mnet/dataloaders/full_image.py:34-83,102-150` expects one
  combined mask per image, cannot preserve an ignore index through one-hot
  conversion, and does not combine per-instrument source masks.
- `official_repo/train.py:119-150` invokes the generic full-image pipeline, not
  `get_surgical_augmentation`; the surgical augmentation merely exists at
  `official_repo/s2mnet/dataloaders/augmentations.py:83-99`.
- The released morphology-aware loss and metrics flatten all channels and have
  no ignore-mask contract (`official_repo/s2mnet/losses/mal.py:174-206`;
  `official_repo/s2mnet/utils/metrics.py:14-28`). They are not used as evidence
  of correct multiclass training or evaluation.
- `official_repo/test.py:193-223` selects output channel 0 and applies a binary
  threshold, so the released evaluator is not used.

The genuine U-Mamba implementation is the official
[`bowang-lab/U-Mamba`](https://github.com/bowang-lab/U-Mamba) source pinned for
later acquisition at commit
[`28459e33ca03769800dd35e23c6e62491d1925b5`](https://github.com/bowang-lab/U-Mamba/commit/28459e33ca03769800dd35e23c6e62491d1925b5).
The selected variant is **U-Mamba_Bot**, the 2-D encoder-decoder in which the
Mamba block is confined to the bottleneck. The primary
[U-Mamba paper](https://arxiv.org/abs/2401.04722) specifies channels-first
features, a softmax segmentation head, an unweighted Dice-plus-cross-entropy
loss, disabled test-time augmentation, and a seven-stage/six-pooling
endoscopy configuration. The official repository exposes this variant through
`nnUNetTrainerUMambaBot`.

The similarly named TensorFlow model at
`official_repo/s2mnet/models/baselines.py:188-254` is explicitly a lightweight
“SSM-like” gated dense surrogate. It must not be reported or trained as the
original U-Mamba.

### Adaptation classification

| Change | Classification | Scientific effect |
| --- | --- | --- |
| S2M-Net final 1 × 1 convolution: 8 linear logits | Final-head change | Changes only target cardinality and removes in-head softmax |
| U-Mamba final segmentation heads: 8 logits | Final-head change | Changes only target cardinality |
| Shared source-mask combiner, ignore mask, sampler, augmenter, loss wrapper, and evaluator | Framework plumbing | Makes the same data and objective available to both frameworks |
| NHWC input for TensorFlow and NCHW transpose for PyTorch | Framework plumbing | No change to image values |
| Disable U-Mamba deep supervision | Training-plumbing exception | Removes auxiliary losses so both models optimize only the common full-resolution objective; the inference path is unchanged |
| U-Mamba_Bot Mamba block and S2M-Net MRF-SE/SSTM/BFP blocks | Unchanged architecture | These are the model differences being compared |

## 3. Dataset contract

### Source, eligible frames, and split

- Source: official Hugging Face EndoVis17 backup, revision
  `518d8a542b83b6af8cf2c37e4aa210b218655248`.
- Read-only local source: `/home/sarah/Datasets/EndoVis17_HF`.
- Eligible sequences: `instrument_dataset_1` through
  `instrument_dataset_8`.
- Eligible frame IDs in every sequence: exactly `frame000` through
  `frame224`, inclusive. Frames `225–299` are ineligible for every operation.
- Eligible total: 1,800 frames.
- Train: sequences 2, 3, 7, 8 (900 frames).
- Validation: sequence 4 (225 frames).
- Held-out local test: sequences 1, 5, 6 (675 frames).

No frame from a sequence may cross partitions. Test images, masks, summaries,
and intermediate test metrics must not be read by training or checkpoint
selection.

### Pairing and target construction

For each eligible ID, pair
`instrument_dataset_N/left_frames/frameNNN.png` with every same-stem PNG under
that sequence's `ground_truth/<instrument-directory>/`. A missing image, a
missing expected per-instrument mask, a size mismatch, an unknown directory, or
an extra eligible-stem ambiguity is a dataset-contract failure; it is never
silently skipped.

The native integer target uses ignore index **255** and this fixed mapping:

| Target value | Class | Accepted instrument-directory identity |
| ---: | --- | --- |
| 0 | Background | no valid foreground mask |
| 1 | Bipolar Forceps | names containing `Bipolar_Forceps`, including `Maryland_Bipolar_Forceps` |
| 2 | Prograsp Forceps | names containing `Prograsp_Forceps` |
| 3 | Large Needle Driver | names containing `Large_Needle_Driver` |
| 4 | Vessel Sealer | names containing `Vessel_Sealer` |
| 5 | Grasping Retractor | names containing `Grasping_Retractor` |
| 6 | Monopolar Curved Scissors | names containing `Monopolar_Curved_Scissors` |
| 7 | Other | `Other_labels` |

`Left_`, `Right_`, `_Left`, `_Right`, and `_labels` identify instances or sides
and do not define additional semantic classes. Multiple masks of the same type
are unioned. The seventh foreground class is **Other**, never “Ultrasound
Probe.”

Each per-instrument mask is converted to a foreground-presence mask only at
pixels whose decoded part value is one of `{10, 20, 30, 40}` from
`training/mappings.json`; 0 is background. Instrument type comes from the
verified directory-to-type mapping above, not from part value or appearance.
Pixels assigned to two different instrument types are set to 255. Same-type
instance overlap remains that semantic type.

### Deterministic sequence-7 rule

The same decoder is applied to all sequences, which makes the sequence-7
exception explicit:

1. An `L` mask is interpreted as its uint8 value.
2. An `RGB` or `RGBA` mask is valid at a pixel only if `R = G = B`; alpha, when
   present, must be 255. The shared grayscale value is then interpreted as the
   part value.
3. Any non-grayscale pixel, non-opaque pixel, or value outside
   `{0, 10, 20, 30, 40}` is set to ignore, not rounded, thresholded, or
   reassigned.
4. Ignore status has precedence over background or class assignment in the
   combined target.

This rule preserves the canonical labels while deterministically excluding the
sparse non-canonical sequence-7 pixels found by the completed integrity audit.

Before training, the implementation must produce and freeze a manifest with
source revision, relative paths, sequence and frame IDs, source SHA-256 hashes,
combined-target SHA-256 hashes, class counts, conflict counts, and ignored-pixel
counts. Manifest generation does not authorize modification of the source data.

## 4. Common input and augmentation pipeline

These are independent reconstruction choices and remain fixed after any result
is observed.

### Spatial and intensity processing

- Input: RGB float32 in `[0,1]`.
- Spatial size: **384 × 384** for both models.
- Image resize: OpenCV `INTER_LINEAR_EXACT`.
- Integer target resize: OpenCV `INTER_NEAREST_EXACT`.
- Aspect ratio: direct resize to 384 × 384; no crop or letterbox.
- Normalization: compute one RGB mean and standard deviation in float64 from
  all raw pixels of the 900 eligible training images only. Record the three
  means and standard deviations in the frozen manifest. Apply
  `(x - mean) / std` channel-wise after augmentation. Validation and test use
  those same training statistics.

384 × 384 is the smallest square independent choice here that retains the
official U-Mamba endoscopy height and is divisible by its six 2-D pooling
operations as well as S2M-Net's five downsamplings. It replaces both the
paper's unrecoverable 352 choice and the released surgical YAML's 512 choice;
it cannot be changed in response to memory use or results.

### Training-only augmentation

The following operations are sampled independently, in this order, from a
framework-independent deterministic generator. Validation and test receive
resize and normalization only.

1. Horizontal flip with probability 0.5.
2. In-plane rotation with probability 0.7; angle uniformly sampled from
   `[-15°, +15°]`. OpenCV bilinear interpolation is used for the image and
   nearest-neighbour interpolation for the target. Image borders use
   `BORDER_REFLECT_101`; target borders are 255.
3. Brightness/contrast with probability 0.8. Sample
   `c ~ Uniform(0.8, 1.2)` and `b ~ Uniform(-0.2, 0.2)`, then apply
   `clip((x - 0.5) * c + 0.5 + b, 0, 1)`.
4. Gaussian blur with probability 0.2, a 5 × 5 kernel, and
   `sigma ~ Uniform(0.1, 1.5)`.
5. Additive Gaussian noise with probability 0.3 and
   `sigma ~ Uniform(0, 0.05)` in `[0,1]` units, followed by clipping.

There is no vertical flip, random crop, scale/translation, elastic transform,
class-balanced sampling, mixup, cutmix, or test-time augmentation.

For each seed, a documented counter-based generator keyed by
`(seed, training-pass, shuffled-position)` must produce the same sample order
and augmentation parameters for both architectures. The 900 training samples
are shuffled once per pass; all are used exactly once per pass.

## 5. Model contract

### S2M-Net

- Source: the pinned official submodule.
- Input interface: `(B, 384, 384, 3)`.
- Encoder/decoder settings remain
  `filters=[24,32,64,80,128]`, MRF-SE enabled with kernels `[3,5,7]`,
  `se_reduction=16`, `expand_ratio=6`, SSTM enabled at all five stages with
  `k=32`, SSM state dimension 16, spectral stages
  `[true,true,true,true,true]`, SSM stages
  `[false,false,true,true,true]`, and BFP soft routing.
- Use ELU, encoder/global dropout 0.1, and SSTM dropout 0.1.
- Set the internal Keras kernel regularizer coefficient to 0 so that only the
  common optimizer applies weight decay; this changes training regularization,
  not the inference graph.
- Replace only the last 1 × 1 output layer with eight float32 **linear logits**.
  Softmax belongs to the shared loss/evaluator.

### U-Mamba

- Source: official U-Mamba commit
  `28459e33ca03769800dd35e23c6e62491d1925b5`.
- Variant: 2-D **U-Mamba_Bot**, not U-Mamba_Enc and not the bundled TensorFlow
  surrogate.
- Input interface: `(B, 3, 384, 384)`, obtained only by transposing the common
  normalized tensor.
- Freeze a seven-stage/six-pooling plan before training:
  features `[32,64,128,256,320,320,320]`, 3 × 3 convolutions, first-stage
  stride `(1,1)`, then six `(2,2)` strides, InstanceNorm, LeakyReLU, residual
  encoder/decoder blocks, skip connections, and the official bottleneck Mamba
  block.
- Pass encoder block counts `[2,2,2,2,2,2,2]` and decoder block counts
  `[2,2,2,2,2,2]` to the pinned constructor. Its declared depth reduction
  produces effective encoder counts `[2,2,2,2,1,1,1]` and decoder counts
  `[2,2,2,2,1,1]`.
- Use convolution bias, InstanceNorm `eps=1e-5` with affine parameters,
  LeakyReLU slope 0.01, no dropout, and the pinned He initialization with
  scale `1e-2`. The bottleneck Mamba layer remains `d_state=16`, `d_conv=4`,
  and `expand=2`.
- Set `num_segmentation_heads=8`.
- Disable deep supervision and auxiliary heads/losses. Train and infer from the
  single full-resolution output only.
- Return eight linear logits. Softmax belongs to the shared loss/evaluator.

The explicit feature/stage plan is an independent freeze of the official
U-Mamba/nnU-Net design at the declared 384 × 384 input; it prevents later
self-configuration from becoming a model-specific, result-dependent choice.

The official U-Mamba source is not currently vendored or installed. Acquiring
that exact commit and verifying that the frozen seven-stage constructor matches
this contract is an operational prerequisite for implementation, not an open
scientific choice. Any constructor incompatibility must stop implementation and
amend this protocol before results exist.

## 6. Common objective and optimization

### Loss

Let `V` be pixels whose target is not 255. The common loss is

\[
\mathcal L = \mathcal L_{\mathrm{CE}} + \mathcal L_{\mathrm{Dice}}.
\]

`CE` is the unweighted mean sparse softmax cross-entropy over `V` and all eight
classes. For foreground class \(c \in \{1,\ldots,7\}\), using probabilities
\(p_{ic}\), one-hot targets \(g_{ic}\), and \(\epsilon=10^{-6}\),

\[
D_c^{soft} =
\frac{2\sum_{i\in V}p_{ic}g_{ic}+\epsilon}
{\sum_{i\in V}p_{ic}+\sum_{i\in V}g_{ic}+\epsilon}.
\]

The sums are over the complete effective batch. `Dice` loss is one minus the
equal-weight mean of \(D_c^{soft}\) over foreground classes present in the
effective-batch ground truth. If no foreground class is present, its Dice term
is zero and CE remains active. Background is excluded from the Dice term.
There are no class weights and no MASL/MAL components.

### Optimizer and budget

| Field | Frozen value |
| --- | --- |
| Optimizer | AdamW |
| Betas / epsilon | `(0.9, 0.999)` / `1e-8` |
| Initial/peak learning rate | `1e-4` |
| Weight decay | `1e-4`, decoupled; exclude biases and normalization scale/offset |
| Scheduler | 500-step linear warm-up from 0 to `1e-4`, then cosine decay to `1e-6` |
| Effective batch | 4 images |
| Maximum optimizer steps | **10,125** |
| Processed training samples | **40,500** (45 complete passes × 900) |
| Validation frequency | every 225 optimizer steps, including step 10,125 |
| Early stopping | disabled |
| Gradient clipping | global L2 norm 1.0, after gradient accumulation |
| Mixed precision | disabled; parameters, forward pass, loss, gradients, and optimizer state are float32 |

For optimizer update \(t\), numbered from 1, the learning rate is
`1e-4 * t / 500` for \(1 \le t \le 500\). For
\(501 \le t \le 10125\), let
\(q=(t-500)/(10125-500)\); the learning rate is
`1e-6 + 0.5 * (1e-4 - 1e-6) * (1 + cos(pi*q))`.

Each optimizer update is computed from exactly four sequential augmented
samples. Gradient accumulation averages, rather than sums, their losses and
gradients. Framework-native AdamW implementations are allowed, but their
versions and settings must be recorded; this TensorFlow/PyTorch implementation
difference is an unavoidable framework exception.

## 7. RTX 4060 8 GB policy

The target device is one RTX 4060 with 8 GB VRAM.

1. Start both models at micro-batch 2 with two accumulation steps.
2. On verified OOM, reduce the affected model to micro-batch 1 and increase
   accumulation to four, preserving effective batch 4, sample order, 10,125
   optimizer updates, and scheduler position.
3. If micro-batch 1 still OOMs, enable activation/gradient checkpointing only
   if the pinned implementation already supports it without changing layer
   mathematics. Apply that setting to all three seeds of the affected model
   and disclose it.
4. If these measures fail, that model is infeasible on the declared hardware
   and the campaign stops incomplete.

If a fallback becomes necessary after an earlier seed completed under a
different micro-batch/checkpointing setting, that earlier run is retained but
invalidated and rerun so all three seeds for the architecture share one
hardware setting.

Resolution, task, classes, loss, sample count, and optimizer-step budget may
never be reduced silently. CPU offload, architecture pruning, fewer stages,
and model substitution are not permitted fallbacks.

## 8. Seeds, run matrix, and checkpoint selection

### Paired run matrix

| Pair | S2M-Net | U-Mamba_Bot |
| ---: | --- | --- |
| 1 | seed 42 | seed 42 |
| 2 | seed 7 | seed 7 |
| 3 | seed 123 | seed 123 |

Each seed controls framework initialization, sample order, and the shared
augmentation generator. Runs are from scratch; no pretrained weights are used.
The six runs may be scheduled in any resource-safe order, but results are
analyzed only as the three declared seed pairs.

### Selection rule

At every 225-step boundary, evaluate all 225 validation frames without
augmentation or TTA. Compute one validation loss over the entire validation
set using the same CE-plus-Dice definition: CE is pooled over all valid pixels,
and Dice is pooled over the validation set. Sequence 4 contains only Prograsp
Forceps and Large Needle Driver, so only those two foreground classes enter the
validation Dice term; CE still evaluates all eight logits and penalizes
incorrect class predictions.

Select the checkpoint with the **lowest pooled validation loss**. A strict
decrease is required; exact ties retain the earlier checkpoint. There is no
early stopping. Preserve the selected checkpoint and the final-step recovery
checkpoint separately. Held-out test data is never used for selection,
fallback choice, rerun choice, or debugging.

This criterion supports identical model selection but does not make sequence 4
representative of performance across the full taxonomy.

## 9. Held-out evaluation and primary metric

Use the selected checkpoint once on the 675 eligible test frames, in ascending
sequence/frame order, with no TTA or post-processing.

1. Apply softmax at 384 × 384.
2. Resize each of the eight probability channels to the native annotation
   dimensions with the same shared OpenCV bilinear evaluator.
3. Take `argmax` with lowest-index tie breaking.
4. Exclude native target pixels equal to 255.

For each foreground class \(c=1,\ldots,7\), pool TP, FP, and FN over **all
valid pixels of all 675 test frames**, then compute

\[
D_c = \frac{2TP_c}{2TP_c+FP_c+FN_c}.
\]

The primary score is the equal-weight arithmetic mean of the seven foreground
class Dice values. Background is excluded. No epsilon changes a non-empty hard
metric.

Absent-class rule:

- if a class is absent from ground truth and prediction, report `NA` and
  exclude it from the macro mean;
- if it is absent from ground truth but predicted, its Dice is 0 and it remains
  in the macro mean;
- if it is present in ground truth but not predicted, its Dice is 0.

The verified held-out split contains every foreground class, but the rule is
predeclared for integrity. Both models are evaluated by the same evaluator on
the identical ordered frame manifest.

## 10. Secondary reporting

Secondary outputs may be computed only from the same selected checkpoints and
predictions:

- per-class Dice;
- foreground macro IoU, using the same global pooling and absent-class rule;
- per-sequence/per-class TP, FP, FN, Dice, and IoU as diagnostics;
- trainable and total parameter counts;
- peak allocated and peak reserved GPU memory;
- training wall time, optimizer steps per second, and single-image inference
  time after 20 warm-up images, summarized over the remaining ordered test
  images.

Runtime is measured with device synchronization on the same RTX 4060 and is
descriptive, not a second training protocol. NSD, ensembles, TTA, threshold
tuning, post-processing, and alternate resolutions are outside scope.

## 11. Matched-comparison safeguards

- One immutable eligible-frame manifest and target-construction implementation.
- One split, class mapping, ignore rule, native evaluation grid, and evaluator.
- One per-seed sample order and augmentation-parameter stream shared by models.
- One loss, effective batch, processed-sample count, optimizer-step budget,
  scheduler, validation cadence, checkpoint rule, and seed set.
- One held-out evaluation with predictions generated for every declared frame.
- Configurations are frozen and hashed before the first run.
- Test results remain sealed until all six runs have reached a terminal state.

Unavoidable exceptions are framework tensor order, native TensorFlow versus
PyTorch optimizer kernels, model-specific micro-batch after a documented OOM,
and already-supported checkpointing after the prescribed fallback. These do
not change effective batch or scientific inputs. U-Mamba deep supervision is
disabled to prevent a model-specific auxiliary objective.

## 12. Predeclared analysis and interpretation

For seed \(s\), define the paired primary-score difference in percentage
points:

\[
\Delta_s = 100(Dice_{\mathrm{S2M},s} - Dice_{\mathrm{U\text{-}Mamba},s}).
\]

Report all three \(\Delta_s\), their arithmetic mean, sample standard deviation
(`ddof=1`), and the two models' per-seed scores. A **1.0 percentage-point**
practical-equivalence tolerance is an independent predeclared choice:

1. **Practical comparability:** every \(|\Delta_s| \le 1.0\).
2. **Consistent S2M-Net advantage:** every \(\Delta_s > 1.0\).
3. **U-Mamba advantage:** every \(\Delta_s < -1.0\).
4. **Smaller or inconsistent advantage:** every other complete three-pair
   pattern; report the mean direction and sign pattern without a superiority
   claim.
5. **Insufficient or incomplete evidence:** fewer than all three valid pairs,
   a protocol violation, or a model that cannot complete on the declared
   hardware.

The categories are applied in the order above. The later report may compare the
descriptive mean difference with 17.85 points for context, but cannot treat
agreement or disagreement as reproduction of the original numerical claim.
Three seeds support descriptive paired evidence, not an overstated claim of
formal statistical significance.

## 13. Failure, exclusion, and rerun rules

- **OOM:** apply only the fallback sequence in Section 7. Preserve the failed
  log. A successful fallback is used for all seeds of that model.
- **Interrupted run:** resume from the latest complete recovery checkpoint,
  restoring model, optimizer, scheduler, gradient-accumulation, RNG, sampler,
  and augmentation-counter state. Do not restart selectively based on metrics.
- **NaN/Inf:** stop immediately and preserve the last finite state. Retry once
  from that state with the identical configuration and RNG state. Recurrence
  marks the seed failed; hyperparameters may not be tuned.
- **Corrupt sample, missing mask, unknown directory, or pairing failure:** abort
  the campaign before training or at first detection. Never skip or replace a
  sample.
- **Failed seed:** do not substitute another seed. Report the pair incomplete;
  do not compute a three-pair conclusion.
- **Rerun:** permitted only for documented infrastructure failure or verified
  implementation defect. It must use the same seed and frozen protocol. Keep
  both attempts, identify the superseded run, explain the defect, and never
  choose between reruns by validation or test performance.
- **Protocol/code defect found after test unsealing:** invalidate affected
  results and require a documented protocol amendment before any new campaign;
  do not patch results in place.

## 14. Required implementation artifacts

Implementation must preserve, per campaign or run as appropriate:

- this protocol and its Git hash;
- exact S2M-Net and U-Mamba source revisions plus clean/dirty state;
- environment lock files, Python/framework/CUDA/cuDNN/driver versions, GPU
  identity, and deterministic-runtime flags;
- immutable eligible-frame manifest, source and combined-target hashes,
  mapping, split, training normalization statistics, anomaly/conflict counts,
  and ignore counts;
- one frozen human-readable configuration and canonical configuration hash per
  model/seed;
- run manifest containing architecture constructor arguments, parameter counts,
  seed, effective/micro-batch, accumulation, fallback state, and start/end
  timestamps;
- exact per-step learning rate and per-validation training/validation metrics;
- full logs, terminal status, failure records, and resume lineage;
- selected and final recovery checkpoints with hashes;
- native-grid uint8 prediction PNGs (values 0–7), or equivalently complete
  lossless per-frame TP/FP/FN inputs plus prediction hashes; preserving the
  prediction PNGs is preferred;
- per-frame/per-sequence/per-class metric inputs, aggregate JSON/CSV tables, and
  the paired three-seed summary;
- peak-memory and runtime measurement logs.

No implementation, preprocessing, configuration generation, or run is
authorized by this document alone. The immediate next stage is a narrow
implementation against this frozen contract.
