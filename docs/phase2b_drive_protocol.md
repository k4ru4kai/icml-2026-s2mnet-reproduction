# Phase 2B — Proposed DRIVE Protocol and Acquisition Specification

> **PROTOCOL_STATUS=PROVISIONAL**
>
> DRIVE has not been downloaded, accessed, trained on, or evaluated. No
> empirical reproduction result or performance conclusion exists. This
> document records evidence and proposed decisions for review before any data
> acquisition or experiment is authorized.

## Scope and evidence policy

The target claim is the DRIVE Full Model versus No-SSTM ablation. This
specification keeps five categories separate:

1. behavior confirmed from the released implementation;
2. protocol statements in arXiv v1 of the paper;
3. protocol statements found only in the current official README;
4. provisional reproduction decisions; and
5. unresolved questions.

The released implementation is pinned at
`3ec59668ab9b438ab9b170306d29b01e9270fd5a`. Paper references below point to
`sources/alphaxiv-2601.01285v1.pdf`; PDF page numbers are used because the PDF
has no source line numbering. Source-code and configuration references use
exact file and line locations. A statement from one evidence category must not
be silently promoted into another category.

## 1. Behavior confirmed from the released implementation

### Configuration and data discovery

- `official_repo/configs/retinal.yaml:1-7` is labeled for CHASE-DB / DRIVE, but
  its default directories are `data/CHASE-DB/train`, `data/CHASE-DB/val`, and
  `data/CHASE-DB/test`.
- The comment in `official_repo/configs/retinal.yaml:2` says that the file
  inherits `default.yaml`, but `official_repo/train.py:75-98` and
  `official_repo/test.py:52-54` only parse the selected YAML file. No YAML
  inheritance or merge is implemented; missing values are supplied only where
  executable code uses local fallbacks.
- `official_repo/s2mnet/dataloaders/base.py:18-58` discovers lowercase PNG,
  JPEG, TIFF, and BMP images, then searches for a mask with the same stem or a
  limited `_mask` suffix. GIF masks, separate observer directories, and FOV
  masks are not discovered.
- Every split must contain `images/` and `masks/` directories
  (`official_repo/s2mnet/dataloaders/base.py:61-77`). The loader does not enforce
  DRIVE IDs, a 20/20 split, an annotation observer, or split disjointness.

### Released retinal training path

- `official_repo/configs/retinal.yaml:10-26` selects 256×256 patch mode, stride
  32, minimum foreground ratio 0.005, 4,000 patches per epoch, green-channel
  CLAHE, and `use_fov_mask: true`.
- Full-resolution images are converted from BGR to RGB; CLAHE extracts the green
  channel and replicates the enhanced result into three float32 channels in
  `[0,1]` (`official_repo/s2mnet/dataloaders/patch_dataset.py:103-125` and
  `official_repo/s2mnet/utils/preprocessing.py:12-43`). Vessel masks are read in
  grayscale and binarized with `>127`.
- Patches are extracted on a fixed sliding grid, and only patches satisfying the
  foreground-ratio threshold enter an in-memory pool
  (`official_repo/s2mnet/dataloaders/patch_dataset.py:109-135`).
- Each batch is sampled randomly from that pool. The requested sequence index is
  ignored, and `on_epoch_end` does nothing
  (`official_repo/s2mnet/dataloaders/patch_dataset.py:138-159`). The same
  stochastic sampling is used for validation, although augmentation is disabled
  outside the training split (`official_repo/train.py:119-141`).
- Patch augmentation comprises horizontal and vertical flips, ±180° rotation,
  elastic transformation, grid distortion, brightness/contrast changes,
  Gaussian noise, and Gaussian blur with the probabilities and parameters in
  `official_repo/s2mnet/dataloaders/augmentations.py:24-46`.
- `PatchDataset` resets global NumPy state to its own default seed 42, while
  `build_dataloader` does not pass `training.seed`
  (`official_repo/s2mnet/dataloaders/patch_dataset.py:58-87` and
  `official_repo/train.py:119-141`). Albumentations receives no explicit seed or
  replay schedule.
- With 4,000 patches and batch size 16, the released loader reports 250 steps per
  epoch using integer division
  (`official_repo/s2mnet/dataloaders/patch_dataset.py:95-100`).

### Released FOV and evaluation behavior

- `use_fov_mask` does not load a supplied DRIVE FOV mask. It creates a centered
  circle with radius `min(height,width)//2 - margin`, then zeros the image and
  vessel mask outside that circle
  (`official_repo/s2mnet/utils/preprocessing.py:46-75`).
- `official_repo/test.py:61-100` resizes the complete image to the configured
  input size before patch inference. With the retinal values
  `input_size=patch_size=256`, evaluation normally consists of one 256×256
  patch, not native-resolution tiled inference.
- `official_repo/test.py:205-223` resizes and thresholds the vessel mask, applies
  the synthetic circle to the image and ground truth, and then computes metrics
  without masking the prediction or excluding invalid pixels.
- Test metrics are per-image hard Dice, IoU, precision, recall, and F1 over the
  entire resized square (`official_repo/s2mnet/utils/metrics.py:63-97`). Accuracy,
  specificity, AUC, and FOV-restricted metrics are not implemented.
- `evaluation.use_tta` and `evaluation.tta_n` in
  `official_repo/configs/default.yaml:70-73` are not consumed. The `--tta` flag
  controls a hard-coded eight-transform implementation
  (`official_repo/test.py:107-151` and `official_repo/test.py:314-326`).

### Released optimization and ablation definition

- The retinal YAML specifies Adam-compatible training values of batch size 16,
  100 epochs, and learning rate `5e-4`
  (`official_repo/configs/retinal.yaml:13-16`). `train.py` explicitly constructs
  Adam with gradient clipping at norm 1.0
  (`official_repo/train.py:210-222`).
- Missing retinal-YAML values fall back in code to cosine warmup, minimum
  learning rate `1e-6`, validation-Dice checkpointing, and early-stopping
  patience 40 (`official_repo/train.py:101-116` and
  `official_repo/train.py:227-252`).
- The default released loss contains `core`, `boundary`, `structure`, `scale`, and
  `texture`, with learned weights and morphology modulation
  (`official_repo/configs/default.yaml:34-44` and
  `official_repo/train.py:178-186`).
- Official ablation ID 15 disables SSTM at all five stages while retaining
  MRF-SE, BFP with soft routing, and the same five-component loss
  (`official_repo/experiments/ablation_configs.py:134-148`).

## 2. Protocol statements found in the paper

The following statements are attributed only to arXiv v1 unless independently
confirmed elsewhere.

### General experimental setup, Sections 4.1 and 4.2

On PDF page 5, Section 4.1 states that:

- DRIVE contains 40 images;
- results are reported as mean ± standard deviation over five runs;
- all images are resized to 352×352 using Lanczos interpolation; and
- augmentation includes 360° rotation, horizontal and vertical flips with
  `p=0.5`, elastic deformation with `alpha=1`, `sigma=50`, `p=0.3`,
  ShiftScaleRotate with `p=0.7`, brightness/contrast jitter of ±0.4 with
  `p=0.8`, Gaussian noise with `p=0.5`, and coarse dropout with `p=0.3`.

On PDF pages 5-6, Section 4.2 states a general regime of RMSprop at `1e-4`,
gradient clipping at norm 1.0, 45 epochs, batch size 4, 40 random augmentations
per training sample per epoch, and early stopping on validation Dice with
patience 15. The 1,612-image and 403-batch calculation in that paragraph refers
to the combined polyp training set, not explicitly to DRIVE.

### Target ablation setup, Section 5.3

On PDF page 6, Section 5.3 explicitly includes DRIVE in the ablation study and
states that all ablation configurations use identical hyperparameters:

- optimizer: Adam;
- learning rate: `1e-4`;
- batch size: 8;
- epochs: 100; and
- virtual epoch expansion: 30×.

Section 5.3 does not specify the DRIVE train/validation IDs, annotation observer,
FOV handling, image size, interpolation for labels, exact augmentation pipeline,
checkpoint-selection rule, threshold, TTA policy, loss configuration for the
Full Model versus No-SSTM row, or seed values. The general statements in
Sections 4.1 and 4.2 are evidence for some of these fields, but they conflict
with Section 5.3 on optimizer, batch size, epoch count, and expansion.

## 3. Protocol statements found only in the current README

The current README has a general training table at
`official_repo/README.md:667-683`. Relative to the paper and active retinal
configuration, it is the only source for these protocol statements:

- batch size 8 **per GPU**;
- 150 training epochs;
- early-stopping patience 50;
- the explicit seed set `{42, 123, 456, 789, 2024}`.

The same table also repeats Adam at `1e-4`, cosine annealing with ten-epoch
warmup, minimum learning rate `1e-6`, clipping at 1.0, and `1e-4` weight decay.
Some of those values overlap the default YAML or executable fallbacks, but the
table as a claimed unified protocol is not Section 5.3 and is not the active
retinal YAML. The README's retinal `PatchDataset` example at
`official_repo/README.md:613-627` is cross-confirmed by released code, so that
behavior is recorded in Section 1 rather than treated as README-only evidence.

README-only statements do not override the paper, retinal YAML, or executable
behavior. In particular, 150 epochs and patience 50 conflict with both Section
5.3 and the released retinal configuration.

## 4. Provisional reproduction decisions

All choices in this section are proposed for review. They are not claims about
what the authors used.

### Dataset split, annotations, and custody

| Provisional decision | Evidence and conflict | Rationale | Status |
|---|---|---|---|
| Preserve official test IDs 01-20 untouched; assign 21-36 to training and 37-40 to fixed validation; never derive validation data from the official test set. | The paper states only that DRIVE has 40 images and does not give split IDs. The README is also silent. Released code requires separate directories but enforces no count, ID, or disjointness rule. | Preserve the official test boundary, prevent test leakage, and make checkpoint selection reproducible. | **Provisional** |
| Use the first manual vessel annotation for every image. | Paper, README, YAML, and executable code do not identify an observer; the generic loader accepts one mask per image. | A single declared observer avoids mixing annotation policies across splits or variants. | **Provisional** |
| Retain and integrity-check the supplied official FOV mask for every image. | Paper and README do not define FOV handling. Released code substitutes a centered circle and does not load supplied FOV files. | Preserve the dataset's validity domain and avoid an undocumented geometric approximation. | **Provisional** |
| Use a reproduction-owned adapter and manifest; do not modify `official_repo` or silently reshape raw files for its generic loader. | Released discovery supports only `images/` and `masks/`, same-stem pairing, limited formats, no observer/FOV paths, and no split-integrity checks. | Keep upstream provenance intact while making every mapping and transformation auditable. | **Provisional** |
| Use fixed complete validation images rather than the released stochastic validation-patch mechanism. | Released validation disables augmentation but still samples random positive-filtered patches. Paper and README do not define DRIVE validation sampling. | Stabilize checkpoint selection and evaluate the same anatomical content for both variants. | **Provisional** |
| Evaluate complete images with deterministic, asserted full coverage. | Released retinal test preprocessing reduces the full image to one 256×256 patch; generic tiling can omit uncovered borders. Paper says 352×352 resizing but does not define coverage. | Prevent silent unevaluated pixels and keep Full Model and No-SSTM on an identical domain. | **Provisional** |
| Compute per-image hard Dice only inside the supplied FOV, then macro-average images. | The paper reports Dice without defining its pixel domain or aggregation. Released code computes hard metrics over the entire resized square after applying a synthetic circle only to image and ground truth. | Make the valid-pixel domain explicit and prevent image size or background area from weighting the aggregate. | **Provisional** |
| Preserve raw acquisition files unchanged; place any derived representation separately and link it to raw hashes in the manifest. | Paper, README, YAML, and executable code provide no raw-data custody or transformation manifest. | Retain recoverable provenance and allow independent integrity verification. | **Provisional** |

### Named protocol separation

**Evidence and conflict:** Paper Sections 4.1, 4.2, and 5.3, the README, retinal
YAML, and executable code prescribe incompatible preprocessing or training
settings.

**Rationale:** Named protocols prevent a hybrid configuration from being
silently reported as either paper-faithful or released-code-faithful.

**Status:** **Provisional.** Two protocol names will prevent accidental
conflation:

1. `paper_section_5_3_reference` — the proposed primary Full Model versus
   No-SSTM comparison and provisional paper-faithful candidate. It uses Section
   5.3 as the training-budget authority and makes every missing choice below
   explicit and provisional; the name does not imply that unresolved author
   behavior has been confirmed.
2. `released_code_reference` — a separately labeled diagnostic description of
   the released retinal pipeline: 256×256 positive-filtered patches, stride 32,
   4,000 random patches per epoch, batch 16, Adam `5e-4`, synthetic circular FOV,
   and stochastic validation patches. It must not be reported as paper-faithful.

The primary comparison will use the released Full Model and official No-SSTM
architecture definitions, but not the released loader's stochastic validation,
generic FOV circle, or incomplete evaluation domain. Any exact released-loader
run would be a separate diagnostic and require separate approval.

### Primary training reference

For `paper_section_5_3_reference`, provisionally use:

| Field | Provisional choice | Evidence | Conflict | Rationale | Status |
|---|---|---|---|---|---|
| Optimizer | Adam | Directly specified by Section 5.3; also used by executable code and the README table. | Paper Section 4.2 instead specifies RMSprop. | Give the target ablation section priority over the paper's general regime. | **Provisional** |
| Learning rate | `1e-4` | Directly specified by Section 5.3; also stated by Section 4.2 and the README. | Retinal YAML specifies `5e-4`. | Use the target section's explicit value. | **Provisional** |
| Batch size | 8 | Directly specified by Section 5.3. | Section 4.2 says 4, the README says 8 per GPU, and retinal YAML says 16. | Use the target section's explicit value without inferring a multi-device effective batch. | **Provisional** |
| Maximum epochs | 100 | Directly specified by Section 5.3 and retinal YAML. | Section 4.2 says 45; README says 150. | Use the target section's explicit budget. | **Provisional** |
| Dataset expansion | 30× | Directly specified by Section 5.3; default YAML and the README full-image example also contain 30×. | Section 4.2 describes 40 augmentations per sample; released retinal patch mode ignores `expansion_factor`. | Use the target section's explicit expansion. With 16 training images, this provisionally gives 480 augmented samples and 60 full batches per epoch. | **Provisional** |

The Full Model and No-SSTM runs must use identical data, sample order,
augmentations, loss, optimizer settings, checkpoint rule, threshold, and seeds.
Only the SSTM ablation setting may differ.

### Fields not fully specified by Section 5.3

#### Image size and inference geometry

- **Evidence:** Paper Section 4.1 says all images are resized to 352×352 with
  Lanczos interpolation. The README's full-image example also uses 352, while
  the retinal example and retinal YAML use 256×256 patches. The executable test
  path resizes the entire image to 256×256 and therefore executes one patch.
- **Conflict:** The paper's global 352×352 statement and README full-image
  example disagree with the released retinal-specific 256×256 patch path;
  Section 5.3 does not resolve which applies to DRIVE.
- **Provisional choice:** Use 352×352 complete-image training for
  `paper_section_5_3_reference`. At validation and test time, predict the entire
  352×352 preprocessed image deterministically, resize the probability map back
  to the original image dimensions, and calculate metrics on the original mask
  and FOV grid. Record an explicit coverage assertion for every original pixel.
- **Rationale:** This follows the paper's only explicit image-size statement and
  avoids the released test path's nominal sliding-window operation collapsing to
  one 256×256 patch. Native-resolution 256-patch overlap tiling belongs to the
  separate `released_code_reference` or a separately labeled sensitivity study.

#### Interpolation

- **Evidence:** The paper names Lanczos but does not distinguish images from
  discrete labels. Released code uses OpenCV's unspecified/default resize call
  for images and masks. The README and YAML do not specify separate image,
  vessel-mask, or FOV-mask interpolation rules.
- **Conflict:** Applying the paper's single Lanczos statement to discrete masks
  would not preserve their label set, while the executable default is not the
  named paper interpolation.
- **Provisional choice:** Use Lanczos for RGB image resizing, nearest-neighbor for
  vessel and FOV masks, and bilinear interpolation only for probability maps
  returned to the original grid. Re-binarize masks after nearest-neighbor resize
  and assert their values are `{0,1}`.
- **Rationale:** This follows the image evidence while preventing interpolation
  from inventing fractional annotation or FOV labels.

#### Preprocessing and channel handling

- **Evidence:** The paper does not specify CLAHE or green-channel replication.
  The README retinal example and released retinal YAML enable CLAHE; executable
  code uses green-channel CLAHE replicated into three channels in `[0,1]`.
- **Conflict:** The only retinal-specific implementation evidence adds
  preprocessing that is absent from the paper protocol, while Section 5.3 gives
  no alternative.
- **Provisional choice:** Use the released green-channel CLAHE implementation,
  replicated to three float32 channels in `[0,1]`, for both model variants.
- **Rationale:** It is the only released retinal-specific preprocessing path.
  This remains a declared released-code choice within the paper-reference
  protocol, not a paper claim.

#### Training augmentation

- **Evidence:** Section 4.1 specifies the augmentation family and probabilities
  listed above. Section 5.3 specifies 30× expansion but does not restate the
  transforms. Section 4.2 instead says 40 augmentations per sample. The released
  patch augmentation has different elastic, photometric, and probability values;
  adds grid distortion and blur; and omits ShiftScaleRotate and coarse dropout.
  The retinal YAML enables augmentation implicitly through the training split
  but does not enumerate transforms, and the README does not provide an
  alternative complete transform specification.
- **Conflict:** Section 4.2's 40 augmentations, Section 5.3's 30× expansion, and
  the released transform set cannot all describe one unchanged pipeline.
- **Provisional choice:** Combine Section 5.3's 30× expansion with Section 4.1's
  transforms and probabilities. Apply spatial transforms identically to image,
  vessel mask, and FOV mask, using discrete interpolation for both masks. Generate
  every augmentation from an explicit deterministic seed schedule.
- **Rationale:** Section 5.3 is authoritative for the target ablation budget,
  while Section 4.1 is the paper's only detailed augmentation specification.
  The 40× statement is not used because it conflicts with the target section's
  explicit 30× expansion.

#### Checkpoint selection and early stopping

- **Evidence:** Paper Section 4.2 specifies validation-Dice early stopping with
  patience 15. The README specifies patience 50. Executable code falls back to
  patience 40 and monitors batch-global soft validation Dice on stochastic
  patches. The retinal YAML omits checkpoint and patience fields. Section 5.3
  specifies 100 epochs but no checkpoint rule.
- **Conflict:** Patience values 15, 40, and 50 appear in different sources, and
  released checkpoint selection uses stochastic patch-level validation rather
  than the proposed fixed full-image endpoint.
- **Provisional choice:** Train all 100 epochs without early termination and
  select the checkpoint with the highest fixed-validation primary metric:
  per-image hard Dice inside the supplied FOV, macro-averaged over IDs 37-40.
  Define a deterministic tie-breaker before execution: earliest epoch wins.
- **Rationale:** This holds the Section 5.3 budget constant across paired variants
  and avoids checkpoint selection from stochastic validation patches. It is a
  provisional reproduction rule, not an author-reported rule.

#### Threshold

- **Evidence:** The paper does not state a threshold. Released metrics use 0.5,
  and `official_repo/configs/default.yaml:70-73` declares 0.5. The retinal YAML
  does not contain an evaluation section because the claimed YAML inheritance is
  not implemented. The README does not state a DRIVE decision threshold.
- **Conflict:** There is no paper- or README-confirmed DRIVE threshold; 0.5 is
  supported only by default configuration and executable fallbacks.
- **Provisional choice:** Fix the probability threshold at 0.5 for validation and
  test. Do not tune it on test data.
- **Rationale:** It is the only explicit released threshold and avoids an
  unreported validation optimization.

#### Test-time augmentation

- **Evidence:** The paper does not state that TTA was used. The default YAML says
  `use_tta: true` and `tta_n: 8`, but executable evaluation ignores those values
  unless the CLI `--tta` flag is supplied; the implementation is always eight-way.
  The README demonstrates optional `--tta` evaluation at
  `official_repo/README.md:491-497` but does not tie it to the DRIVE result.
- **Conflict:** Default YAML implies TTA is enabled, executable behavior requires
  an explicit CLI flag, and neither paper nor README connects TTA to the target
  DRIVE number.
- **Provisional choice:** Disable TTA for the primary comparison. If later
  approved, report eight-way TTA only as a separately labeled secondary result
  applied identically to both model variants.
- **Rationale:** No paper evidence links TTA to the Section 5.3 DRIVE result, and
  no-TTA evaluation minimizes an undocumented source of protocol variation.

#### Loss composition

- **Evidence:** The paper defines five-component MASL, but Section 5.3 does not
  restate which loss configuration was used for the Full Model versus No-SSTM
  row. The default YAML, `train.build_loss`, and official No-SSTM ablation all use
  the five components with learned weights and morphology modulation. The README
  describes the same five component categories.
- **Conflict:** General paper evidence supports MASL, but the target ablation
  paragraph does not state whether the Full Model versus No-SSTM row used the
  complete learned and morphology-modulated released configuration.
- **Provisional choice:** Use the released five-component `MorphologyAwareLoss`
  configuration with learned weights and morphology modulation, identically for
  Full Model and No-SSTM.
- **Rationale:** This is the only loss configuration shared by the released Full
  Model defaults and official No-SSTM definition. Before training, a diagnostic
  must confirm that the loss's learned weights receive gradients and optimizer
  updates; failure requires review rather than a silent implementation change.

#### Seed handling

- **Evidence:** The paper reports five runs but does not list seeds. The README
  alone lists `{42, 123, 456, 789, 2024}`. The retinal executable defaults to 42,
  and the released patch loader separately resets NumPy to 42.
- **Conflict:** The README seed list is not paper-confirmed, and the executable
  data sampler does not consistently inherit the requested run seed.
- **Provisional choice:** Use paired seeds `{42, 123, 456, 789, 2024}` for the
  final comparison. Use independent, explicitly constructed RNG streams for
  model initialization, training order, augmentation, and any sampling; derive
  them deterministically from the run seed. Give Full Model and No-SSTM identical
  sample and augmentation schedules within each seed.
- **Rationale:** The README provides the only explicit five-seed set, and paired
  deterministic schedules reduce irrelevant variance in the ablation difference.
  The seed list remains README-derived, not paper-confirmed.

### Evaluation endpoint

**Evidence:** The paper reports Dice but does not define the DRIVE pixel domain
or aggregation. Released evaluation computes per-image hard metrics on the full
resized square and does not use supplied FOV masks.

**Conflict:** The reported claim cannot be tied to either full-square or supplied
FOV-only scoring, and released preprocessing does not guarantee native-grid
coverage.

**Rationale:** A supplied-FOV, per-image macro endpoint makes both pixel validity
and image weighting explicit while preventing test-driven selection.

**Status:** **Provisional.** The endpoint is:

- Evaluate every validation and test image completely and deterministically.
- Assert that the inference coverage map is nonzero at every evaluated pixel.
- Retain predictions as probabilities until thresholding at 0.5.
- For each image, restrict both prediction and first-manual ground truth to pixels
  where the supplied official FOV mask equals one.
- Compute hard Dice separately for each image inside that FOV.
- Macro-average the per-image Dice values across IDs 01-20 for the test result.
- Never select a checkpoint, threshold, transform, or protocol using test IDs.
- Report Full Model and No-SSTM with exactly the same evaluator.

This FOV-restricted macro hard Dice is the primary endpoint. Any soft Dice, IoU,
precision, recall, specificity, F1, topology, boundary, or TTA metric is
secondary and must be labeled with its exact domain and aggregation rule.

## 5. Planned dataset manifest

The reproduction-owned manifest will contain one row per DRIVE image and at
least these fields:

| Field | Required content |
|---|---|
| `image_id` | Canonical two-digit DRIVE ID. |
| `assigned_split` | Exactly one of `train`, `validation`, or `test`. |
| `relative_image_path` | Path relative to the preserved dataset root. |
| `relative_vessel_mask_path` | Path to the selected manual vessel annotation. |
| `relative_fov_mask_path` | Path to the supplied official FOV mask. |
| `annotation_observer` | Explicit observer identity; provisionally `first_manual`. |
| `original_dimensions` | Original height, width, and channel count. |
| `file_format` | Detected container/extension for image and masks. |
| `sha256_image` | SHA-256 of the raw image file. |
| `sha256_vessel_mask` | SHA-256 of the raw vessel-mask file. |
| `sha256_fov_mask` | SHA-256 of the raw FOV-mask file. |
| `provenance_source` | Authoritative source identifier, acquisition date, and archive identity or revision. |
| `integrity_check_status` | `pending`, `pass`, or `fail`, with failures blocking use. |

Integrity checks must verify unique IDs, the 16/4/20 assigned counts, no
cross-split path or hash overlap, readable files, matching spatial dimensions,
binary vessel and FOV masks after decoding, one first-manual annotation per
image, one FOV mask per image, and exactly one manifest row per expected ID.

## 6. Pre-download acquisition checklist

No acquisition may begin until every pre-download item is reviewed.

- [ ] Identify a legitimate authoritative DRIVE source operated by the dataset
      owner or its designated institutional host; record the exact source page
      and do not substitute an unverified mirror silently.
- [ ] Confirm the expected archive identity and contents before transfer: 20
      official training images, 20 official test images, manual vessel
      annotations including the first observer, and a corresponding FOV mask for
      every image.
- [ ] Review the license, registration conditions, citation requirements,
      redistribution limits, and terms of use. Record the review outcome without
      publishing credentials or restricted text.
- [ ] Confirm that the anticipated image, vessel-mask, and FOV-mask filenames or
      IDs can be paired unambiguously before writing an adapter.
- [ ] Define a raw-data location under ignored `data/` storage. Preserve the
      downloaded archive and extracted raw files byte-for-byte; never normalize,
      rename, or overwrite the only raw copy.
- [ ] Record SHA-256 for the archive before extraction and for every image,
      vessel mask, and FOV mask after extraction.
- [ ] Verify the expected official split: test IDs 01-20 and training IDs 21-40.
      Apply the reproduction-owned 21-36 / 37-40 train/validation assignment only
      in the manifest or derived view; do not alter the official test set.
- [ ] Verify one-to-one image / first-manual vessel-mask / FOV-mask
      correspondence, dimensions, decodability, and binary-mask values.
- [ ] Verify that `/data/` and reproduction raw/interim/processed data paths are
      excluded by `.gitignore:45-50`. Run an ignore check before and after
      acquisition, and never use force-add for dataset content.
- [ ] Produce and review the manifest and integrity report before allowing any
      preprocessing, training, validation, or test evaluation.

## 7. Unresolved questions

1. What authoritative archive name, version, checksum, and terms currently apply
   to DRIVE acquisition?
2. What exact raw filenames, encodings, and directory layout will the
   authoritative source provide?
3. Can the authors confirm the exact DRIVE train/validation split and whether the
   first manual annotation was used?
4. Did the reported DRIVE results use supplied FOV masks, and was loss or only
   evaluation restricted to the FOV?
5. Does Section 4.1's 352×352 Lanczos statement apply unchanged to the Section
   5.3 DRIVE ablation?
6. Did the Section 5.3 30× expansion use the Section 4.1 augmentation pipeline,
   another pipeline, or the released retinal patch augmentation?
7. Was Section 5.3 trained for every one of the 100 epochs, or was early stopping
   used? Which validation metric and tie-breaking rule selected the checkpoint?
8. Was the DRIVE test threshold fixed at 0.5, selected on validation, or chosen by
   another rule?
9. Was TTA used for the reported DRIVE ablation result?
10. Did the Full Model versus No-SSTM row use the complete learned and
    morphology-modulated MASL configuration?
11. Which five seeds produced the paper's mean and standard deviation, and were
    data order and augmentations paired across ablations?
12. Should the primary paper-reference evaluator score on the original image
    grid after inverse resizing, or on the 352×352 grid? The current provisional
    decision uses the original grid and must remain labeled as such.

Until these questions are resolved or the provisional choices are explicitly
approved, `PROTOCOL_STATUS=PROVISIONAL` remains in force. No DRIVE acquisition or
experiment is authorized by this document alone.
