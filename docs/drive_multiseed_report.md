# DRIVE multi-seed experimental result audit

## Scope

This document audits the completed Full S2M-Net and No-SSTM DRIVE runs for
seeds 42, 7, and 123. It reports the recorded experiment state and computes
descriptive multi-seed summaries. It does **not** decide whether any paper
claim is verified.

Only local run artifacts were read. No test images, Trackio/logbook content,
claim pages, GitHub, or Hugging Face resources were accessed or updated.

## Experimental setup

All six frozen configurations use:

- training IDs 21–36 (16 images), validation IDs 37–40 (4 images), and
  hidden-test IDs 1–20 with `access_hidden_test: false`;
- a 352 × 352 input, green-channel CLAHE with clip limit 2.0 and an 8 × 8
  tile grid, Lanczos4 image resizing, nearest-neighbour discrete-mask
  resizing, bilinear probability resizing, and supplied-FOV masking;
- batch size 2, 30 deterministic replicas per training image, 480 samples and
  240 optimizer steps per epoch, 100 epochs, and therefore 24,000 optimizer
  steps per run;
- Adam at base learning rate \(10^{-4}\), 10 warm-up epochs, cosine decay to
  \(10^{-6}\), and no early stopping;
- the same morphology-aware loss, augmentation, validation split, fixed
  threshold 0.5, no test-time augmentation, and identical recovery settings.

Every numerical setup value above is recorded identically in these six exact
configuration sources:

- `outputs/drive/full_seed42/config/frozen.yaml`
- `outputs/drive/full_seed7/config/frozen.yaml`
- `outputs/drive/full_seed123/config/frozen.yaml`
- `outputs/drive/no_sstm_seed42/config/frozen.yaml`
- `outputs/drive/no_sstm_seed7/config/frozen.yaml`
- `outputs/drive/no_sstm_seed123/config/frozen.yaml`

The train/validation IDs, GPU device, software versions, repository commit,
trainer hash, and `hidden_test_accessed: false` flags are independently
recorded in:

- `outputs/drive/full_seed42/config/runtime.json`
- `outputs/drive/full_seed7/config/runtime.json`
- `outputs/drive/full_seed123/config/runtime.json`
- `outputs/drive/no_sstm_seed42/config/runtime.json`
- `outputs/drive/no_sstm_seed7/config/runtime.json`
- `outputs/drive/no_sstm_seed123/config/runtime.json`

Programmatic canonical comparison established the following:

- after removing only `run.name`, `run.output_dir`, and `run.seed`, all three
  Full configurations are identical;
- after removing those same operational identity fields, all three No-SSTM
  configurations are identical;
- the shared `data`, `preprocessing`, `augmentation`, `loss`, `optimizer`,
  `learning_rate_schedule`, `training`, `validation`, and `recovery` sections
  are identical across all six configurations;
- after accounting for run identity, the only Full-versus-No-SSTM
  configuration differences are `run.variant`, `model.use_sstm`, and the five
  Boolean entries of `model.sstm_stages`. Full sets SSTM and all five stages
  to true; No-SSTM sets them to false.

The recorded parameter counts are 4,791,544 for Full and 4,503,720 for
No-SSTM. Sources: the `[Model] parameters=...` line in each exact log path
listed in the artifact inventory below.

## Artifact inventory

All paths are relative to the project root.

| Run | Frozen configuration | Log | Status/runtime | Metric files | Selected best checkpoint | Final recovery checkpoint |
|---|---|---|---|---|---|---|
| Full, seed 42 | `outputs/drive/full_seed42/config/frozen.yaml` | `outputs/drive/full_seed42/logs/train.log` | `outputs/drive/full_seed42/status.json`; `outputs/drive/full_seed42/config/runtime.json` | `outputs/drive/full_seed42/metrics/history.json`; final: `outputs/drive/full_seed42/metrics/epochs/epoch_100.json` | metadata: `outputs/drive/full_seed42/checkpoints/best/checkpoint`; prefix: `outputs/drive/full_seed42/checkpoints/best/ckpt-21` | metadata: `outputs/drive/full_seed42/checkpoints/latest/checkpoint`; prefix: `outputs/drive/full_seed42/checkpoints/latest/ckpt-24000` |
| Full, seed 7 | `outputs/drive/full_seed7/config/frozen.yaml` | `outputs/drive/full_seed7/logs/train.log` | `outputs/drive/full_seed7/status.json`; `outputs/drive/full_seed7/config/runtime.json` | `outputs/drive/full_seed7/metrics/history.json`; final: `outputs/drive/full_seed7/metrics/epochs/epoch_100.json` | metadata: `outputs/drive/full_seed7/checkpoints/best/checkpoint`; prefix: `outputs/drive/full_seed7/checkpoints/best/ckpt-14` | metadata: `outputs/drive/full_seed7/checkpoints/latest/checkpoint`; prefix: `outputs/drive/full_seed7/checkpoints/latest/ckpt-24000` |
| Full, seed 123 | `outputs/drive/full_seed123/config/frozen.yaml` | `outputs/drive/full_seed123/logs/train.log` | `outputs/drive/full_seed123/status.json`; `outputs/drive/full_seed123/config/runtime.json` | `outputs/drive/full_seed123/metrics/history.json`; final: `outputs/drive/full_seed123/metrics/epochs/epoch_100.json` | metadata: `outputs/drive/full_seed123/checkpoints/best/checkpoint`; prefix: `outputs/drive/full_seed123/checkpoints/best/ckpt-24` | metadata: `outputs/drive/full_seed123/checkpoints/latest/checkpoint`; prefix: `outputs/drive/full_seed123/checkpoints/latest/ckpt-24000` |
| No-SSTM, seed 42 | `outputs/drive/no_sstm_seed42/config/frozen.yaml` | `outputs/drive/no_sstm_seed42/logs/train.log` | `outputs/drive/no_sstm_seed42/status.json`; `outputs/drive/no_sstm_seed42/config/runtime.json` | `outputs/drive/no_sstm_seed42/metrics/history.json`; final: `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_100.json` | metadata: `outputs/drive/no_sstm_seed42/checkpoints/best/checkpoint`; prefix: `outputs/drive/no_sstm_seed42/checkpoints/best/ckpt-13` | metadata: `outputs/drive/no_sstm_seed42/checkpoints/latest/checkpoint`; prefix: `outputs/drive/no_sstm_seed42/checkpoints/latest/ckpt-24000` |
| No-SSTM, seed 7 | `outputs/drive/no_sstm_seed7/config/frozen.yaml` | `outputs/drive/no_sstm_seed7/logs/train.log` | `outputs/drive/no_sstm_seed7/status.json`; `outputs/drive/no_sstm_seed7/config/runtime.json` | `outputs/drive/no_sstm_seed7/metrics/history.json`; final: `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_100.json` | metadata: `outputs/drive/no_sstm_seed7/checkpoints/best/checkpoint`; prefix: `outputs/drive/no_sstm_seed7/checkpoints/best/ckpt-14` | metadata: `outputs/drive/no_sstm_seed7/checkpoints/latest/checkpoint`; prefix: `outputs/drive/no_sstm_seed7/checkpoints/latest/ckpt-24000` |
| No-SSTM, seed 123 | `outputs/drive/no_sstm_seed123/config/frozen.yaml` | `outputs/drive/no_sstm_seed123/logs/train.log` | `outputs/drive/no_sstm_seed123/status.json`; `outputs/drive/no_sstm_seed123/config/runtime.json` | `outputs/drive/no_sstm_seed123/metrics/history.json`; final: `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_100.json` | metadata: `outputs/drive/no_sstm_seed123/checkpoints/best/checkpoint`; prefix: `outputs/drive/no_sstm_seed123/checkpoints/best/ckpt-34` | metadata: `outputs/drive/no_sstm_seed123/checkpoints/latest/checkpoint`; prefix: `outputs/drive/no_sstm_seed123/checkpoints/latest/ckpt-24000` |

Each checkpoint prefix above consists of an `.index` file and one
`.data-00000-of-00001` shard. TensorFlow's checkpoint reader successfully read
every tensor in each selected best checkpoint; no numeric tensor was
non-finite. The checkpointed best epoch, best metric, completed-epoch count,
and optimizer iteration agreed with the corresponding status and epoch-metric
files. Sources: the six best-checkpoint prefixes in the table and the six
corresponding `status.json` files.

## Run-validation table

Wall time is `last_update_time - start_time`. Each row's status, epoch count,
step count, best epoch/metric, best/latest paths, and timestamps come from the
row's exact `status.json`; parameter count and completion marker come from the
row's exact log; the final-epoch check comes from the listed
`metrics/epochs/epoch_100.json`.

| Variant | Seed | Status | Epochs | Final optimizer step | Parameters | Wall time | Selected epoch / step | Selected validation FOV macro hard Dice | Recovery/restart | Required outputs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| Full | 42 | completed | 100 | 24,000 | 4,791,544 | 4,362.021432 s | 21 / 5,040 | 0.767566043502 | no restore; all epoch resume offsets 0 | complete |
| Full | 7 | completed | 100 | 24,000 | 4,791,544 | 4,130.858726 s | 14 / 3,360 | 0.761168602150 | launched new; no restore; all epoch resume offsets 0 | complete |
| Full | 123 | completed | 100 | 24,000 | 4,791,544 | 4,075.777422 s | 24 / 5,760 | 0.767830616927 | launched new; no restore; all epoch resume offsets 0 | complete |
| No-SSTM | 42 | completed | 100 | 24,000 | 4,503,720 | 3,893.563035 s | 13 / 3,120 | 0.761247455426 | no restore; all epoch resume offsets 0 | complete |
| No-SSTM | 7 | completed | 100 | 24,000 | 4,503,720 | 3,886.555447 s | 14 / 3,360 | 0.765539691798 | launched new; no restore; all epoch resume offsets 0 | complete |
| No-SSTM | 123 | completed | 100 | 24,000 | 4,503,720 | 3,826.175806 s | 34 / 8,160 | 0.771879465983 | launched new; no restore; all epoch resume offsets 0 | complete |

Exact row sources:

- Full 42: `outputs/drive/full_seed42/status.json`,
  `outputs/drive/full_seed42/logs/train.log`, and
  `outputs/drive/full_seed42/metrics/epochs/epoch_100.json`.
- Full 7: `outputs/drive/full_seed7/status.json`,
  `outputs/drive/full_seed7/logs/train.log`, and
  `outputs/drive/full_seed7/metrics/epochs/epoch_100.json`.
- Full 123: `outputs/drive/full_seed123/status.json`,
  `outputs/drive/full_seed123/logs/train.log`, and
  `outputs/drive/full_seed123/metrics/epochs/epoch_100.json`.
- No-SSTM 42: `outputs/drive/no_sstm_seed42/status.json`,
  `outputs/drive/no_sstm_seed42/logs/train.log`, and
  `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_100.json`.
- No-SSTM 7: `outputs/drive/no_sstm_seed7/status.json`,
  `outputs/drive/no_sstm_seed7/logs/train.log`, and
  `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_100.json`.
- No-SSTM 123: `outputs/drive/no_sstm_seed123/status.json`,
  `outputs/drive/no_sstm_seed123/logs/train.log`, and
  `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_100.json`.

For seeds 7 and 123, the sequential controller's `mode=new` decisions and
successful completion records are in
`outputs/drive/.screen/drive_seeds_7_123.log`. Seed-42 and per-epoch
restart evidence comes from the six exact train logs and all 600 files under
the six `metrics/epochs/` directories.

## Model selection and checkpoint used

The configured primary selection metric is native-grid, supplied-FOV,
per-image hard Dice macro-averaged across validation IDs 37–40 at threshold
0.5. The checkpoint direction is maximum; strict improvement replaces the
best checkpoint, so an exact tie retains the earlier epoch. Test-time
augmentation is disabled. Sources: each frozen configuration's `validation`
section and the implementation in
`repro/experiments/drive_train.py` (`RecoveryAndValidationCallback.on_epoch_end`).

The selected validation result for every run is the metric record from the
best epoch below, paired with its `checkpoints/best/ckpt-*` checkpoint. The
`checkpoints/latest/ckpt-24000` files are final recovery state, not the
model-selected result.

| Variant | Seed | Selected metric file | Selected checkpoint |
|---|---:|---|---|
| Full | 42 | `outputs/drive/full_seed42/metrics/epochs/epoch_021.json` | `outputs/drive/full_seed42/checkpoints/best/ckpt-21` |
| Full | 7 | `outputs/drive/full_seed7/metrics/epochs/epoch_014.json` | `outputs/drive/full_seed7/checkpoints/best/ckpt-14` |
| Full | 123 | `outputs/drive/full_seed123/metrics/epochs/epoch_024.json` | `outputs/drive/full_seed123/checkpoints/best/ckpt-24` |
| No-SSTM | 42 | `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_013.json` | `outputs/drive/no_sstm_seed42/checkpoints/best/ckpt-13` |
| No-SSTM | 7 | `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_014.json` | `outputs/drive/no_sstm_seed7/checkpoints/best/ckpt-14` |
| No-SSTM | 123 | `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_034.json` | `outputs/drive/no_sstm_seed123/checkpoints/best/ckpt-34` |

## Per-seed results

### Model-selected validation metrics

These are all validation metrics present in each selected epoch file.
`FOV37`–`FOV40` are the native-grid supplied-FOV hard Dice values for the four
validation images. All other Dice/IoU/precision/recall fields are the recorded
resized-batch auxiliary metrics.

| Variant | Seed | Val loss | Soft Dice | Soft IoU | Precision | Recall | FOV37 | FOV38 | FOV39 | FOV40 | FOV macro hard Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 42 | 0.156467571855 | 0.646622627974 | 0.477900102735 | 0.698801010847 | 0.753058433533 | 0.785165304383 | 0.760115962404 | 0.759048958731 | 0.765933948488 | 0.767566043502 |
| Full | 7 | 0.186492361128 | 0.660509586334 | 0.493105188012 | 0.731959342957 | 0.747199803591 | 0.773668534151 | 0.743640133422 | 0.761715451735 | 0.765650289291 | 0.761168602150 |
| Full | 123 | 0.147577464581 | 0.639969944954 | 0.470618382096 | 0.689222574234 | 0.756868779659 | 0.786917808223 | 0.764591507243 | 0.756557363120 | 0.763255789122 | 0.767830616927 |
| No-SSTM | 42 | 0.199760064483 | 0.651072591543 | 0.482713237405 | 0.698383122683 | 0.756037175655 | 0.775510204086 | 0.754305355832 | 0.755982335426 | 0.759191926362 | 0.761247455426 |
| No-SSTM | 7 | 0.191166765988 | 0.658678382635 | 0.491070419550 | 0.713289260864 | 0.765334695578 | 0.775439334951 | 0.756519018153 | 0.763334704515 | 0.766865709571 | 0.765539691798 |
| No-SSTM | 123 | 0.101066272706 | 0.625256508589 | 0.454857632518 | 0.724745333195 | 0.753917396069 | 0.793929654118 | 0.758930136366 | 0.767282706875 | 0.767375366573 | 0.771879465983 |

Exact row sources, respectively:

1. `outputs/drive/full_seed42/metrics/epochs/epoch_021.json`
2. `outputs/drive/full_seed7/metrics/epochs/epoch_014.json`
3. `outputs/drive/full_seed123/metrics/epochs/epoch_024.json`
4. `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_013.json`
5. `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_014.json`
6. `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_034.json`

### Final epoch-100 training metrics

These are the training metrics at the completed budget endpoint, not the
model-selected validation result.

| Variant | Seed | Train loss | Soft Dice | Soft IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Full | 42 | 0.036992838793 | 0.535214358817 | 0.365988904734 | 0.799106692026 | 0.718935353309 |
| Full | 7 | 0.037194120356 | 0.539140093823 | 0.369869398822 | 0.801096682747 | 0.719521998366 |
| Full | 123 | 0.037063835608 | 0.536860561247 | 0.367687555030 | 0.801934443911 | 0.715915527940 |
| No-SSTM | 42 | 0.035809446196 | 0.559184655671 | 0.388930890337 | 0.804143506040 | 0.724743333956 |
| No-SSTM | 7 | 0.036049370204 | 0.563780705507 | 0.393345133836 | 0.805666958044 | 0.726310839007 |
| No-SSTM | 123 | 0.035736761491 | 0.558057900394 | 0.387849914407 | 0.804124551018 | 0.725557515522 |

### Final epoch-100 validation metrics

These are the validation metrics at the completed budget endpoint, not the
model-selected result.

| Variant | Seed | Val loss | Soft Dice | Soft IoU | Precision | Recall | FOV37 | FOV38 | FOV39 | FOV40 | FOV macro hard Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 42 | 0.036202808842 | 0.520089417696 | 0.351451218128 | 0.767983317375 | 0.798512935638 | 0.736196988439 | 0.711779007087 | 0.742645355576 | 0.713892659310 | 0.726128502603 |
| Full | 7 | 0.036958668381 | 0.526704311371 | 0.357506930828 | 0.770649909973 | 0.781913906336 | 0.739781917971 | 0.719493204988 | 0.745087102836 | 0.719565578772 | 0.730981951142 |
| Full | 123 | 0.037343518808 | 0.522290945053 | 0.353482127190 | 0.762389332056 | 0.772605478764 | 0.744274532546 | 0.707373271895 | 0.766790908431 | 0.727957573605 | 0.736599071619 |
| No-SSTM | 42 | 0.037904849276 | 0.547833830118 | 0.377281725407 | 0.755706787109 | 0.770461469889 | 0.745637684420 | 0.719463460407 | 0.768620192568 | 0.727488767596 | 0.740302526248 |
| No-SSTM | 7 | 0.036741876975 | 0.551664680243 | 0.380905896425 | 0.771457582712 | 0.777540296316 | 0.754621347649 | 0.721299294098 | 0.763540671586 | 0.716642579633 | 0.739025973241 |
| No-SSTM | 123 | 0.036632217467 | 0.554575711489 | 0.383682474494 | 0.773484975100 | 0.772677540779 | 0.750118936616 | 0.730667228696 | 0.768723743304 | 0.728943395528 | 0.744613326036 |

The exact row sources for both epoch-100 tables are:

1. `outputs/drive/full_seed42/metrics/epochs/epoch_100.json`
2. `outputs/drive/full_seed7/metrics/epochs/epoch_100.json`
3. `outputs/drive/full_seed123/metrics/epochs/epoch_100.json`
4. `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_100.json`
5. `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_100.json`
6. `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_100.json`

### Test results

No test metric, prediction, evaluation, or inference result file exists in any
of the six run directories. Every `config/runtime.json` records
`hidden_test_accessed: false`. Consequently, this report contains **validation
results only** and no test result is inferred or substituted.

Sources: the complete file inventories under
`outputs/drive/full_seed42/`, `outputs/drive/full_seed7/`,
`outputs/drive/full_seed123/`, `outputs/drive/no_sstm_seed42/`,
`outputs/drive/no_sstm_seed7/`, and
`outputs/drive/no_sstm_seed123/`, together with each run's exact
`config/runtime.json`.

## Aggregate results

All aggregate values were calculated programmatically with Python's arithmetic
mean and sample standard deviation (`statistics.stdev`, denominator
\(n-1\)). The sample size is three seeds per architecture.

For every Full aggregate number below, the exact inputs are:

- `outputs/drive/full_seed42/metrics/epochs/epoch_021.json`
- `outputs/drive/full_seed7/metrics/epochs/epoch_014.json`
- `outputs/drive/full_seed123/metrics/epochs/epoch_024.json`

For every No-SSTM aggregate number below, the exact inputs are:

- `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_013.json`
- `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_014.json`
- `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_034.json`

### Selected-checkpoint validation metrics: mean and sample standard deviation

| Validation metric | Full mean | Full sample SD | No-SSTM mean | No-SSTM sample SD |
|---|---:|---:|---:|---:|
| Loss | 0.163512465854 | 0.020391547071 | 0.163997701059 | 0.054669321385 |
| Soft Dice | 0.649034053087 | 0.010480002162 | 0.645002494256 | 0.017518275514 |
| Soft IoU | 0.480541224281 | 0.011473698285 | 0.476213763158 | 0.018961114426 |
| Precision | 0.706660976013 | 0.022426371519 | 0.712139238914 | 0.013218678022 |
| Recall | 0.752375672261 | 0.004870512988 | 0.758429755767 | 0.006073056209 |
| FOV hard Dice, image 37 | 0.781917215586 | 0.007197108898 | 0.781626397718 | 0.010654991513 |
| FOV hard Dice, image 38 | 0.756115867690 | 0.011033611582 | 0.756584836784 | 0.002313092695 |
| FOV hard Dice, image 39 | 0.759107257862 | 0.002579538453 | 0.762199915605 | 0.005735015966 |
| FOV hard Dice, image 40 | 0.764946675634 | 0.001471203090 | 0.764477667502 | 0.004584673610 |
| FOV macro hard Dice | 0.765521754193 | 0.003772260504 | 0.766222204402 | 0.005348764316 |

For completeness, aggregating the epoch-100 primary validation metric rather
than the selected checkpoints gives:

| Endpoint | Full mean | Full sample SD | No-SSTM mean | No-SSTM sample SD |
|---|---:|---:|---:|---:|
| Epoch-100 validation FOV macro hard Dice | 0.731236508455 | 0.005239923993 | 0.741313941842 | 0.002927772010 |

The exact Full epoch-100 inputs are
`outputs/drive/full_seed42/metrics/epochs/epoch_100.json`,
`outputs/drive/full_seed7/metrics/epochs/epoch_100.json`, and
`outputs/drive/full_seed123/metrics/epochs/epoch_100.json`. The exact No-SSTM
inputs are `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_100.json`,
`outputs/drive/no_sstm_seed7/metrics/epochs/epoch_100.json`, and
`outputs/drive/no_sstm_seed123/metrics/epochs/epoch_100.json`.

## Paired Full-minus-No-SSTM differences

The paired endpoint is the model-selected validation FOV macro hard Dice.
Each subtraction uses runs with the same seed.

| Seed | Full selected Dice | No-SSTM selected Dice | Full − No-SSTM |
|---:|---:|---:|---:|
| 42 | 0.767566043502 | 0.761247455426 | +0.006318588075 |
| 7 | 0.761168602150 | 0.765539691798 | −0.004371089648 |
| 123 | 0.767830616927 | 0.771879465983 | −0.004048849056 |
| Mean paired difference | — | — | **−0.000700450210** |

Exact sources:

- seed 42:
  `outputs/drive/full_seed42/metrics/epochs/epoch_021.json` and
  `outputs/drive/no_sstm_seed42/metrics/epochs/epoch_013.json`;
- seed 7:
  `outputs/drive/full_seed7/metrics/epochs/epoch_014.json` and
  `outputs/drive/no_sstm_seed7/metrics/epochs/epoch_014.json`;
- seed 123:
  `outputs/drive/full_seed123/metrics/epochs/epoch_024.json` and
  `outputs/drive/no_sstm_seed123/metrics/epochs/epoch_034.json`;
- mean paired difference: the three paired differences above, calculated
  programmatically as their arithmetic mean.

## Anomalies and limitations

- **No restart or recovery restore occurred.** Every one of the 600 epoch
  records has `resumed_from_batch: 0`; no train log contains
  `[Recovery] Restored`, an interruption marker, a failure marker, or a
  traceback. The seed-7/123 campaign log records all four runs as `mode=new`.
  Sources: all files under the six `metrics/epochs/` directories, the six
  exact `logs/train.log` paths in the artifact inventory, and
  `outputs/drive/.screen/drive_seeds_7_123.log`.
- **Recovery checkpoints were nevertheless active.** Each run log records 480
  periodic recovery saves, consistent with 24,000 steps and a 50-step
  interval; each run retains latest checkpoints at steps 23,900, 23,950, and
  24,000 plus its separate best checkpoint. Sources: the six train logs, the
  six frozen configurations, and each run's `checkpoints/latest/checkpoint`.
- **No OOM or dynamic OOM workaround was recorded.** No run or campaign log
  contains an out-of-memory or resource-exhaustion marker. Batch size 2 was
  fixed before all six runs as the common local GPU-memory adaptation; it was
  not changed between variants or seeds. Sources: the six frozen
  configurations, the six train logs,
  `outputs/drive/.screen/drive_seeds_7_123.log`, and
  `docs/drive_full_seed42_run.md`.
- **Non-fatal warnings were present.** Every run-local train log contains one
  Albumentations warning that `ShiftScaleRotate` is a special case of
  `Affine`. In the combined seed-7/123 console log, each of the four runs also
  records three duplicate CUDA-plugin registration messages, one TensorFlow
  layout-optimizer `INVALID_ARGUMENT` message, and two “cannot spawn child
  process” informational messages. Full seeds 7 and 123 each additionally
  record 10 complex64-to-float32 cast warnings; the No-SSTM runs record none.
  All four runs continued to completion. Sources: the six exact train logs
  and programmatic segmentation of
  `outputs/drive/.screen/drive_seeds_7_123.log` between each run's
  `[Campaign] new run=...` and `[Campaign] completed run=...` markers.
- **Seed-42 parent-console diagnostics are unavailable.** C/C++ TensorFlow
  messages emitted before or outside the Python-level tee are not present in
  the seed-42 run-local logs, and no separate seed-42 screen log is present.
  Warning-count parity with the newer campaign therefore cannot be assessed.
  Sources: `outputs/drive/full_seed42/logs/train.log`,
  `outputs/drive/no_sstm_seed42/logs/train.log`, and the file inventory under
  `outputs/drive/.screen/`.
- **No required training artifact is missing.** All six runs have frozen and
  runtime configurations, status, history, 100 per-epoch metric files,
  completion logs, one readable best checkpoint, and a readable final
  recovery checkpoint. No test output exists because the hidden test split
  was not accessed. Sources: the six run-directory inventories and exact
  paths in the artifact inventory.
- **Inference is limited by the experimental sample.** There are three paired
  seeds and four validation images, and checkpoint selection and reporting
  use the same validation set. The descriptive means and sample standard
  deviations above do not by themselves establish a paper-level claim.
  Sources for the seed count and validation-image count: the six frozen
  configurations and the selected metric files listed above.
