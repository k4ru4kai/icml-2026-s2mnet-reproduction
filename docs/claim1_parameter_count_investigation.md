# Claim 1 S2M-Net parameter-count investigation

## Scope

This is a staged, build-only investigation of the S2M-Net parameter count,
encoder structure, and TransUNet and Swin-Unet parameter comparisons. It does
not train a model, access any dataset, or change any claim or publication.

The released source was inspected at the pinned submodule commit
`3ec59668ab9b438ab9b170306d29b01e9270fd5a`. Both models were rebuilt on CPU
with TensorFlow 2.15.1. Counts were calculated in two independent ways:

1. `tf.keras.Model.count_params()`;
2. the sum of the products of all model-weight shapes, separately for
   `trainable_weights` and `non_trainable_weights`.

The two methods agreed for both builds.

## Finding in brief

The stated Phase 2A count of **4,788,385 cannot be reproduced from the
recorded Phase 2A configuration or code**. The current files, the historical
Phase 2A commit, and a fresh rebuild all give:

- Phase 2A synthetic build: **4,766,008 total** =
  **4,743,384 trainable** + **22,624 non-trainable**;
- DRIVE Full build: **4,791,544 total** =
  **4,768,920 trainable** + **22,624 non-trainable**.

The reproducible DRIVE-minus-Phase-2A difference is therefore **25,536**, not
3,159. It is entirely trainable and is exactly localized to the spectral
weight tensors in SSTM stages 4 and 5.

The number 3,159 is only the arithmetic difference
`4,791,544 - 4,788,385`. No current or historical project source records
4,788,385, and no layer or documented configuration change between these two
builds produces a 3,159-parameter delta. The provenance of 4,788,385 remains
unresolved.

## Source configurations and build paths

### Phase 2A synthetic validation

Exact sources:

- configuration:
  `official_repo/configs/retinal.yaml` (especially lines 4–11);
- diagnostic entry point:
  `repro/diagnostics/phase2a_full_model_drive.py` (lines 14–37);
- historical Phase 2A result:
  `docs/phase2a_validation.md` (lines 14–20), also present at commit
  `f76d4251cb518ac577946e9bf7d7e682b0c9eb7c`;
- build path:
  `phase2a_full_model_drive.py` →
  `official_repo/train.py:load_config` →
  `official_repo/train.py:build_model` →
  `official_repo/s2mnet/models/s2mnet.py:S2MNet`.

The Phase 2A script passes `official_repo/configs/retinal.yaml` with no CLI
overrides. That YAML contains a 256 input size and one output class. Although
its header says it inherits from `default.yaml`, `train.load_config` merely
loads that one YAML file; it does not merge `default.yaml`
(`official_repo/train.py`, lines 75–98). The other architecture settings are
therefore supplied by the Python defaults in `train.build_model`
(`official_repo/train.py`, lines 153–175).

### DRIVE Full training

Exact sources:

- frozen configuration:
  `outputs/drive/full_seed42/config/frozen.yaml` (model section, lines 55–97);
- configuration adapter:
  `repro/experiments/drive_train.py:official_model_config`
  (lines 490–495);
- build call:
  `repro/experiments/drive_train.py:build_training_objects`
  (lines 498–522);
- recorded count:
  `outputs/drive/full_seed42/logs/train.log`, line 8;
- common released constructor:
  `official_repo/train.py:build_model` →
  `official_repo/s2mnet/models/s2mnet.py:S2MNet`.

The DRIVE trainer copies the frozen `model` mapping into the format expected
by the official builder, then calls the same `official_train.build_model`
used by Phase 2A. The three Full logs for seeds 42, 7, and 123 all record
4,791,544; seed does not affect model structure.

## Resolved architecture comparison

The following table shows every model-construction argument after applying
the Python defaults used by `official_repo/train.py`.

| Setting | Phase 2A | DRIVE Full | Parameter-count effect |
|---|---|---|---|
| Input size | 256 × 256 | 352 × 352 | **Different; causes the actual delta through SSTM `actual_k`** |
| Input channels | 3, fixed in `S2MNet` | 3, fixed in `S2MNet` | Same |
| Output classes | 1 | 1 | Same |
| Encoder filters | 24, 32, 64, 80, 128 | 24, 32, 64, 80, 128 | Same |
| MRF-SE | enabled at all 5 stages | enabled at all 5 stages | Same |
| MRF-SE kernels | 3, 5, 7 | 3, 5, 7 | Same |
| SE reduction | 16 | 16 | Same |
| Expansion ratio | 6 | 6 | Same |
| SSTM | enabled at all 5 stages | enabled at all 5 stages | Same |
| Requested SSTM K | 32 | 32 | Same requested value |
| SSTM spectral stages | all 5 | all 5 | Same |
| SSTM spatial/SSM stages | stages 3–5 | stages 3–5 | Same |
| `sstm_ssm_dim` | 16 | 16 | Same and unused by released forward math |
| SSTM dropout | 0.1 | 0.1 | Same; no parameters |
| BFP decoder | enabled, 4 stages | enabled, 4 stages | Same |
| BFP routing | soft | soft | Same |
| MRF-SE dropout | 0.1 | 0.1 | Same; no parameters |
| L2 regularizer | 0.0001 | 0.0001 | Same; no additional weights |
| Activation | ELU | ELU | Same; no parameters |

Thus, after resolution of defaults, **input size is the only constructor
difference**.

## Independently rebuilt counts

| Build | Input/output shape | Weight tensors | Trainable tensors | Non-trainable tensors | Trainable parameters | Non-trainable parameters | Total parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 2A | `(None, 256, 256, 3)` → `(None, 256, 256, 1)` | 409 | 305 | 104 | 4,743,384 | 22,624 | **4,766,008** |
| DRIVE Full | `(None, 352, 352, 3)` → `(None, 352, 352, 1)` | 409 | 305 | 104 | 4,768,920 | 22,624 | **4,791,544** |
| DRIVE − Phase 2A | — | 0 | 0 | 0 | **+25,536** | **0** | **+25,536** |

For Phase 2A, both the fresh count and the pre-existing audit source
`docs/phase2a_validation.md` agree on 4,766,008. For DRIVE, the fresh count
agrees with the count printed by each completed Full training log.

## Layer-by-layer comparison

Counts below are top-level Keras layer counts. Custom-layer rows include all
weights owned by their nested layers. Every row was also checked by summing
the shapes of its owned variables.

| Layer | Class | Phase 2A total | DRIVE Full total | DRIVE − Phase |
|---|---|---:|---:|---:|
| `input` | InputLayer | 0 | 0 | 0 |
| `stem_conv` | Conv2D | 448 | 448 | 0 |
| `stem_bn` | BatchNormalization | 64 | 64 | 0 |
| `stem_act` | Activation | 0 | 0 | 0 |
| `enc1_down` | Conv2D | 3,480 | 3,480 | 0 |
| `enc1_bn` | BatchNormalization | 96 | 96 | 0 |
| `enc1_act` | Activation | 0 | 0 | 0 |
| `mrfse_stage1` | MRF_SE_Block | 87,537 | 87,537 | 0 |
| `sstm_stage1` | SpectralSelectiveTokenMixer | 24,672 | 24,672 | 0 |
| `enc2_down` | Conv2D | 6,944 | 6,944 | 0 |
| `enc2_bn` | BatchNormalization | 128 | 128 | 0 |
| `enc2_act` | Activation | 0 | 0 | 0 |
| `mrfse_stage2` | MRF_SE_Block | 148,588 | 148,588 | 0 |
| `sstm_stage2` | SpectralSelectiveTokenMixer | 32,896 | 32,896 | 0 |
| `enc3_down` | Conv2D | 18,496 | 18,496 | 0 |
| `enc3_bn` | BatchNormalization | 256 | 256 | 0 |
| `enc3_act` | Activation | 0 | 0 | 0 |
| `mrfse_stage3` | MRF_SE_Block | 552,152 | 552,152 | 0 |
| `sstm_stage3` | SpectralSelectiveTokenMixer | 82,624 | 82,624 | 0 |
| `enc4_down` | Conv2D | 46,160 | 46,160 | 0 |
| `enc4_bn` | BatchNormalization | 320 | 320 | 0 |
| `enc4_act` | Activation | 0 | 0 | 0 |
| `mrfse_stage4` | MRF_SE_Block | 849,550 | 849,550 | 0 |
| `sstm_stage4` | SpectralSelectiveTokenMixer | 46,960 | 65,200 | **+18,240** |
| `enc5_down` | Conv2D | 92,288 | 92,288 | 0 |
| `enc5_bn` | BatchNormalization | 512 | 512 | 0 |
| `enc5_act` | Activation | 0 | 0 | 0 |
| `mrfse_stage5` | MRF_SE_Block | 2,124,208 | 2,124,208 | 0 |
| `sstm_stage5` | SpectralSelectiveTokenMixer | 75,136 | 82,432 | **+7,296** |
| `bfp_stage1` | BFP_DecoderStage | 301,841 | 301,841 | 0 |
| `bfp_stage2` | BFP_DecoderStage | 180,545 | 180,545 | 0 |
| `bfp_stage3` | BFP_DecoderStage | 52,385 | 52,385 | 0 |
| `bfp_stage4` | BFP_DecoderStage | 26,137 | 26,137 | 0 |
| `head_up` | UpSampling2D | 0 | 0 | 0 |
| `head_conv1` | Conv2D | 6,944 | 6,944 | 0 |
| `head_conv2` | Conv2D | 4,624 | 4,624 | 0 |
| `output` | Conv2D | 17 | 17 | 0 |
| **Total** | — | **4,766,008** | **4,791,544** | **+25,536** |

## Exact cause

`SpectralSelectiveTokenMixer.build` sets

```text
actual_k = min(requested_k, feature_height, feature_width)
freq_weights.shape = (actual_k, actual_k, channels)
```

Source:
`official_repo/s2mnet/models/blocks.py`, lines 79–93.

Five stride-2 encoder stages give these feature and spectral-filter shapes:

| Stage | Channels | Phase 2A feature size / `actual_k` | DRIVE feature size / `actual_k` | Phase spectral weights | DRIVE spectral weights | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 128 / 32 | 176 / 32 | 24,576 | 24,576 | 0 |
| 2 | 32 | 64 / 32 | 88 / 32 | 32,768 | 32,768 | 0 |
| 3 | 64 | 32 / 32 | 44 / 32 | 65,536 | 65,536 | 0 |
| 4 | 80 | 16 / 16 | 22 / 22 | 20,480 | 38,720 | **+18,240** |
| 5 | 128 | 8 / 8 | 11 / 11 | 8,192 | 15,488 | **+7,296** |

The exact arithmetic is:

```text
stage 4: (22² - 16²) × 80  = 18,240
stage 5: (11² -  8²) × 128 =  7,296
total:                            25,536
```

All other nested weights within `sstm_stage4` and `sstm_stage5` have identical
shapes. In particular, the Dense projections, gates, fusion layers, and layer
normalizations are unchanged. The two `freq_weights` tensors are trainable,
which is why the complete delta appears in the trainable partition.

This resolution-dependent weight allocation is unusual: input spatial
resolution normally changes activation sizes but not convolutional model
parameters. Here it changes parameter count because the released SSTM
creates a learnable frequency grid whose shape is clamped to the feature-map
size at build time.

## Excluded explanations

- **Input channels:** not the cause. `S2MNet` fixes the input to three channels
  in `official_repo/s2mnet/models/s2mnet.py`, line 84, and both builds are RGB.
- **Output classes:** not the cause. Both use one output channel and the same
  17-parameter `output` convolution.
- **Normalization state:** not the cause. Both builds contain the same 104
  non-trainable tensors and 22,624 non-trainable parameters. No training was
  performed. Batch-normalization moving values can change during training,
  but their number and shape do not.
- **Model variant:** not the cause. Both are Full models with all five MRF-SE
  and SSTM stages and all four soft-routing BFP stages enabled.
- **Configuration defaults:** not the cause after resolution. Phase 2A relies
  on Python defaults while DRIVE records them explicitly, but their effective
  values are identical.
- **Seed, loss, or optimizer:** not the cause. Seed does not change model
  topology. The reported counts are model-only counts; neither the five MAL
  scalars nor Adam state is included.
- **`sstm_ssm_dim`:** not the cause. It is 16 in both configurations and is
  stored but unused in the released SSTM forward math
  (`official_repo/s2mnet/models/blocks.py`, lines 45–46 and 58–74).

The cause is specifically **input resolution interacting with the
shape-dependent SSTM spectral-filter allocation**.

## Canonical count for Claim 1

The paper's method section describes five feature resolutions
`{176, 88, 44, 22, 11}` and channels `{24, 32, 64, 80, 128}`. Those feature
resolutions arise from the released five-stage encoder at a 352 × 352 input,
not at 256 × 256. See the primary paper:
[arXiv:2601.01285, Section 3.1](https://arxiv.org/abs/2601.01285).

The same choice is independently supported by:

- the `S2MNet` constructor default and example, which use `input_size=352`
  (`official_repo/s2mnet/models/s2mnet.py`, lines 13 and 27–30);
- `official_repo/configs/default.yaml`, line 11;
- the released full-model example in `official_repo/README.md`, lines 519–533;
- the DRIVE Full frozen configuration, whose effective architecture matches
  all other released defaults.

Therefore, the build that best corresponds to the paper's stated stage
resolutions is the **352 × 352 RGB, one-class released Full build**. The
canonical exact count for the released implementation should be:

> **4,791,544 total parameters: 4,768,920 trainable and 22,624
> non-trainable.**

The 256 × 256 Phase 2A build is legitimate for the retinal synthetic
diagnostic, but it is not the paper's reference-resolution build. Its exact
count should remain recorded as 4,766,008.

## Relationship to the paper's “4.7M”

In decimal millions:

| Count | Decimal millions | Conventional rounding to one decimal |
|---|---:|---:|
| Rebuilt Phase 2A | 4.766008M | 4.8M |
| Supplied but unreproduced Phase 2A value | 4.788385M | 4.8M |
| Canonical released 352 build | 4.791544M | 4.8M |

Consequently, neither the supplied pair nor the reproducible pair
**conventionally rounds to 4.7M** at one decimal place; each rounds to 4.8M.
They can only be described as “about 4.7M” if 4.7M is treated as a coarse
approximation or truncation. The exact canonical released-code count should
be preserved alongside that paper wording.

The released README also gives an approximate component table labelled
“Table 10” and totalling 4.70M (`official_repo/README.md`, lines 63–81). A
direct grouping of the released 352 build gives
4,219,051 encoder parameters, 560,908 BFP-decoder parameters, and 11,585 head
parameters, so that approximate component allocation is not an exact
layer-level accounting of the released model.

## Unresolved ambiguity

1. **Origin of 4,788,385:** no occurrence was found in the current workspace,
   the Phase 2A historical commit, or the pinned official repository history.
   It is 22,377 above the reproducible Phase 2A count, but that excess does
   not correspond to a model layer, the loss's five learned scalars,
   optimizer state, or normalization state.
2. **Paper equations versus released SSTM:** the paper describes a
   bottlenecked spatial path with `d=16`, while the released code stores
   `ssm_state_dim=16` but projects channel dimension C directly to C. It also
   has other previously documented spectral implementation differences in
   `REPRODUCTION_PLAN.md`, lines 80–90. Thus 4,791,544 is the canonical exact
   count for the **released implementation at the paper's reference
   resolution**, not proof that every paper-described operation has been
   instantiated exactly.
3. **Resolution dependence:** the released implementation does not have one
   resolution-independent parameter count. Any exact Claim 1 count must state
   the input resolution and output-class convention.

## Five-stage encoder and channel-dimension verification

### Scope and evidence chain

This section verifies only the encoder-stage count, feature resolutions,
channel dimensions, and stage-local modules in the canonical released-code
Full build. It does not evaluate the parameter-efficiency comparisons with
TransUNet or Swin-Unet and does not assign a verdict to Claim 1 as a whole.

The canonical configuration-to-model path is:

1. `outputs/drive/full_seed42/config/frozen.yaml`, lines 55–97, records
   `input_size: 352`, `filters: [24, 32, 64, 80, 128]`, MRF-SE at all stages,
   and SSTM at all stages.
2. `repro/experiments/drive_train.py:official_model_config`, lines 490–495,
   copies that model mapping without changing its architecture values.
3. `repro/experiments/drive_train.py:build_training_objects`, lines 498–501,
   calls `official_train.build_model`.
4. `official_repo/train.py:build_model`, lines 153–175, forwards the filter
   sequence and the MRF-SE/SSTM settings to the released `S2MNet`
   constructor.
5. `official_repo/s2mnet/models/s2mnet.py:S2MNet`, lines 81–129, asserts that
   there are exactly five filter entries, creates a separate 16-channel
   full-resolution stem, and iterates once over each filter value to
   instantiate the five encoder stages.

The paper comparison uses the retrievable
[arXiv v1 HTML](https://arxiv.org/html/2601.01285v1), because the current v2
is a withdrawal stub without the paper body. Section 3.1 of v1 states five
stages, resolutions `{176, 88, 44, 22, 11}`, channels
`{24, 32, 64, 80, 128}`, and MRF-SE followed by SSTM at each stage. Section
4.2 repeats the channel sequence and the MRF-SE/SSTM hyperparameters. The
released README's Table 9 independently records the same stem and five-stage
layout (`official_repo/README.md`, lines 47–61).

### Instantiated encoder

The 352 × 352 RGB input first passes through `stem_conv` (3 × 3, stride 1,
16 filters), `stem_bn`, and ELU, so the stem output is
`352 × 352 × 16`. The stem is not counted as one of the five encoder stages.
At each encoder iteration, the released model applies:

1. a 3 × 3 stride-2 convolution, batch normalization, and ELU;
2. one residual MRF-SE block; and
3. one residual SSTM layer.

MRF-SE preserves the stage's spatial and channel dimensions. Its released
implementation expands channels by six, applies parallel depthwise 3 × 3,
5 × 5, and 7 × 7 branches, concatenates and fuses them, applies SE with
reduction 16, projects back to the stage channel count, and adds the residual
(`official_repo/s2mnet/models/blocks.py`, lines 201–212 and 249–312).

SSTM also preserves the stage shape. The frozen configuration enables its
spectral path at all five stages but enables its selective/SSM path only at
stages 3–5 (`outputs/drive/full_seed42/config/frozen.yaml`, lines 71–92).
The requested spectral size is `K=32`; the released layer clamps this to
`min(32, H, W)` at build time
(`official_repo/s2mnet/models/blocks.py`, lines 79–105).

| Component | Output resolution for 352 input | Output channels | Applied released-code modules |
|---|---:|---:|---|
| Stem (not an encoder stage) | 352 × 352 | 16 | 3 × 3 stride-1 convolution → BN → ELU |
| Encoder stage 1 | 176 × 176 | **24** | 3 × 3 stride-2 convolution → BN → ELU; MRF-SE (`r=6`, kernels 3/5/7, SE reduction 16); SSTM spectral path only (`K_actual=32`, selective/SSM off) |
| Encoder stage 2 | 88 × 88 | **32** | 3 × 3 stride-2 convolution → BN → ELU; MRF-SE (`r=6`, kernels 3/5/7, SE reduction 16); SSTM spectral path only (`K_actual=32`, selective/SSM off) |
| Encoder stage 3 | 44 × 44 | **64** | 3 × 3 stride-2 convolution → BN → ELU; MRF-SE (`r=6`, kernels 3/5/7, SE reduction 16); dual-path SSTM (`K_actual=32`, spectral and selective/SSM on) |
| Encoder stage 4 | 22 × 22 | **80** | 3 × 3 stride-2 convolution → BN → ELU; MRF-SE (`r=6`, kernels 3/5/7, SE reduction 16); dual-path SSTM (`K_actual=22`, spectral and selective/SSM on) |
| Encoder stage 5 / bridge | 11 × 11 | **128** | 3 × 3 stride-2 convolution → BN → ELU; MRF-SE (`r=6`, kernels 3/5/7, SE reduction 16); dual-path SSTM (`K_actual=11`, spectral and selective/SSM on) |

Therefore, read in stage order after the separate stem, the instantiated
encoder channel sequence is exactly:

> **{24, 32, 64, 80, 128}.**

### Runtime tensor-shape confirmation

One diagnostic forward pass was run on CPU with a zero tensor of shape
`(1, 352, 352, 3)` and `training=False`. It rebuilt the model from
`outputs/drive/full_seed42/config/frozen.yaml` through
`drive_train.official_model_config` and `official_train.build_model`. It did
not compile or train the model and did not read any image data.

The Keras symbolic shapes and actual runtime shapes agreed at every probe:

| Probe | Symbolic shape | Observed runtime shape |
|---|---|---|
| `stem_act` | `(None, 352, 352, 16)` | `(1, 352, 352, 16)` |
| `enc1_act` → `mrfse_stage1` → `sstm_stage1` | `(None, 176, 176, 24)` at all three points | `(1, 176, 176, 24)` at all three points |
| `enc2_act` → `mrfse_stage2` → `sstm_stage2` | `(None, 88, 88, 32)` at all three points | `(1, 88, 88, 32)` at all three points |
| `enc3_act` → `mrfse_stage3` → `sstm_stage3` | `(None, 44, 44, 64)` at all three points | `(1, 44, 44, 64)` at all three points |
| `enc4_act` → `mrfse_stage4` → `sstm_stage4` | `(None, 22, 22, 80)` at all three points | `(1, 22, 22, 80)` at all three points |
| `enc5_act` → `mrfse_stage5` → `sstm_stage5` | `(None, 11, 11, 128)` at all three points | `(1, 11, 11, 128)` at all three points |

The probe also returned 4,791,544 parameters, providing a cross-check that
the inspected tensors belong to the already established canonical 352 × 352
released-code build.

### Paper-versus-code comparison

| Encoder property | Paper/released documentation | Released canonical Full code | Assessment |
|---|---|---|---|
| Number of stages | Paper Sections 3.1 and 4.2: five; Figure 2 caption: five | Constructor requires five filters and executes five encoder-loop iterations | Match |
| Resolutions | Paper Section 3.1: `{176, 88, 44, 22, 11}`; README Table 9 gives the same stride-2 progression | Runtime: 176, 88, 44, 22, 11 | Match at 352 × 352 input |
| Channels | Paper Sections 3.1 and 4.2: `{24, 32, 64, 80, 128}`; README Table 9 agrees | Runtime: 24, 32, 64, 80, 128 | Exact match |
| Separate stem | README Table 9: 352², 16 filters | 16-channel, full-resolution convolution + BN + ELU before stage 1 | Match; the stem is separate from the five stages |
| MRF-SE placement | Paper says every encoder stage; Figure 2 and README Table 9 show it in every stage | `mrfse_stage1` through `mrfse_stage5` are instantiated | Match |
| MRF-SE internals | Paper Sections 3.2/4.2: `r=6`, kernels 3/5/7, SE reduction 16, residual structure | Same effective settings and shape-preserving residual implementation | Match at the structural/configuration level |
| SSTM placement | Paper says every encoder stage; Figure 2 and README Table 9 show it in every stage | `sstm_stage1` through `sstm_stage5` are instantiated | Match at the layer-placement level |
| SSTM branch use | Paper describes SSTM generally as a dual-branch spectral and content-gated spatial module | Stages 1–2 are spectral-only; only stages 3–5 instantiate the selective/SSM path and fusion layer | Qualification / implementation discrepancy |
| SSTM `d=16` | Paper Sections 3.3/4.2 describe a spatial bottleneck dimension `d=16` | Configuration passes 16 and the layer stores it, but released Dense layers project C→C; the source explicitly says `ssm_state_dim` is unused | Implementation discrepancy; no effect on stage shapes |
| Stage-adaptive `K` | Paper discusses `K=32` and stage-adaptive truncation | Requested `K=32`, clamped to 32, 32, 32, 22, and 11 across stages | Structurally consistent with stage adaptation, though the report's earlier SSTM equation/code ambiguity remains |

The diagram itself needs care. The released architecture image
`official_repo/newly_archi.pdf`, embedded and captioned as Figure 2 in
`official_repo/README.md` lines 31–35, shows five downsampled resolutions and
MRF-SE/SSTM blocks. However, its fourth 22 × 22 encoder panel is labelled
“Encoder Stage 5,” followed by a separate 11 × 11 “Encoder Stage 5 (Bridge).”
The 22 × 22 panel should be stage 4. The diagram also does not print channel
counts. The paper prose, README Table 9, constructor sequence, layer names,
and runtime tensors consistently resolve this as stage 4 with 80 channels
followed by stage 5 with 128 channels.

There is also imprecise README wording at lines 37–41: after saying “each
encoder stage integrates,” it lists BFD and MASL alongside MRF-SE and SSTM.
The released model and README Table 9 place BFD in the decoder, while MASL is
the training objective; neither is an encoder-stage module. This wording does
not change the instantiated five-stage encoder.

The previously identified SSTM differences remain important but do not alter
the verified stage count, spatial resolutions, or output channel dimensions:

- `ssm_state_dim=16` is unused in the released forward computation
  (`official_repo/s2mnet/models/blocks.py`, lines 45–46 and 58–105);
- stages 1–2 omit the selective/SSM path despite the paper's generic
  dual-branch SSTM description; and
- the released spectral forward implementation differs from the paper's
  centered crop/pad equations, as already recorded in
  `REPRODUCTION_PLAN.md`, lines 80–90, and in this report's unresolved
  ambiguity section.

### Conservative component verdict

**Supported for this component only.** For the canonical 352 × 352 released
Full configuration, S2M-Net has exactly five post-stem encoder stages with
output channels **{24, 32, 64, 80, 128}** and resolutions
**{176, 88, 44, 22, 11}**. Each stage instantiates both an MRF-SE block and an
SSTM layer. This agrees with the paper's encoder-stage count and channel
sequence, subject to the documented qualifications that SSTM is
spectral-only at stages 1–2, `d=16` is unused, and the architecture figure
contains a stage-label typo. This is not a verdict on the complete Claim 1
and does not evaluate either parameter-efficiency baseline.

## TransUNet parameter-comparison verification

### Scope and result in brief

This section investigates only the statement that S2M-Net has roughly 13
times fewer parameters than TransUNet. It does not investigate Swin-Unet or
assign a verdict to Claim 1 as a whole.

The exact TransUNet variant behind the S2M-Net paper's comparison **cannot be
identified from the paper or released configuration files**:

- the [S2M-Net paper v1](https://arxiv.org/html/2601.01285v1) states “13×
  fewer than TransUNet” in the introduction and “4.7M parameters ... 13×
  fewer than TransUNet” in related work, but does not give a TransUNet
  parameter total, backbone name, head configuration, or baseline
  configuration;
- the paper cites Chen et al.'s original TransUNet work, but the citation
  alone does not select a model variant;
- the released S2M-Net README supplies **60.0M** and **12.8×**, but still does
  not name a matching TransUNet variant or configuration
  (`official_repo/README.md`, lines 25, 182–212, and 350–364); and
- no released S2M-Net YAML, baseline-training script, or invocation was found
  that constructs a 60.0M TransUNet. The only local TransUNet constructor is
  a much smaller TensorFlow approximation in
  `official_repo/s2mnet/models/baselines.py`, lines 95–181.

Thus **60.0M is the comparison count documented by the released S2M-Net
materials**, but it is a one-decimal reported value with no recoverable
architecture/configuration provenance. It must not be presented as the
parameter count of a specifically identified original TransUNet variant.

### Source trace

#### S2M-Net sources

The source chain for the comparison is:

1. [S2M-Net paper v1, introduction and related
   work](https://arxiv.org/html/2601.01285v1): states 4.7M and the approximate
   13× comparison, and cites Chen et al. (2021).
2. `official_repo/README.md`, line 25: expands the comparison to “12.8× fewer
   parameters (4.7M vs. 60M).”
3. `official_repo/README.md`, lines 182–212: Table 1 records
   **TransUNet 60.0M** and S2M-Net 4.7M.
4. `official_repo/README.md`, lines 350–364: Table 18 again records
   **TransUNet 60.0M**, S2M-Net 4.7M, and **12.8× fewer**.

The paper body itself does not contain the README's Table 18 or an explicit
60.0M count. Its related-work statement gives only a broad 60–105M range for
several transformer methods. Therefore, the distinction is important:

> **Paper body:** 4.7M and “13× fewer”; no explicit TransUNet count.  
> **Released S2M-Net README:** TransUNet 60.0M and “12.8× fewer.”

The paper's cited source is:

- J. Chen et al.,
  [“TransUNet: Transformers Make Strong Encoders for Medical Image
  Segmentation”](https://arxiv.org/pdf/2102.04306);
- authors' official repository:
  [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet).

#### Original TransUNet sources

The authoritative code audit was pinned to official TransUNet commit
`26de0c4d9a5145589ea249d169af7f7130823e03` from 8 February 2021, the paper's
release date and the repository's last model-code update that day. Its
`networks/` directory is byte-for-byte unchanged at current repository commit
`02ef0010b36eb8328b5e689eadaf613602edf9b8`; subsequent relevant commits only
changed documentation.

Exact authoritative paths:

- model configuration:
  [`networks/vit_seg_configs.py`](https://github.com/Beckschen/TransUNet/blob/26de0c4d9a5145589ea249d169af7f7130823e03/networks/vit_seg_configs.py),
  especially `get_b16_config` and `get_r50_b16_config`;
- Transformer, decoder, and segmentation head:
  [`networks/vit_seg_modeling.py`](https://github.com/Beckschen/TransUNet/blob/26de0c4d9a5145589ea249d169af7f7130823e03/networks/vit_seg_modeling.py);
- hybrid ResNetV2 backbone:
  [`networks/vit_seg_modeling_resnet_skip.py`](https://github.com/Beckschen/TransUNet/blob/26de0c4d9a5145589ea249d169af7f7130823e03/networks/vit_seg_modeling_resnet_skip.py);
- executable default selection:
  [`train.py`](https://github.com/Beckschen/TransUNet/blob/26de0c4d9a5145589ea249d169af7f7130823e03/train.py).

The original paper identifies TransUNet as a hybrid ResNet-50/ViT Base
encoder with CUP and three skip connections. It specifies a default 224 ×
224 input, patch size 16, 12 Transformer layers, hidden size 768, MLP size
3072, and 12 heads, and says the Base model is used for its experiments
(original paper, Sections 4.2 and 4.4). The official README's example command
and `train.py` select `R50-ViT-B_16`.

Neither the original TransUNet paper nor its official README reports a
scalar model parameter count. Therefore there is no authoritative
**reported** count to equate directly with S2M-Net's 60.0M. The authoritative
source provides an architecture from which a count can be reproduced.

### Authoritative configuration and count-affecting details

The original authors' default Synapse construction resolves as follows:

| Setting | Authoritative default |
|---|---|
| Variant | `R50-ViT-B_16`, Base TransUNet |
| Input resolution | 224 × 224 |
| Effective input channels | 3; one-channel tensors are repeated to 3 in `forward`, and the ResNet root is fixed at 3 channels |
| CNN backbone | Pre-activation ResNetV2, width 64, block units `(3, 4, 9)` |
| ViT | hidden size 768, 12 layers, 12 heads, MLP dimension 3072 |
| Hybrid patch embedding | 1 × 1 projection from 1024 to 768 channels at a 14 × 14 token grid |
| Skip connections | 3; skip channels `[512, 256, 64, 16]` |
| CUP decoder channels | `[256, 128, 64, 16]` |
| Output classes | 9 for the official Synapse command; the base config's placeholder value 2 is overwritten by `train.py` |
| Segmentation head | 3 × 3 convolution with bias, 16 input channels and `n_classes` outputs |

Two implementation details matter when comparing configurations:

- `VisionTransformer(..., num_classes=...)` stores that argument, but the
  segmentation head reads `config.n_classes`. The official training path
  explicitly sets `config.n_classes` before construction.
- The learned position embedding has
  `(input_size / 16)² × 768` parameters. Consequently, the official
  implementation's exact count changes with input resolution. Output class
  count changes the final head by 145 parameters per class.

No S2M-Net baseline configuration states how `R50-ViT-B_16`, input
resolution, input handling, output classes, or the head were adapted for the
paper's many binary and multiclass datasets. This prevents exact
configuration matching to the claimed 60.0M.

### Independent reproduction from authoritative code

No model was trained and no pretrained weights were downloaded. Counts were
reproduced from the pinned official constructors in two independent
programmatic ways:

1. a tensor-by-tensor ledger calculated from every `Conv2d`, `Linear`,
   normalization, learned position embedding, and segmentation-head shape in
   the authoritative source; and
2. direct execution of the pinned constructors using in-memory,
   shape-only PyTorch compatibility classes, followed by the equivalent of
   summing `p.numel()` over the instantiated parameter objects.

The two methods agreed exactly. The component ledger for the official
224 × 224, nine-class Synapse configuration is:

| Authoritative component | Reproduced parameters |
|---|---:|
| Hybrid ResNetV2 `(3, 4, 9)` | 11,894,848 |
| 1024→768 hybrid patch embedding | 787,200 |
| 14 × 14 × 768 learned position embedding | 150,528 |
| Twelve Base Transformer blocks | 85,054,464 |
| Final Transformer layer normalization | 1,536 |
| CUP decoder with three skips | 7,387,200 |
| 16→9, 3 × 3 segmentation head with bias | 1,305 |
| **Total** | **105,277,081** |

All model parameters are trainable in the constructor; batch-normalization
running statistics are buffers rather than parameters in PyTorch. The exact
reproduced variants are:

| Authoritative-code configuration | Reproduced total | Ratio to S2M-Net 4,791,544 |
|---|---:|---:|
| 224 × 224, 1 output class | 105,275,921 | 21.9711894538× |
| 224 × 224, 2 output classes | 105,276,066 | 21.9712197154× |
| 224 × 224, 9 output classes — official Synapse default | **105,277,081** | **21.9714315469×** |
| 352 × 352, 1 output class — resolution/class sensitivity only | 105,497,105 | 22.0173507746× |
| 352 × 352, 9 output classes — sensitivity only | 105,498,265 | 22.0175928678× |

The 352 × 352 rows are not silently substituted as the S2M-Net paper's
baseline. They show only how the authoritative implementation changes when
adapted to the S2M-Net reference resolution. None reproduces 60.0M.

### The bundled S2M-Net TransUNet approximation

`official_repo/s2mnet/models/baselines.py:TransUNet`, lines 125–181, is not
the original authors' R50–ViT-B/16 TransUNet. It instead uses:

- an RGB input and a small CNN with channels `[64, 128, 256]`;
- hidden dimension 256, eight attention heads, and only two Transformer
  layers by default;
- no ResNet-50/ResNetV2 hybrid backbone; and
- a local three-stage transposed-convolution decoder and 1 × 1 head.

Fresh TensorFlow construction gives 5,437,825 Keras-tracked parameters for
one output class and 5,438,345 for nine classes, independent of input
resolution. Its position embedding is created as a raw `tf.Variable` and
Keras warns that it is not tracked. At 352 × 352, manually including that
`1 × 1936 × 256` tensor would give 5,933,441 parameters for one output class,
still nowhere near 60.0M.

No released call site or frozen configuration was found for this local
constructor. It therefore neither identifies the S2M-Net paper's exact
baseline nor reproduces its reported 60.0M.

### Programmatic ratio and numerical assessment

Using the comparison count from the released S2M-Net materials and the
canonical exact S2M-Net count:

```text
60,000,000 / 4,791,544 = 12.52205969516297878095244456
```

Using the two one-decimal display values in the released README:

```text
60.0 / 4.7 = 12.76595744680851063829787234
```

The latter rounds to the README's 12.8×; the exact-denominator ratio rounds
to 12.5× at one decimal or 13× at zero decimals. An exact 13× ratio would
require 62,290,072 TransUNet parameters, so 60.0M is 3.6765% below that
target. Accordingly, **“roughly 13×” is numerically reasonable if the
unverified 60.0M baseline count is accepted**.

The independently reproduced official Base R50–ViT-B/16 count must not be
used to retrofit the paper's arithmetic. Its approximately 22× ratio shows
that 60.0M does not describe the original authors' default implementation,
not that the S2M-Net paper secretly used a 105.3M baseline.

### Configuration-matching limitations

1. The S2M-Net paper does not name a TransUNet backbone/variant or give an
   exact TransUNet count.
2. The released README gives only 60.0M to one decimal and no construction
   path, configuration, checkpoint, parameter summary, or count script.
3. The cited original paper uses Base R50–ViT with CUP and three skips, while
   the official code names the executable variant `R50-ViT-B_16`; its
   reproduced count is about 105.3M, not 60.0M.
4. The S2M-Net repository's bundled TensorFlow `TransUNet` is an
   architecture-level approximation rather than the cited implementation,
   and its count is approximately 5.4–5.9M depending on whether its untracked
   position tensor is included.
5. Input resolution and output classes affect the authoritative count, while
   the paper's cross-dataset comparison does not record either baseline
   setting. The authoritative model's effective convolutional input remains
   three channels because grayscale inputs are repeated.
6. The independent count is a constructor/parameter-shape reproduction, not
   a training or checkpoint reproduction. PyTorch was not installed in the
   project environment; the official constructors were executed with
   shape-only compatibility classes and independently checked by explicit
   tensor-shape arithmetic.

### Conservative TransUNet-component verdict

**Conditionally supported numerically, but not configuration-verified.** The
released S2M-Net value of 60.0M divided by the canonical exact S2M-Net count
is **12.5220596952×**, which reasonably rounds to “roughly 13×.” However, the
S2M-Net paper does not identify the corresponding TransUNet variant, the
released repository provides no 60.0M construction, the bundled
approximation does not match it, and the cited authors' default Base
R50–ViT-B/16 implementation independently reproduces approximately 105.3M
parameters instead. The arithmetic wording is plausible, but the baseline
identity and 60.0M denominator remain unverified. This is a verdict only on
the TransUNet parameter-comparison component, not on Claim 1 as a whole.

## Swin-UNet parameter-comparison verification

### Scope and result in brief

This section investigates only the statement that S2M-Net has roughly six
times fewer parameters than Swin-Unet. It does not alter or revisit the
TransUNet investigation above and does not assign a verdict to Claim 1 as a
whole.

The exact Swin-Unet baseline used by the S2M-Net paper **cannot be identified
from a citation, configuration, or executable baseline path**:

- the [S2M-Net paper v1](https://arxiv.org/html/2601.01285v1) says “6× fewer
  than Swin-Unet” but gives no Swin-Unet parameter count, variant,
  configuration, or citation on either occurrence of that claim;
- the S2M-Net arXiv v1 source archive contains no bibliography entry for Hu
  Cao et al.'s 2D Swin-Unet paper (arXiv:2105.05537). Its Swin-related
  citation is instead Tang et al.'s **Swin-UNETR**, a different 3D
  architecture;
- the released S2M-Net README reports **27.0M** for “Swin-Unet,” but supplies
  no source, configuration, checkpoint, model summary, or counting script;
  and
- the released S2M-Net code contains no Swin-Unet constructor, configuration,
  or training path.

There is nevertheless a strong, but inferential, numerical match to the
original Swin-Unet authors' released Tiny/lite configuration. A source-level
reproduction of that paper-era official configuration gives **27,168,900
parameters** for the nine-class Synapse model. The released S2M-Net value of
27.0M differs by 168,900 parameters, or 0.6256% of 27.0M. This makes the
official Tiny/lite model a plausible source for the rounded baseline value,
but the missing S2M-Net citation and configuration prevent treating that
identity as verified.

### S2M-Net source and citation trace

The complete recoverable evidence chain in the S2M-Net materials is:

1. The S2M-Net arXiv v1 source,
   `sec/1_intro.tex`, line 64, states “4.7M” and “6× fewer than Swin-Unet.”
   Lines 123–124 repeat the six-times statement. Neither occurrence has a
   Swin-Unet citation.
2. The paper's results table (`sec/1_intro.tex`, lines 486–555; rendered as
   Table 1) contains a `SwinUNet` performance column but no parameter row and
   no configuration note for that model.
3. In the same paper source, `sec/1_intro.tex`, lines 95–99, cites
   `tang2022swinunetr` for **Swin-UNETR**. `main.bib`, lines 175–181, resolves
   this to Tang et al., “Self-Supervised Pre-Training of Swin Transformers for
   3D Medical Image Analysis.” `main.bib` also contains a duplicate Tang entry
   at lines 630–636. It contains no Cao, Swin-Unet, or arXiv:2105.05537 entry.
   Therefore, the paper's only Swin segmentation citation cannot establish the
   identity of its 2D “Swin-Unet” comparison.
4. `official_repo/README.md`, lines 182–212, reports `Swin-Unet` performance
   across the datasets and gives **27.0M** in the “Model Params” row.
5. `official_repo/README.md`, lines 350–363, reports **27.0M** again in its
   efficiency table, which says that all methods were evaluated at 352 × 352.
   Lines 378–388 also list Swin-Unet training time, convergence epoch, memory,
   and Dice, but add no architecture provenance. Unlike the TransUNet row,
   the README does not print a separate Swin-Unet parameter ratio.
6. The README's other Swin-Unet occurrences are results-only: Table 2 at
   lines 214–221, the EndoVis/BraTS/ACDC per-class tables at lines 225–256,
   and the generic “transformer-based models (>27M parameters)” impact
   statement at line 765. None identifies a variant, configuration, count
   method, citation, or implementation.
7. A repository-wide source search finds “Swin-Unet” only in README tables
   and prose. `official_repo/s2mnet/models/__init__.py`, lines 16–32, exports
   U-Net, U-Net++, TransUNet, and UMamba baselines, but no Swin-Unet.
   `REPRODUCTION_PLAN.md`, lines 90–92, independently records that the bundled
   baselines omit Swin-Unet.

The required distinction is therefore:

> **S2M-Net paper body:** “6× fewer”; no explicit baseline count or matching
> citation.  
> **Released S2M-Net README:** Swin-Unet **27.0M**; no executable provenance.  
> **Only Swin-related paper citation:** Tang et al.'s 3D Swin-UNETR, not Cao
> et al.'s 2D Swin-Unet.

### Original Swin-Unet sources

The candidate authoritative architecture is Hu Cao et al.,
[“Swin-Unet: Unet-like Pure Transformer for Medical Image
Segmentation”](https://arxiv.org/pdf/2105.05537), with the authors'
[official repository](https://github.com/HuCaoFighting/Swin-Unet).

The code audit was pinned to paper-era official commit
[`1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9`](https://github.com/HuCaoFighting/Swin-Unet/commit/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9)
from 29 June 2021. The instantiated model path and exact upstream paths are:

1. [`README.md`](https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/README.md)
   directs the user to `train.sh` for the Synapse run.
2. [`train.sh`](https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/train.sh)
   identifies `configs/swin_tiny_patch4_window7_224_lite.yaml` as the intended
   configuration. The paper-era shell file has spacing/typing defects, but
   the current official README prints the equivalent Python invocation
   explicitly.
3. [`config.py`](https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/config.py)
   supplies defaults not overridden by that YAML, including 224 input,
   three model input channels, patch size 4, MLP ratio 4, QKV bias, no
   absolute position embedding, and patch normalization.
4. [`train.py`](https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/train.py),
   lines 78–95, selects nine output classes for Synapse and constructs
   `SwinUnet`.
5. [`networks/vision_transformer.py`](https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/networks/vision_transformer.py),
   lines 23–50, forwards the merged encoder configuration to
   `SwinTransformerSys`. Its forward method repeats a one-channel medical
   image three times, so the instantiated patch embedding remains
   three-channel.
6. [`networks/swin_transformer_unet_skip_expand_decoder_sys.py`](https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/networks/swin_transformer_unet_skip_expand_decoder_sys.py),
   lines 558–742, constructs the patch embedding, four encoder/bottleneck
   stages, decoder and skip projections, final ×4 expansion, and bias-free
   1 × 1 segmentation head.

The current official repository head was also inspected at commit
`f48f623e226e25b6e395c37207915c50aaa9c776`. For the same explicit
architecture arguments, its instantiated topology is unchanged: the main
model file adds an unused `MoEFFNGating` class and formatting/comment fixes,
but no call instantiates that class. Current `train.py` has drifted to a
four-class generic-dataset default, so the paper-era nine-class Synapse path
is the appropriate authoritative configuration for reproducing the original
release rather than the current command-line default.

The [original paper](https://arxiv.org/pdf/2105.05537) confirms a 4 × 4 patch
size and an RGB-sized 48-value patch before embedding (Section 3.1), a
224 × 224 input for its experiments (Section 4.2), and adoption of the
Tiny-based rather than Base model (Section 4.5). It does **not** report a
scalar parameter count. The original repository README likewise does not
report a count; it provides the executable Tiny/lite configuration from which
one can be reproduced.

### Effective authoritative configuration

The paper-era official command resolves to:

| Count-affecting setting | Effective value and source |
|---|---|
| Named variant | `swin_tiny_patch4_window7_224_lite`; paper-era `train.sh`, current official README command, and YAML |
| Input resolution | 224 × 224; `config.py` and original paper Sections 4.2/4.5 |
| Patch size | 4 × 4; `config.py` and original paper Sections 3.1/4.2 |
| Model input channels | **3**; `config.py`. A one-channel tensor is repeated to RGB before the model |
| Embed dimension | **96**; lite YAML |
| Encoder depths | **[2, 2, 2, 2]**; lite YAML overrides the `[2, 2, 6, 2]` Python default |
| Configured decoder depths | YAML says **[2, 2, 2, 1]**, but this value is not forwarded by `vision_transformer.py` |
| Instantiated decoder depth | Initial PatchExpand with no Swin block, then **[2, 2, 2]** blocks at 384, 192, and 96 channels; constructor indexes the reversed encoder `depths` and never reads `depths_decoder` |
| Attention heads | **[3, 6, 12, 24]**; lite YAML |
| Window size | 7; lite YAML |
| MLP ratio / QKV bias | 4 / true; `config.py` |
| Position and patch normalization | absolute position embedding false; patch normalization true; `config.py` |
| Output classes | **9** for the official Synapse path, representing the eight target organs plus background; `train.py` |
| Segmentation head | 1 × 1 convolution, 96 input channels, nine outputs, **no bias** |

The decoder-depth discrepancy is material to reproducibility. The YAML's
`DECODER_DEPTHS: [2, 2, 2, 1]` is dead configuration in the released path:
`vision_transformer.py` does not pass it, and
`SwinTransformerSys.__init__` constructs decoder blocks from the reversed
encoder `depths` even though it accepts and prints a `depths_decoder`
argument. The independently inspected model therefore has two blocks at each
of its three transformer decoder stages.

The name “Tiny” also needs qualification. The checkpoint name and original
paper call this Tiny-based, but the lite YAML reduces the encoder's third
depth from the Python/Swin-T default of 6 to 2. The reproduced count below is
for the authors' actual released **Tiny/lite YAML path**, not for an
unmodified Swin-T classifier or for a Base Swin-Unet.

### Reported and independently reproduced counts

No training, data access, pretrained-weight download, or model checkpoint was
used. The paper-era constructor was evaluated in two independent
programmatic ways:

1. direct execution of the pinned constructor with in-memory, shape-only
   PyTorch compatibility classes, followed by the equivalent of
   `sum(p.numel() for p in model.parameters())`; and
2. an explicit parameter-shape ledger derived from every instantiated
   convolution, linear layer, layer normalization, relative-position-bias
   table, patch merge, patch expansion, and segmentation-head tensor.

Both methods produced 218 parameter tensors and exactly **27,168,900
parameters**:

| Instantiated component | Reproduced parameters |
|---|---:|
| Three-channel patch projection and patch normalization | 4,896 |
| Four encoder/bottleneck stages, including three patch-merging layers | 20,406,954 |
| Final encoder layer normalization | 1,536 |
| Decoder layers, including initial/intermediate patch expansions | 6,219,066 |
| Three skip-concatenation linear projections | 387,744 |
| Final decoder layer normalization | 192 |
| Final ×4 patch expansion and normalization | 147,648 |
| Bias-free 96→9 segmentation head | 864 |
| **Total** | **27,168,900** |

All 27,168,900 are trainable PyTorch parameters at construction. Shifted
attention masks and relative-position indices are registered buffers, not
parameters; there are no frozen parameter tensors. The source does not
provide an independently **reported** original count with which to compare
this constructor total.

The complete count distinction is:

| Evidence class | Swin-Unet count | Status |
|---|---:|---|
| S2M-Net paper | Not stated | Only the approximate “6× fewer” wording is stated |
| S2M-Net released README | **27.0M** | Reported to one decimal, without variant/configuration provenance |
| Original Swin-Unet paper | Not stated | Specifies Tiny, 224 input, and patch size 4, but no count |
| Original Swin-Unet repository README | Not stated | Selects the executable Tiny/lite YAML |
| Reproduced original authors' 224, nine-class Tiny/lite code | **27,168,900** | Exact constructor count |

Changing only the output head to one channel gives **27,168,132**
parameters; each output class changes the bias-free head by 96 parameters.
Changing the model input from three channels to one would remove 3,072 patch
projection weights, but that is not the official path: the wrapper repeats
grayscale input to three channels. With absolute position embedding disabled,
the same explicit architecture at 352 × 352 also has 27,168,900 parameters:
all four stage resolutions still use the configured 7 × 7 attention window.

These sensitivity counts do not establish how the S2M-Net authors adapted
Swin-Unet across binary and multiclass datasets. Their README does not record
the baseline's input-channel convention, head classes, encoder/decoder
depths, or whether the 352 × 352 efficiency row used Cao et al.'s code.

### Programmatic ratio and numerical assessment

Using the only count reported by the released S2M-Net materials and the
canonical exact S2M-Net count:

```text
27,000,000 / 4,791,544 = 5.6349268628233404514286000504
```

Using the independently reproduced original authors' nine-class Tiny/lite
count:

```text
27,168,900 / 4,791,544 = 5.6701764608652242366969811818
```

For reference, performing arithmetic only on the README's displayed
one-decimal values gives:

```text
27.0 / 4.7 = 5.7446808510638297872340425532
```

All three ratios round to **6× at zero decimal places**, although the exact
ratios are approximately 5.63× and 5.67× when the established S2M-Net
denominator is preserved. An exact 6× ratio would require 28,749,264
Swin-Unet parameters. The reported 27.0M is 6.0846% below that target, and
the reproduced 27,168,900 is 5.4971% below it. Thus “roughly 6×” is
numerically reasonable as an integer-level approximation, not an exact
ratio.

### Bundled S2M-Net implementation check

No bundled S2M-Net approximation implements the cited/named Swin-Unet
architecture:

- `official_repo/s2mnet/models/baselines.py` defines U-Net, U-Net++,
  TransUNet, and UMamba only;
- `official_repo/s2mnet/models/__init__.py`, lines 16–32, imports and exports
  those same four baselines;
- `official_repo/train.py` has no Swin-Unet model-selection branch;
- no Swin block, shifted-window attention module, Swin-Unet constructor, or
  Swin-Unet YAML exists in the released source; and
- all local “Swin-Unet” occurrences outside this diagnostic report are
  README result/efficiency entries plus the existing reproduction-plan note
  that the baseline is omitted.

Consequently, there is no local approximation whose structure or count can
be matched to 27.0M. The S2M-Net results and efficiency tables are not backed
by a recoverable Swin-Unet construction in the released project.

### Configuration-matching limitations

1. The S2M-Net paper does not cite Cao et al.'s Swin-Unet paper and never
   states a Swin-Unet parameter count or variant.
2. The paper's Swin-related bibliography entry is for Swin-UNETR, which
   cannot be substituted for the differently named 2D Swin-Unet baseline.
3. The released README's 27.0M is a one-decimal value with no model summary,
   source revision, configuration, checkpoint, or counting rule.
4. The original Swin-Unet paper and README do not publish a scalar parameter
   count. The 27,168,900 value is an independent reconstruction of their
   released paper-era Tiny/lite Synapse path.
5. The original code's decoder-depth YAML is unused, and current repository
   `train.py` has drifted from the original nine-class Synapse default to a
   four-class generic default. Exact reproduction therefore requires a pinned
   commit and inspection of effective, not merely written, configuration.
6. The S2M-Net comparison spans datasets with different channel and class
   requirements but does not document baseline adaptations. The one-class
   and 352 × 352 counts above are sensitivity checks, not silently
   substituted baselines.
7. The constructor reproduction used shape-only PyTorch compatibility
   classes because PyTorch is not installed in the project environment. Its
   result was independently verified by a complete explicit parameter-shape
   ledger, but no official checkpoint was loaded.

### Conservative Swin-Unet-component verdict

**Conditionally supported numerically, but not configuration-verified.** The
released S2M-Net value of 27.0M divided by the canonical exact S2M-Net count
is **5.6349268628×**, and the independently reproduced original authors'
nine-class Tiny/lite count gives **5.6701764609×**. Either reasonably rounds
to “roughly 6×” at integer precision. However, the S2M-Net paper does not cite
or configure Cao et al.'s Swin-Unet, its only Swin segmentation citation is
the different Swin-UNETR architecture, and the released S2M-Net project
contains no Swin-Unet implementation. The official Tiny/lite model is a
plausible source for the 27.0M table value, not a verified identity. This is a
verdict only on the Swin-Unet parameter-comparison component, not on Claim 1
as a whole.
