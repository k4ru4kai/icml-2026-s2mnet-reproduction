# Claim 1 - architecture and parameters


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5e46ec48fa90", "created_at": "2026-07-27T16:46:11+00:00", "title": "Final Claim 1 evidence and verdict"}
-->
## Claim under examination

> “S2M-Net is a 4.7M-parameter architecture, roughly 13x fewer parameters
> than TransUNet and 6x fewer than Swin-UNet, using a five-stage encoder with
> channel dimensions {24, 32, 64, 80, 128}.”

## Verdict

**Partially verified.**

The architecture itself is directly supported by the released configuration,
source construction, and a diagnostic forward pass. The paper's rounded model
size and both baseline ratios require qualifications: 4,791,544 conventionally
rounds to 4.8M rather than 4.7M, the TransUNet ratio depends on an undocumented
reported baseline count, and the exact Swin-Unet comparison configuration
cannot be recovered.

## Evidence classification

- **Directly verified:** reproduced from the released S2M-Net configuration
  and construction path, with parameter-shape accounting or runtime tensor
  inspection.
- **Reported without executable configuration:** stated in the S2M-Net paper
  or README, but not linked to a recoverable baseline configuration, model
  summary, checkpoint, or counting script.
- **Independently reproduced baseline count:** calculated from a pinned
  baseline authors' implementation and its documented configuration; this is
  not silently substituted for the S2M-Net paper's unidentified baseline.
- **Unresolved ambiguity:** paper and released implementation do not provide
  enough matching information for a definitive equivalence.

## Executable audit closure

The local Claim 1 evidence is now backed by an executable, build-only audit:

- `repro/diagnostics/verify_claim1_architecture.py`;
- `tests/test_claim1_architecture_audit.py`; and
- `results/audits/claim1_architecture_parameters.json`.

`CLAIM1_AUDIT_STATUS=PASS` means that the verification procedure and all of
its internal expected-versus-observed checks passed. It does **not** mean
that the complete paper claim was reproduced.

| Evidence | Counting method | Parameters | Ratio to exact 352 S2M-Net |
|---|---|---:|---:|
| S2M-Net, 352 × 352, one class | Executed TensorFlow constructor plus independent shape sum | **4,791,544** | 1.000× |
| S2M-Net, 256 × 256, one class | Executed TensorFlow constructor plus independent shape sum | **4,766,008** | 0.994671× |
| Bundled TensorFlow TransUNet, Keras-tracked | Executed TensorFlow constructor plus tracked-variable shape sum | **5,437,825** | 1.134879× |
| Bundled TensorFlow TransUNet, including untracked positional embedding | Constructor count plus explicit tensor ledger | **5,933,441** | 1.238315× |
| Official TransUNet `R50-ViT-B_16` | Transparent ledger at pinned commit `26de0c4d9a5145589ea249d169af7f7130823e03`; constructor not executed | **105,277,081** | 21.971432× |
| Official Swin-Unet Tiny/lite | Transparent ledger at pinned commit `1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9`; constructor not executed | **27,168,900** | 5.670176× |
| README TransUNet value | Externally reported value only | **60.0M** | 12.522060× |
| README Swin-Unet value | Externally reported value only | **27.0M** | 5.634927× |

The 25,536-parameter difference between the two S2M-Net resolutions is fully
localized to `sstm_stage4/freq_weights:0` (**+18,240**) and
`sstm_stage5/freq_weights:0` (**+7,296**). The instantiated audit also
confirms five encoder stages with channels **{24, 32, 64, 80, 128}**.

The verdict remains **Partially verified**, not falsified. The exact S2M-Net
count conventionally rounds to approximately 4.8M; the claimed 60M TransUNet
configuration is unidentified; the pinned official TransUNet gives
approximately 22× rather than 13×; and the bundled implementation including
its untracked positional embedding gives only approximately 1.24×.
Swin-Unet remains compatible with approximately 6×, but the S2M-Net
materials do not identify the exact compared configuration.

## Exact S2M-Net parameter count

**Evidence class: directly verified.**

The canonical paper-consistent released-code Full build uses a 352 × 352 RGB
input and one output class. Two independent calculations agree:

| Parameter partition | Exact count |
|---|---:|
| Trainable | 4,768,920 |
| Non-trainable | 22,624 |
| **Total** | **4,791,544** |

The two calculations were Keras `count_params()` and the independent sum of
the products of all model-weight shapes. The total also matches the completed
DRIVE Full training logs for seeds 42, 7, and 123.

The paper's “4.7M” is a coarse approximation or truncation, not conventional
one-decimal rounding:

```text
4,791,544 parameters = 4.791544M parameters → 4.8M at one decimal place
```

The exact count is also resolution-dependent in the released model. At
256 × 256, the same effective architecture has 4,766,008 parameters because
the SSTM frequency grids at stages 4 and 5 are smaller. The canonical
4,791,544 value is selected because the paper specifies stage resolutions
{176, 88, 44, 22, 11}, which arise from a 352 × 352 input.

Exact local source paths:

- `docs/claim1_parameter_count_investigation.md`, sections “Canonical count
  for Claim 1” and “Relationship to the paper's ‘4.7M’”;
- `outputs/drive/full_seed42/config/frozen.yaml`, model configuration at
  lines 55–97;
- `outputs/drive/full_seed42/logs/train.log`, recorded parameter count at
  line 8;
- `repro/experiments/drive_train.py:official_model_config`, lines 490–495,
  and `build_training_objects`, lines 498–522;
- `official_repo/train.py:build_model`, lines 153–175; and
- `official_repo/s2mnet/models/s2mnet.py:S2MNet`.

**Component assessment:** the exact released-code count is verified, but the
paper's “4.7M” label is not its conventional one-decimal rounding.

## Five encoder stages

**Evidence class: directly verified.**

The released constructor requires five filter entries and executes five
post-stem encoder iterations. A zero-input diagnostic forward pass at
352 × 352 confirmed the symbolic and runtime output shapes:

| Encoder component | Output resolution | Applied released-code modules |
|---|---:|---|
| Stage 1 | 176 × 176 | Stride-2 convolution, BN, ELU, MRF-SE, SSTM |
| Stage 2 | 88 × 88 | Stride-2 convolution, BN, ELU, MRF-SE, SSTM |
| Stage 3 | 44 × 44 | Stride-2 convolution, BN, ELU, MRF-SE, SSTM |
| Stage 4 | 22 × 22 | Stride-2 convolution, BN, ELU, MRF-SE, SSTM |
| Stage 5 / bridge | 11 × 11 | Stride-2 convolution, BN, ELU, MRF-SE, SSTM |

The separate 16-channel, full-resolution stem is not one of these five
stages. MRF-SE and an SSTM layer are instantiated at every stage. In the
released implementation, SSTM is spectral-only at stages 1–2 and uses both
spectral and selective/spatial paths at stages 3–5.

Exact local source paths:

- `docs/claim1_parameter_count_investigation.md`, section “Five-stage encoder
  and channel-dimension verification”;
- `outputs/drive/full_seed42/config/frozen.yaml`, lines 55–97;
- `repro/experiments/drive_train.py`, lines 490–501;
- `official_repo/train.py`, lines 153–175;
- `official_repo/s2mnet/models/s2mnet.py`, lines 81–129; and
- `official_repo/s2mnet/models/blocks.py`, SSTM and MRF-SE implementations.

**Component assessment:** supported.

## Channel dimensions

**Evidence class: directly verified.**

The frozen configuration, released constructor, symbolic shapes, and runtime
tensor probes all agree:

| Encoder stage | Output channels |
|---:|---:|
| 1 | **24** |
| 2 | **32** |
| 3 | **64** |
| 4 | **80** |
| 5 | **128** |

Therefore the instantiated channel sequence is exactly
**{24, 32, 64, 80, 128}**.

Exact local source paths:

- `docs/claim1_parameter_count_investigation.md`, sections “Instantiated
  encoder,” “Runtime tensor-shape confirmation,” and “Paper-versus-code
  comparison”;
- `outputs/drive/full_seed42/config/frozen.yaml`, filter sequence at
  lines 55–97;
- `official_repo/train.py:build_model`, lines 153–175; and
- `official_repo/s2mnet/models/s2mnet.py:S2MNet`, lines 81–129.

**Component assessment:** supported.

## TransUNet parameter comparison

**Evidence classes: reported without executable configuration, plus an
independent reproduction that does not match the reported value.**

The S2M-Net paper states “roughly 13× fewer” but does not identify a TransUNet
variant or give an explicit TransUNet parameter count. The released S2M-Net
README reports **60.0M** and **12.8× fewer**, without a matching backbone,
configuration, checkpoint, model summary, or count script.

Using the canonical exact S2M-Net denominator:

```text
60,000,000 / 4,791,544
  = 12.52205969516297878095244456
```

This rounds to 13× at integer precision. Arithmetic on the README's two
displayed values gives:

```text
60.0 / 4.7 = 12.76595744680851063829787234
```

which rounds to the README's 12.8×. Thus “roughly 13×” is numerically
reasonable only if the undocumented 60.0M baseline value is accepted.

The original TransUNet authors' official 224 × 224, nine-class Synapse
`R50-ViT-B_16` configuration was independently reproduced from pinned code:

```text
105,277,081 / 4,791,544 = 21.9714315469×
```

The original paper and README do not report a scalar parameter count, and the
reproduced 105,277,081 total does not reproduce 60.0M. It is evidence that the
reported baseline is not the identifiable official default, not a replacement
denominator for the S2M-Net claim. The bundled S2M-Net TensorFlow
“TransUNet” is a much smaller approximation and also does not reproduce
60.0M.

Exact source paths:

- `docs/claim1_parameter_count_investigation.md`, section “TransUNet
  parameter-comparison verification”;
- `official_repo/README.md`, line 25, lines 182–212, and lines 350–364;
- pinned upstream `Beckschen/TransUNet` paths documented in the
  investigation:
  `networks/vit_seg_configs.py`,
  `networks/vit_seg_modeling.py`,
  `networks/vit_seg_modeling_resnet_skip.py`, and `train.py`; and
- `official_repo/s2mnet/models/baselines.py`, lines 95–181, for the
  non-matching bundled approximation.

**Component assessment:** conditionally supported numerically, but not
configuration-verified.

## Swin-Unet parameter comparison

**Evidence classes: reported without executable configuration, plus an
independently reproduced plausible baseline count.**

The S2M-Net paper states “roughly 6× fewer” without giving a Swin-Unet count,
variant, configuration, or citation on either occurrence. The released
S2M-Net README reports **27.0M**, again without executable provenance.

Using the reported value and canonical exact S2M-Net denominator:

```text
27,000,000 / 4,791,544
  = 5.6349268628233404514286000504
```

The original Swin-Unet authors' paper-era 224 × 224, nine-class Tiny/lite
configuration was independently reproduced as **27,168,900 parameters**:

```text
27,168,900 / 4,791,544
  = 5.6701764608652242366969811818
```

Arithmetic on the README's displayed values gives:

```text
27.0 / 4.7 = 5.7446808510638297872340425532
```

All three ratios round to 6× at integer precision, so “roughly 6×” is
numerically reasonable. However, the original Swin-Unet paper and README do
not publish a scalar count, and the S2M-Net materials do not identify their
baseline configuration. The Tiny/lite build is a plausible source for the
27.0M value, not a verified identity. Its output head has nine classes;
changing it to one class gives 27,168,132 parameters, illustrating one of the
undocumented baseline adaptations.

The citation chain is also mismatched: the S2M-Net paper contains no
bibliography entry for Hu Cao et al.'s 2D Swin-Unet. Its Swin segmentation
citation points to Tang et al.'s **Swin-UNETR**, a distinct 3D architecture
that cannot establish the identity of the 27.0M comparison. The released
S2M-Net project contains no bundled Swin-Unet implementation or
configuration.

Exact source paths:

- `docs/claim1_parameter_count_investigation.md`, section “Swin-UNet
  parameter-comparison verification”;
- `official_repo/README.md`, lines 182–212, 214–256, 350–363, 378–388,
  and 765;
- S2M-Net arXiv v1 source paths documented in the investigation:
  `sec/1_intro.tex` and `main.bib`; and
- pinned upstream `HuCaoFighting/Swin-Unet` paths documented in the
  investigation:
  `configs/swin_tiny_patch4_window7_224_lite.yaml`, `config.py`, `train.sh`,
  `train.py`, `networks/vision_transformer.py`, and
  `networks/swin_transformer_unet_skip_expand_decoder_sys.py`.

**Component assessment:** numerically reasonable, but the exact compared
baseline is not configuration-verifiable.

## Unresolved paper–implementation ambiguities

The following qualifications remain open and prevent a stronger overall
verdict:

1. The released S2M-Net parameter total changes with input resolution because
   SSTM allocates learned frequency grids from the stage feature-map sizes.
2. The paper describes an SSTM spatial bottleneck dimension `d=16`; the
   released layer stores this setting but projects channel dimension C
   directly to C, so `d=16` is unused.
3. The paper describes SSTM generally as dual-branch at every encoder stage,
   whereas the released stages 1–2 are spectral-only.
4. The released architecture diagram contains a stage-label typo at the
   22 × 22 encoder panel, although the prose, table, constructor, and runtime
   shapes consistently resolve five stages.
5. Neither reported baseline count has an executable configuration in the
   S2M-Net release. The TransUNet 60.0M value is undocumented, while the
   Swin-Unet name and 27.0M value are not tied to a matching citation or
   baseline build.

## Conclusion

Claim 1 is partially verified. The canonical released-code S2M-Net build at
352 × 352 has 4,791,544 parameters and directly supports a five-stage encoder
with channel dimensions {24, 32, 64, 80, 128}; however, 4,791,544
conventionally rounds to 4.8M, not 4.7M. The “roughly 13×” TransUNet
comparison is conditional on an undocumented reported value of 60.0M, while
“roughly 6×” for Swin-Unet is numerically reasonable but the exact compared
baseline is not configuration-verifiable. The S2M-Net paper's Swin citation
points to the distinct Swin-UNETR architecture, and the documented
paper–implementation ambiguities remain unresolved.

## Authoritative evidence record

This page is supported by
`docs/claim1_parameter_count_investigation.md` (SHA-256
`3455e4378ca09e198e6f4c937a0eb7f831e1cc423736c3a959a6b86c140abc14`),
`repro/diagnostics/verify_claim1_architecture.py` (SHA-256
`16daac79128baea3c193ac2c1e16cc4b6a0d73b42c9166ef28ffab11eae0ce56`),
`tests/test_claim1_architecture_audit.py` (SHA-256
`22cf6cb94319d3d444639a8f7ca84969ef1ec421119d069e503276b9127df4bd`),
and the deterministic
`results/audits/claim1_architecture_parameters.json` (SHA-256
`89098fa4456efe2be46401f64c8fc8963a82e82c5b8eff264c59c63e947e7e37`).
No model was trained and no claim evidence was drawn from hidden-test data.
