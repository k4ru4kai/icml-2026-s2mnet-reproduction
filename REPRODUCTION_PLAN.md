# S2M-Net reproduction plan (approval-gated)

Status: the local environment and official-model smoke validation are complete.
The `.venv` and public Trackio challenge logbook are configured. No dataset has
been downloaded, no training or Hub Job has run, and no definitive empirical
result or experiment artifact is available yet.

## 1. Source snapshot and provenance

- Challenge guide: `ICML-2026-agent-repro/challenge` README, read on 2026-07-17.
- Paper target: arXiv `2601.01285v1`, verified from the version-pinned HTML and a
  PDF with a `%PDF-1.7` signature. The authoritative PDF SHA-256 is
  `250e722fc2069be6b281cdb0c9f99b5b4e87c073799daec0bb93eafd73b2057c`.
- OpenReview id: `eh48NIgu9z`. The forum requires an interactive browser check in
  this environment, so no numerical claim is extracted from a verification page.
- Official code: `sanaullah-ashfat/S2M-Net-Spectral-Spatial-Mixing-for-Medical-Segmentation`
  at commit `3ec59668ab9b438ab9b170306d29b01e9270fd5a` (2026-07-08).
  There are no tags or releases, so this commit is the pinned implementation target.

The paper-v1 numbers and current repository README numbers are kept as two source
versions rather than silently merged:

| Quantity | arXiv v1 target | Current official README |
|---|---:|---:|
| EndoVis17 multiclass Dice | 83.77 | 83.43 |
| BraTS2020 Dice | 80.90 | 79.96 |
| CHASE-DB Dice | 83.70 | 84.95 |
| K=32 mean spectral energy | >95% (general) | 94.8% overall; 96.4% fundus |
| Kvasir full / no-SSTM / K=16 | 96.12 / 88.45 / 89.87 | 96.05 / 88.45 / 89.87 |
| Kvasir reported drops | 7.67 / 6.25 points | 7.60 / 6.18 points by subtraction |

## 2. Exact reproduction scope

The empirical priority is Claim 2 plus the retinal analogue of Claims 4-6:

1. **Architecture audit.** The released DRIVE/retinal Full Model builds with the
   five encoder stages and channels `{24, 32, 64, 80, 128}`. TensorFlow 2.15.1
   confirms exactly 4,766,008 parameters: 4,743,384 trainable and 22,624
   non-trainable. Per-component counts, FLOPs, inference memory, and latency
   remain to be measured.
2. **Spectral mechanism.** On retinal images resized to 352x352, measure the
   fraction of squared FFT magnitude inside centered KxK crops for
   `K={16,24,32,48,64}`. Pre-registered checks are the paper-v1 `K=32 >95%`
   statement, the current README's fundus value (96.4%), and the direction and
   size of the K=16 reduction.
3. **Cost claim.** Compare corrected SSTM with full spatial self-attention at the
   same stage shapes and output widths. Report analytical operations, measured
   latency, throughput, and peak GPU memory separately. The 63% claim is supported
   only for a named cost metric if `1 - SSTM_cost / attention_cost >= 0.63` under
   the pre-registered benchmark; an OOM is recorded, not converted into a speedup.
4. **Controlled retinal ablation.** Primary dataset is DRIVE because the paper's
   ablation table provides a retinal full/no-SSTM pair (84.83 vs 77.78, a 7.05
   point drop; the current README gives 84.06 vs 77.78, a 6.28 point drop).
   The primary comparison is the unmodified official Full Model versus the
   official No-SSTM ablation, with identical data, split, preprocessing,
   optimizer, batch size, epoch budget, evaluator, and seeds. K=16 and a
   paper-faithful SSTM correction are separate secondary diagnostic arms.
5. **CHASE-DB fallback/extension.** If DRIVE access/licensing cannot be cleared,
   use CHASE_DB1 to test the 83.70 (v1) / 84.95 (current README) absolute claim
   and the full/no-SSTM direction. This is explicitly not a direct reproduction
   of the Kvasir-specific 7.67, 8.16, and 6.25 point claims.

Direct Kvasir replication of the no-SSTM, no-boundary-loss, and K=16 numeric drops
is a later extension, not part of the first retinal-focused campaign.

## 3. Repository structure and audit findings

The official repository contains:

- `s2mnet/models/blocks.py`: SSTM, MRF-SE, and BFP/BFD blocks.
- `s2mnet/models/s2mnet.py`: five-stage encoder-decoder.
- `s2mnet/losses/`: MASL/MAL and five components.
- `s2mnet/dataloaders/`: full-image and retinal patch loaders.
- `s2mnet/utils/spectral.py` and `scripts/analyze_spectral.py`: centered FFT
  energy and reconstruction analysis.
- `experiments/ablation_configs.py`: 23 named ablations; ids 8, 15, and 2 are
  K=16, no-SSTM, and no-boundary-loss respectively.
- `train.py`, `test.py`, and YAML configs.

Material implementation risks and validation findings:

1. `tf.signal.fft2d` transforms the last two axes. The released SSTM calls it on
   channels-last `[B,H,W,C]`, so it transforms `(W,C)`, not `(H,W)`.
2. The released SSTM bilinearly resizes the complex spectrum instead of applying
   `fftshift -> centered crop -> zero pad -> ifftshift`; it therefore does not
   implement equations 4-7 as written.
3. `ssm_state_dim=16` is stored but unused. The selective path projects C to C,
   not through the claimed `d=16` bottleneck.
4. Frequency weights are real, not complex, and their initializer is unshifted.
5. The README parameter table is close to the released model, but the bundled
   baselines omit Swin-Unet and do not reproduce the stated TransUNet/UMamba
   sizes; `train.py` cannot select any of those baseline builders.
6. The YAML files claim inheritance from `default.yaml`, but `load_config` does
   not merge defaults. Paper, README, default YAML, and retinal YAML disagree on
   optimizer, learning rate, epochs, and batch size.
7. The official requirements are lower bounds only and mix Albumentations v1 and
   v2 argument names. The reproduction environment therefore uses the exact pins
   in `requirements-repro.txt`. The official source has no lockfile, package
   metadata, container, CI, or unit test suite.
8. The project `.venv` is configured with TensorFlow 2.15.1, Keras 2.15.0, and
   NumPy 1.26.4. The official Full Model builds and completes one deterministic,
   finite forward pass; the exact parameter count is 4,766,008. GPU detection
   inside the Codex sandbox may fail, while the external Ubuntu host has verified
   TensorFlow access to the RTX 4060.
9. Validation for `PatchDataset` is stochastic patch sampling rather than fixed
   full-image evaluation. `test.py` resizes retinal images to one 256x256 patch,
   and does not evaluate native-resolution tiled predictions inside the supplied
   FOV mask.
10. `test.py` treats every task as binary (channel 0 and thresholding), so it
    cannot verify the multiclass EndoVis/BraTS claims.
11. Trainable MASL weights live in a loss Layer outside the model. A smoke test
    must prove they are included in optimizer variables; otherwise the claimed
    learned weights remain at initialization and require a custom `train_step`.
12. HDF5 loading lists the loss only, not the three custom model layers. Save/load
    round-trip must be tested before a long run.
13. No checkpoints, raw result tables, split manifests, data-preparation scripts,
    or run logs are released. The repository says MIT in README but contains no
    `LICENSE` file and GitHub detects no license.

The reproduction will preserve the official checkout untouched. All fixes and
tests go under a separate `repro/` tree and every deviation is documented.

## 4. Validated environment

The project `.venv` uses Python 3.11. TensorFlow 2.15.1, Keras 2.15.0, and
NumPy 1.26.4 are installed and were smoke-tested on 2026-07-21. TensorFlow was
built for CUDA 12.2 and cuDNN 8. Outside the Codex sandbox, TensorFlow detects
`/physical_device:GPU:0` on an NVIDIA GeForce RTX 4060; sandbox GPU discovery
may still fail because the device is not exposed there. Mixed precision remains
disabled for the primary comparison unless every variant passes numerical parity.

The approved installation command was:

```bash
uv --cache-dir /tmp/s2mnet-uv-cache pip install \
  --python .venv/bin/python \
  --requirement requirements-repro.txt \
  'tensorflow[and-cuda]==2.15.1'
```

Future experiment records must also capture OS, CUDA, cuDNN, GPU model,
TensorFlow build info, Git commit, config hash, and dataset manifest hash.

## 5. Dataset acquisition, split, and licensing

### Primary: DRIVE

- Authoritative source: the DRIVE Grand Challenge / Utrecht Image Sciences
  Institute. It contains 40 images with the standard 20 training / 20 test split.
- The site requires browser access in this environment and an explicit reusable
  license was not confirmed during phase 1. Raw images will not be republished
  until terms are verified.
- A public HF mirror is visible at
  `Zomba/DRIVE-digital-retinal-images-for-vessel-extraction`, pinned at revision
  `e43e3f8a4b1146db231648786912d0ac882067b2`; its metadata is not treated as a
  replacement license.
- Fixed split: train IDs 21-36, validation IDs 37-40, official test IDs 01-20.
  Use observer-1 vessel masks and official FOV masks if present. Threshold is 0.5.
- The visible HF mirror tree has images and observer-1 labels but no FOV-mask
  files. Prefer the authoritative archive; if masks remain unavailable, derive
  the FOV deterministically from the black fundus background, store it in the
  manifest, and never substitute the repository's generic centered circle
  without reporting that protocol change.

After approval, the mirror fallback command would be:

```bash
hf download Zomba/DRIVE-digital-retinal-images-for-vessel-extraction \
  --repo-type dataset \
  --revision e43e3f8a4b1146db231648786912d0ac882067b2 \
  --local-dir data/raw/drive
```

### Fallback/extension: CHASE_DB1

- The official Kingston record states 28 images from 14 children, two eyes per
  child, two manual segmentations, a 2.4 MB archive, and CC BY 4.0.
- Its file endpoint returns a browser challenge to automated requests here.
- The visible HF mirror `Zomba/CHASE_DB1-retinal-dataset` is pinned at revision
  `d956ca1aa2cc805b5f62f105d8c4b6d5ba812d8f` and contains all 28 images.
- Subject-disjoint fixed split: subjects 01-05 train (10 images), 06-07 validation
  (4 images), and 08-14 test (14 images). Never split left/right eyes across sets.

For either dataset, write `manifest.csv` with source revision, relative path,
SHA-256, dimensions, split, subject/image id, mask observer, and FOV provenance.
No training begins until counts, pairings, binary values, and leakage tests pass.

## 6. Implementation and smoke tests

Implement two explicit SSTM modes:

- `released`: byte-faithful official behavior used for the primary Full Model
  versus No-SSTM comparison.
- `spatial_fft`: transpose to `[B,C,H,W]`, spatial `fft2d`, centered crop,
  learnable complex filter, centered zero-pad, inverse FFT, transpose back; use
  the declared `d=16` selective bottleneck as a separate diagnostic variant.

The primary empirical result will use the unmodified official implementation.
The paper-faithful `spatial_fft` variant is reported separately and does not
replace that comparison. No silent patching.

Smoke gates:

1. Build at 64, 256, and 352 pixels; assert output shape and finite values.
2. Assert FFT axes using separable sinusoidal inputs and compare against NumPy.
3. Assert K=32 crop/pad and Hermitian/reconstruction behavior.
4. Global-dependence test: a remote input impulse changes distant output pixels.
5. One forward/backward step for full K32, K16, and no-SSTM.
6. Confirm nonzero gradients for spectral weights and MASL weights; confirm MASL
   weights change after one optimizer step.
7. Save/load round trip and identical inference within tolerance.
8. Confirm parameter counts and profile FLOPs/peak memory.
9. Run a tiny synthetic 2-image, 2-epoch overfit test for all variants.
10. Run data-manifest, split-leakage, native-resolution tiling, and FOV tests.

Planned smoke commands:

```bash
.venv/bin/pytest -q repro/tests
trackio logbook run --page "Smoke tests" -- \
  .venv/bin/python -m repro.audit_model --config repro/configs/drive.yaml
trackio logbook run --page "Claim 2 - SSTM mechanism" -- \
  .venv/bin/python -m repro.analyze_spectrum \
  --manifest data/processed/drive/manifest.csv --resize 352 \
  --k 16 24 32 48 64 --output outputs/spectral
```

## 7. Pilot experiment

Run one seed (`42`) for 10 epochs for:

- released official Full Model K=32 (primary);
- released official No-SSTM (primary);
- released official K=16 (secondary diagnostic);
- paper-faithful `spatial_fft` Full Model K=32 (secondary diagnostic).

Common settings: same DRIVE split, 256x256 native retinal patches, green-channel
CLAHE, supplied FOV, batch 8, Adam 1e-4, 4,000 sampled training patches per epoch,
fixed augmentation sequence per seed, no test-time augmentation, and fixed 0.5
threshold. Evaluate complete native-resolution validation/test images with overlap
tiling; do not use stochastic validation patches.

Pilot acceptance gates:

- all runs finite and save/load cleanly;
- full/K16 spectral gradients nonzero;
- no split leakage and every test pixel inside FOV covered;
- measured memory <=14 GB on a 16 GB T4 (otherwise batch 4 plus identical gradient
  accumulation to effective batch 8 for every variant);
- exact measured step time is used to forecast final cost before launch.

## 8. Final experiment

Conditional on pilot approval, run the unmodified official Full Model and
official No-SSTM for seeds `{42,123,456,789,2024}`, 100 epochs each, with
identical settings and fixed data order/augmentation streams for paired
comparisons. Train all 100 epochs and select the best validation-Dice checkpoint
under the same rule; evaluate the untouched test set once per run. K=16 and the
paper-faithful `spatial_fft` implementation remain separately labelled secondary
experiments and are not part of the primary inference.

Primary endpoint: per-image hard Dice within FOV, macro-averaged over the test set,
then mean and sample standard deviation over seeds. Secondary endpoints: soft Dice,
IoU, precision, recall/sensitivity, specificity, F1, HD95, clDice/topology score,
parameter count, FLOPs, peak memory, training time, inference latency, and throughput.

Report paired full-minus-ablation differences for each seed, a paired t-test only
for comparison with the paper, Wilcoxon as a robustness check, and a hierarchical
bootstrap confidence interval over seeds/images. Do not claim reproduction from a
direction-only match: absolute score, effect size, uncertainty, and protocol drift
are all shown.

Qualitative outputs: fixed best/median/worst cases selected by full-model Dice,
with the same cases shown for all variants; probability maps, binary masks, error
maps, BFD routing maps, vessel skeleton/connectivity overlays, FFT magnitude and
K=16/K=32 reconstructions. Store raw CSV/JSON behind every figure.

## 9. GPU memory, runtime, and cost gate

- Smoke: local CPU or GPU, approximately 5-15 minutes; no paid Job required.
- Pilot: expected 6-10 GB GPU memory at batch 8 and 256x256; approximately 0.5-2
  T4 GPU-hours total. This must be replaced by measured timing before final runs.
- Primary final comparison: 10 runs x 50,000 optimizer steps/run at batch 8. At
  0.10-0.30 s/step, approximately 14-42 T4 GPU-hours. Current `t4-small` pricing
  observed on 2026-07-17 is $0.40/hour, giving a provisional $5.60-$16.80 range.
  Any secondary K=16 or `spatial_fft` run requires a separate estimate and approval.
- Each Job gets a hard timeout; no final campaign launches if the pilot projection
  exceeds the agreed cap. Use one seed per Job with the two primary variants
  sequentially on the same GPU to reduce hardware variance.

After approval, first run the seconds-long CPU canary required by the challenge:

```bash
hf jobs run --timeout 2m python:3.12 python -c "print('ok')"
```

An illustrative paid command (not yet authorized or run) is:

```bash
hf jobs run --flavor t4-small --timeout 8h --detach \
  --label paper=eh48NIgu9z --label seed=42 \
  -v ./repro:/workspace/repro:ro \
  -v hf://datasets/Zomba/DRIVE-digital-retinal-images-for-vessel-extraction:/data:ro \
  -v hf://buckets/K4ru4k4i/s2mnet-repro:/outputs:rw \
  --secrets HF_TOKEN \
  tensorflow/tensorflow:2.15.1-gpu \
  bash /workspace/repro/jobs/run_seed.sh 42 /data /outputs
```

## 10. Trackio logbook

The challenge logbook is configured and public at
`https://huggingface.co/spaces/K4ru4k4i/eh48NIgu9z`, with automatic sync enabled
and `private: false`. It documents scope and claims but contains no definitive
empirical result. Title:
`Repro - S2M-Net: Spectral-Spatial Mixing with Morphology-Aware Adaptive Loss`.
Metadata:

```json
{
  "paper": {"openreview_id": "eh48NIgu9z", "arxiv_id": "2601.01285"},
  "tags": ["icml2026-repro", "paper-eh48NIgu9z"],
  "space_id": "K4ru4k4i/eh48NIgu9z",
  "autosync": true,
  "private": false
}
```

Index is TOC-only with Status and Decision columns. Pages:

1. Provenance and preregistered protocol
2. Claim 1 - architecture and parameters
3. Claim 2 - SSTM mechanism, energy, and cost
4. Retinal data, licensing, split, and manifest
5. Smoke tests and released-code audit
6. Pilot - full vs no-SSTM vs K16
7. Final retinal ablation
8. Qualitative and topology analysis
9. Source-version differences and limitations
10. Conclusion

Every training script uses `trackio.init/log/finish` and is launched through
`trackio logbook run`. Log config, git/data hashes, epoch, learning rate, every
train/validation metric, component losses, MASL weights, gradient norms, step time,
GPU memory, and checkpoint path. Alerts: NaN/Inf, zero spectral gradient, unchanged
MASL weights, empty/full masks, validation stall, and memory >90%.

Training metrics and artifacts may be added only after the corresponding experiment
is approved. Record each Job URL, hardware, timeout/cost cap, command, exit code,
and artifact persistence check. The final logbook is intended to include raw-data
figure cells, claim-specific artifacts, the reproduction bundle, an outcome-first
executive summary with Scope & Cost table, and a strict-polish Posterly poster.
Before any experiment output is synchronized, scan it for secrets and restricted
raw data and verify all artifact links.

## 11. Approval boundary

Approval of this plan would authorize only the next implementation/smoke/pilot
steps explicitly agreed by the user. Paid final training, creation of new Hub
resources, artifact uploads, and publication of experiment outputs or restricted
data remain separately gated if their measured cost or data-sharing implications
differ from this plan.
