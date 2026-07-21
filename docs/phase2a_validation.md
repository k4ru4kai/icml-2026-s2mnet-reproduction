# Phase 2A — Synthetic Full-Model Validation

## Purpose

Phase 2A comprises two lightweight diagnostics, each based on a single
synthetic input and requiring no access to data:

- `repro/diagnostics/phase2a_full_model_drive.py` builds the official Full
  Model and performs a forward pass to verify its shape, parameter count, and
  the numerical finiteness of its output and weights.
- `repro/diagnostics/phase2a_full_model_trainability.py` verifies training-graph
  connectivity with a backward pass and a single weight update.

## Verified configuration

Both tests use `train.build_model` and
`official_repo/configs/retinal.yaml`, with `256 × 256` RGB input and binary
`sigmoid` output. The resolved official defaults enable `MRF-SE` in 5 stages,
`SSTM` in 5 stages, and `BFP` in 4 stages with `soft` routing. The model
contains 4,766,008 parameters: 4,743,384 trainable and 22,624 non-trainable.

## Results

The forward pass used a `float32` input with shape `(1, 256, 256, 3)`,
constructed over the interval `[0, 1]`, and produced a `float32` output with
shape `(1, 256, 256, 1)`. The output and weights were finite, with no `NaN` or
`Inf` values, and the parameter partition was consistent.

The trainability test, using seed 42 and a synthetic binary mask, found 305
trainable variables and zero `None` gradients. All 4,743,384 gradient values
were finite, and the global norm was `0.261258841`. A single `Adam` update with
a learning rate of `1e-4` advanced the optimizer iteration to 1 and changed
all 305 trainable variables. The loss changed from `0.7063784` to
`0.701960444`; a decrease after a single step was not a requirement of the
test.

Binary Cross-Entropy was used solely as a simple, numerically stable
diagnostic loss on `sigmoid` probabilities. It is not the paper's
Morphology-Aware Loss, and it does not include the model's regularization
terms.

During the backward pass through the `SSTM` spectral path, TensorFlow emitted
warnings for `complex64 → float32` conversions that discard the imaginary
component. In this run, the warnings were not accompanied by missing
gradients, `NaN`, or `Inf` values, but this behavior should be kept in mind in
subsequent phases.

`OVERALL_STATUS=PASS`

## Validation limitations

These results demonstrate only the Full Model's buildability, numerical
finiteness, and differentiable connectivity on synthetic data. DRIVE was
neither accessed nor evaluated: it was not downloaded or loaded, and no
empirical results, segmentation metrics, or conclusions about performance on
DRIVE are therefore available.
