# Claim - DRIVE Full Model versus No-SSTM


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_22edbcf62bfa", "created_at": "2026-07-17T20:46:36+00:00", "title": "Completed three-seed validation comparison"}
-->
## Result scope

All six DRIVE runs completed successfully: Full S2M-Net and No-SSTM for
seeds 42, 7, and 123. The comparison below reports **validation results
only**. No test results were produced, and every run recorded
`hidden_test_accessed: false`.

The authoritative local audit is `docs/drive_multiseed_report.md`. The six
runs used the same DRIVE split, preprocessing, training budget, and evaluation
protocol. They differed only in seed and in whether the SSTM architecture was
enabled.

## Selected-checkpoint validation Dice

The primary metric is validation supplied-FOV macro hard Dice at threshold
0.5. Values are the mean and sample standard deviation across the three
seeds.

| Architecture | Validation Dice, mean ± sample SD |
|---|---:|
| Full S2M-Net | 0.765521754193 ± 0.003772260504 |
| No-SSTM | 0.766222204402 ± 0.005348764316 |

For each run, model selection maximized the primary validation metric. The
selected checkpoints occurred between epochs 13 and 34.

## Paired validation differences

| Seed | Full − No-SSTM validation Dice |
|---:|---:|
| 42 | +0.006318588075 |
| 7 | −0.004371089648 |
| 123 | −0.004048849056 |
| **Mean paired difference** | **−0.000700450210** |

## Model size and training time

SSTM adds 287,824 parameters: Full has 4,791,544 parameters and No-SSTM has
4,503,720, an increase of about 6.39% relative to No-SSTM. Based on the
recorded mean wall-clock time across these runs, Full required about 8.3% more
training time than No-SSTM.

## Limitations and conclusion

The experiment has only three paired seeds and four validation images.
Checkpoint selection and reporting also use the same validation set. These
constraints materially limit the strength and generality of the result.

**Conclusion:** Under this experimental protocol, the three-seed validation
comparison does not show a stable Dice improvement from SSTM.

This is a conservative conclusion about this protocol and validation sample,
not a statement that SSTM is universally ineffective and not a universal
paper-claim verdict.
