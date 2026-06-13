# KLUE RoBERTa Base Result Interpretation

## Why there are three different numbers
- `82.41` (`0.8241` Macro F1): fast subset validation result.
- `81.94` (`0.8194` Macro F1): full-test check of the same fast checkpoint.
- `80.28` (`0.8028` Macro F1): original baseline rerun test result.
- `81.45` (`0.8145` Macro F1): current best rerun result from Run A.

## What changed between runs
- Exploratory checkpoints with `0.8241` and `0.8194` were kept only for internal validation and are not shown as final timeline stages.
- Fast check: `train 120k / valid 20k / test 20k`
- Fast check: `epochs=1`, `max_length=96`, `batch_size=8`, `learning_rate=2e-5`
- Baseline rerun: `train 200k / valid 25k / test 25k`
- Baseline rerun: `epochs=2`, `max_length=128`, `batch_size=8`, `learning_rate=2e-5`
- Run A rerun: `train 200k / valid 25k / test 25k`
- Run A rerun: `epochs=2`, `max_length=128`, `batch_size=8`, `learning_rate=1e-5`

## Why the final score is lower
- The fast run was designed for quick feasibility checking, not final reporting.
- The final rerun used a stricter shared split and a reporting-grade configuration.
- Therefore the final reported score prioritizes reproducibility and fair comparison over the single highest exploratory value.

## Presentation-ready wording
- We first used a fast subset setting to verify that the transformer family clearly outperformed the linear baselines.
- After confirming feasibility, we reran KLUE RoBERTa base on the final shared split and then compared multiple hyperparameter settings on that same split.
- Run A became the new official result because it improved over the earlier baseline rerun while keeping the same data split and input length.
