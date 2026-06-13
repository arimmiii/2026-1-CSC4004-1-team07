# Clickbait Visualization Notes

Generated artifacts:
- `presentation_model_comparison.svg`: final presentation slide visual
- `professor_report_summary.svg`: professor-facing summary visual
- `roberta_base_hparam_comparison.svg`: baseline vs Run A vs Run B comparison

## Dataset
- train: 200,000 samples (clickbait 100,913, non-clickbait 99,087)
- valid: 25,000 samples (clickbait 12,614, non-clickbait 12,386)
- test: 25,000 samples (clickbait 12,614, non-clickbait 12,386)

## Final metrics
- KLUE RoBERTa base: valid macro F1 0.8126, test macro F1 0.8145, test accuracy 0.8146
- DeBERTa base: valid macro F1 0.7763, test macro F1 0.7792, test accuracy 0.7802
- Logistic Regression: valid macro F1 0.6790, test macro F1 0.6814, test accuracy 0.6814
- Linear SVM: valid macro F1 0.6766, test macro F1 0.6773, test accuracy 0.6773

## Why RoBERTa-base numbers differ
- `0.8241`: fast subset validation result. This was a quick feasibility run on `120k/20k/20k`, `epochs=1`, `max_length=96`.
- `0.8194`: full-test check of the same fast-run checkpoint. This confirmed the fast result was not only a validation artifact.
- `0.8028`: original baseline rerun. This used `epochs=2`, `max_length=128`, and `learning_rate=2e-5` on the final shared split.
- `0.8145`: current best rerun (Run A). This kept the same split and length, but lowered the learning rate to `1e-5`.
- The latest reported score now uses Run A because it outperformed both the original baseline and Run B on the same reduced split.

## KLUE RoBERTa base rerun comparison
- Baseline: lr `2e-5`, epochs `2`, max_length `128`, test macro F1 `0.8028`
- Run A: lr `1e-5`, epochs `2`, max_length `128`, test macro F1 `0.8145`
- Run B: lr `2e-5`, epochs `3`, max_length `128`, test macro F1 `0.8054`
