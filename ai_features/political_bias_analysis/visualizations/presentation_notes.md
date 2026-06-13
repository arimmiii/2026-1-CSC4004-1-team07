# Presentation Notes

## Transformer 4-Experiment Summary

- Test accuracy / macro F1 최고: `Shorter Context + Bigger Batch` (`0.750`, `0.737`)
- Test top2 최고: `Stronger Regularization` (`0.936`)
- Class 2 F1 최고: `Stronger Regularization` (`0.656`)

## Interpretation

- `Shorter Context + Bigger Batch`는 test generalization이 가장 좋았다.
- `Stronger Regularization`은 정확도는 baseline과 비슷하지만, top2와 class 2 안정성이 가장 좋았다.
- `Low LR + Longer`는 class 2 recall은 끌어올렸지만 precision 손실이 커서 전체 macro F1이 내려갔다.
- baseline은 validation 수치는 가장 좋았지만 test에서는 `Shorter Context + Bigger Batch`가 앞섰다.

## Slide Order

- `charts/transformer_experiments_test.svg`: 발표 메인 비교
- `charts/transformer_experiments_valid.svg`: 과적합/일반화 비교
- `charts/transformer_experiments_class2.svg`: 중도(class 2) 병목 설명