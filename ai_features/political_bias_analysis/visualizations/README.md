# Visualizations

발표용/보고용 시각화 산출물 폴더입니다.

생성 방식:
- `build_visualizations.py`가 현재 저장된 `metrics.json`과 README의 최종 수치를 바탕으로 SVG 차트와 요약 CSV를 생성
- `matplotlib` 없이 표준 라이브러리만 사용

주요 산출물:
- `charts/classification_5class_test.svg`
- `charts/classification_3class_test.svg`
- `charts/regression_5class_test.svg`
- `charts/kopolitic_class_report.svg`
- `charts/transformer_experiments_test.svg`
- `charts/transformer_experiments_valid.svg`
- `charts/transformer_experiments_class2.svg`
- `data/results_summary.csv`
- `data/transformer_experiment_summary.csv`
- `presentation_notes.md`

실행:

```bash
python3 ai_features/political_bias_analysis/visualizations/build_visualizations.py
```
