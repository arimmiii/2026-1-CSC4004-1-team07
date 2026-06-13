from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "visualization" / "output"


@dataclass
class ModelResult:
    label: str
    family: str
    valid_accuracy: float
    valid_macro_f1: float
    valid_weighted_f1: float
    test_accuracy: float
    test_macro_f1: float
    test_weighted_f1: float
    note: str


@dataclass
class StageResult:
    label: str
    split_text: str
    settings_text: str
    metric_label: str
    metric_value: float
    detail_text: str


@dataclass
class HyperparamRun:
    run_id: str
    label: str
    learning_rate: str
    epochs: int
    max_length: int
    valid_macro_f1: float
    test_macro_f1: float
    note: str


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def normalize_metrics(label: str, family: str, path: Path, note: str) -> ModelResult:
    payload = read_json(path)
    valid = payload["valid"]
    test = payload["test"]

    def pull(split: dict, plain: str, prefixed: str) -> float:
        if plain in split:
            return float(split[plain])
        return float(split[prefixed])

    return ModelResult(
        label=label,
        family=family,
        valid_accuracy=pull(valid, "accuracy", "valid_accuracy"),
        valid_macro_f1=pull(valid, "macro_f1", "valid_macro_f1"),
        valid_weighted_f1=pull(valid, "weighted_f1", "valid_weighted_f1"),
        test_accuracy=pull(test, "accuracy", "test_accuracy"),
        test_macro_f1=pull(test, "macro_f1", "test_macro_f1"),
        test_weighted_f1=pull(test, "weighted_f1", "test_weighted_f1"),
        note=note,
    )


def collect_results() -> list[ModelResult]:
    return [
        normalize_metrics(
            "Linear SVM",
            "Linear baseline",
            ROOT / "clickbait_svm_linear" / "models" / "metrics.json",
            "Fast CPU baseline",
        ),
        normalize_metrics(
            "Logistic Regression",
            "Linear baseline",
            ROOT / "clickbait_logreg_tfidf" / "metrics.json",
            "TF-IDF reference baseline",
        ),
        normalize_metrics(
            "DeBERTa base",
            "Transformer",
            ROOT
            / "clickbait_transformer_finetune"
            / "models"
            / "deberta_v3_base_title_body_run1"
            / "metrics.json",
            "General transformer comparison run",
        ),
        normalize_metrics(
            "KLUE RoBERTa base",
            "Transformer",
            ROOT
            / "clickbait_transformer_finetune"
            / "models"
            / "klue_roberta_base_runA_lr1e5_ep2_len128"
            / "metrics.json",
            "Selected final model (Run A)",
        ),
    ]


def split_stats(csv_path: Path) -> tuple[int, int, int]:
    total = 0
    positives = 0
    negatives = 0
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            label = int(row["label"])
            if label == 1:
                positives += 1
            else:
                negatives += 1
    return total, positives, negatives


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def fmt(value: float) -> str:
    return f"{value:.4f}"


def wrap_text(text: str, limit: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        tentative = word if not current else f"{current} {word}"
        if len(tentative) <= limit:
            current = tentative
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build_presentation_svg(results: list[ModelResult]) -> str:
    width = 1600
    height = 900
    left = 110
    bar_x = 610
    bar_w = 650
    top = 220
    row_h = 135
    max_score = 0.85

    palette = {
        "bg0": "#f4efe6",
        "bg1": "#efe3cf",
        "ink": "#1f1a17",
        "muted": "#6d5e53",
        "svm": "#8d6a9f",
        "logreg": "#d2875a",
        "deberta": "#4f7cac",
        "base": "#1f6f78",
        "large": "#c44536",
        "grid": "#d4c6b4",
    }
    color_map = {
        "Linear SVM": palette["svm"],
        "Logistic Regression": palette["logreg"],
        "DeBERTa base": palette["deberta"],
        "KLUE RoBERTa base": palette["base"],
    }

    sorted_results = sorted(results, key=lambda item: item.test_macro_f1, reverse=True)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{palette["bg0"]}"/>',
        f'<rect x="40" y="40" width="{width - 80}" height="{height - 80}" rx="28" fill="{palette["bg1"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
        f'<text x="100" y="120" font-size="54" font-weight="700" fill="{palette["ink"]}" font-family="Georgia, Times New Roman, serif">Clickbait Detection Model Comparison</text>',
        f'<text x="100" y="170" font-size="24" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Final rerun on the same reduced split (train 200k / valid 25k / test 25k)</text>',
        f'<text x="{bar_x}" y="210" font-size="20" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Test Macro F1</text>',
    ]

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        x = bar_x + (tick / max_score) * bar_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 20}" x2="{x:.1f}" y2="{top + row_h * len(sorted_results) - 10}" stroke="{palette["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top - 35}" text-anchor="middle" font-size="16" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{tick:.1f}</text>'
        )

    for index, result in enumerate(sorted_results):
        y = top + index * row_h
        bar_len = (result.test_macro_f1 / max_score) * bar_w
        color = color_map[result.label]
        parts.extend(
            [
                f'<text x="{left}" y="{y + 28}" font-size="28" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{escape(result.label)}</text>',
                f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="36" rx="18" fill="#f8f3ec"/>',
                f'<rect x="{bar_x}" y="{y}" width="{bar_len:.1f}" height="36" rx="18" fill="{color}"/>',
                f'<text x="{min(bar_x + bar_len + 18, width - 220):.1f}" y="{y + 26}" font-size="22" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{fmt(result.test_macro_f1)}</text>',
                f'<text x="{left}" y="{y + 58}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{escape(result.note)}</text>',
                f'<text x="{left}" y="{y + 84}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Valid Macro F1 {fmt(result.valid_macro_f1)}  /  Test Accuracy {fmt(result.test_accuracy)}</text>',
            ]
        )

    best = sorted_results[0]
    gap = best.test_macro_f1 - sorted_results[1].test_macro_f1
    parts.extend(
        [
            f'<rect x="100" y="760" width="1400" height="96" rx="22" fill="{palette["ink"]}"/>',
            f'<text x="130" y="813" font-size="30" font-weight="700" fill="#fff9f2" font-family="Arial, Helvetica, sans-serif">Takeaway</text>',
            f'<text x="320" y="813" font-size="24" fill="#fff9f2" font-family="Arial, Helvetica, sans-serif">{escape(best.label)} is the strongest final model with Test Macro F1 {fmt(best.test_macro_f1)}.</text>',
            f'<text x="320" y="843" font-size="20" fill="#e6d9c8" font-family="Arial, Helvetica, sans-serif">It leads the next baseline by {gap:.4f} points and is the recommended deployment candidate.</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts)


def build_report_svg(results: list[ModelResult], split_counts: dict[str, tuple[int, int, int]]) -> str:
    width = 1800
    height = 1360
    palette = {
        "paper": "#fbfaf7",
        "panel": "#f1ece3",
        "ink": "#181714",
        "muted": "#635b52",
        "grid": "#d6cdc0",
        "svm": "#6b4f8a",
        "logreg": "#ba6f3b",
        "deberta": "#4f7cac",
        "base": "#0f6670",
        "large": "#b13a2f",
        "good": "#2f7d32",
        "warn": "#b06a00",
        "bad": "#9f2d2d",
    }
    color_map = {
        "Linear SVM": palette["svm"],
        "Logistic Regression": palette["logreg"],
        "DeBERTa base": palette["deberta"],
        "KLUE RoBERTa base": palette["base"],
    }

    sorted_results = sorted(results, key=lambda item: item.test_macro_f1, reverse=True)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{palette["paper"]}"/>',
        f'<text x="70" y="90" font-size="52" font-weight="700" fill="{palette["ink"]}" font-family="Georgia, Times New Roman, serif">Clickbait Detection Evaluation Summary</text>',
        f'<text x="70" y="132" font-size="24" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Professor-facing summary of data setup, rerun metrics, and model selection rationale</text>',
        f'<rect x="60" y="180" width="450" height="290" rx="20" fill="{palette["panel"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
        f'<text x="90" y="235" font-size="30" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">Dataset Split</text>',
    ]

    y = 285
    for split_name, counts in [("Train", split_counts["train"]), ("Valid", split_counts["valid"]), ("Test", split_counts["test"])]:
        total, positive, negative = counts
        parts.append(
            f'<text x="90" y="{y}" font-size="23" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{split_name}: {total:,} samples</text>'
        )
        parts.append(
            f'<text x="90" y="{y + 32}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">clickbait {positive:,} / non-clickbait {negative:,}</text>'
        )
        y += 74

    parts.extend(
        [
            f'<rect x="540" y="180" width="1200" height="540" rx="20" fill="{palette["panel"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
            f'<text x="570" y="235" font-size="30" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">Model Metrics</text>',
        ]
    )

    table_x = 570
    table_y = 280
    col_x = [table_x, 930, 1125, 1290, 1445, 1590]
    headers = ["Model", "Family", "Valid F1", "Test F1", "Accuracy", "Decision"]
    for header, x in zip(headers, col_x):
        parts.append(
            f'<text x="{x}" y="{table_y}" font-size="20" font-weight="700" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{header}</text>'
        )

    row_y = table_y + 44
    for result in sorted_results:
        decision = "Adopt" if result.label == "KLUE RoBERTa base" else ("Reject" if "large" in result.label.lower() else "Baseline")
        decision_color = palette["good"] if decision == "Adopt" else (palette["bad"] if decision == "Reject" else palette["warn"])
        parts.append(f'<line x1="{table_x}" y1="{row_y + 12}" x2="1710" y2="{row_y + 12}" stroke="{palette["grid"]}" stroke-width="1"/>')
        parts.append(f'<circle cx="{table_x + 12}" cy="{row_y - 6}" r="8" fill="{color_map[result.label]}"/>')
        label_lines = wrap_text(result.label, 17)
        for offset, line in enumerate(label_lines[:2]):
            parts.append(
                f'<text x="{table_x + 30}" y="{row_y + offset * 22}" font-size="20" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>'
            )
        family_lines = wrap_text(result.family, 16)
        for offset, line in enumerate(family_lines[:2]):
            parts.append(
                f'<text x="{col_x[1]}" y="{row_y + offset * 20}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>'
            )
        parts.append(f'<text x="{col_x[2]}" y="{row_y}" font-size="19" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{fmt(result.valid_macro_f1)}</text>')
        parts.append(f'<text x="{col_x[3]}" y="{row_y}" font-size="19" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{fmt(result.test_macro_f1)}</text>')
        parts.append(f'<text x="{col_x[4]}" y="{row_y}" font-size="19" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{fmt(result.test_accuracy)}</text>')
        parts.append(f'<text x="{col_x[5]}" y="{row_y}" font-size="20" font-weight="700" fill="{decision_color}" font-family="Arial, Helvetica, sans-serif">{decision}</text>')
        row_y += 86

    interp_x = 60
    interp_y = 510
    interp_w = 450
    interp_h = 360
    parts.extend(
        [
            f'<rect x="{interp_x}" y="{interp_y}" width="{interp_w}" height="{interp_h}" rx="20" fill="{palette["panel"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
            f'<text x="{interp_x + 30}" y="{interp_y + 55}" font-size="30" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">Interpretation</text>',
        ]
    )
    interp_lines = [
        "1. KLUE RoBERTa base remains the strongest final model.",
        "2. DeBERTa base is competitive, but still below the KLUE model.",
        "3. Linear models remain useful low-cost baselines.",
        "Only the final shared-split rerun is used as the official transformer score.",
    ]
    line_y = interp_y + 105
    for idx, text in enumerate(interp_lines):
        font_size = 20 if idx < 3 else 18
        color = palette["ink"] if idx < 3 else palette["muted"]
        for wrapped in wrap_text(text, 46):
            parts.append(
                f'<text x="{interp_x + 30}" y="{line_y}" font-size="{font_size}" fill="{color}" font-family="Arial, Helvetica, sans-serif">{escape(wrapped)}</text>'
            )
            line_y += 28 if idx < 3 else 24
        line_y += 10

    chart_x = 560
    chart_y = 860
    chart_w = 1160
    chart_h = 340
    parts.extend(
        [
            f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" rx="20" fill="{palette["panel"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
            f'<text x="{chart_x + 30}" y="{chart_y + 50}" font-size="30" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">Validation-to-Test Stability</text>',
        ]
    )

    base_y = chart_y + 260
    for idx, result in enumerate(sorted_results):
        x = chart_x + 80 + idx * 270
        scale = 220 / 0.85
        valid_h = result.valid_macro_f1 * scale
        test_h = result.test_macro_f1 * scale
        parts.append(f'<line x1="{x - 20}" y1="{base_y}" x2="{x + 180}" y2="{base_y}" stroke="{palette["grid"]}" stroke-width="1"/>')
        parts.append(f'<rect x="{x}" y="{base_y - valid_h:.1f}" width="56" height="{valid_h:.1f}" rx="8" fill="{color_map[result.label]}" opacity="0.45"/>')
        parts.append(f'<rect x="{x + 70}" y="{base_y - test_h:.1f}" width="56" height="{test_h:.1f}" rx="8" fill="{color_map[result.label]}"/>')
        parts.append(f'<text x="{x + 28}" y="{base_y + 30}" text-anchor="middle" font-size="16" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Valid</text>')
        parts.append(f'<text x="{x + 98}" y="{base_y + 30}" text-anchor="middle" font-size="16" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Test</text>')
        label_lines = wrap_text(result.label, 14)
        for offset, line in enumerate(label_lines[:2]):
            parts.append(
                f'<text x="{x + 63}" y="{base_y + 58 + offset * 18}" text-anchor="middle" font-size="15" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def build_summary_markdown(results: list[ModelResult], split_counts: dict[str, tuple[int, int, int]]) -> str:
    lines = [
        "# Clickbait Visualization Notes",
        "",
        "Generated artifacts:",
        "- `presentation_model_comparison.svg`: final presentation slide visual",
        "- `professor_report_summary.svg`: professor-facing summary visual",
        "- `roberta_base_hparam_comparison.svg`: baseline vs Run A vs Run B comparison",
        "",
        "## Dataset",
    ]
    for split_name in ("train", "valid", "test"):
        total, positive, negative = split_counts[split_name]
        lines.append(f"- {split_name}: {total:,} samples (clickbait {positive:,}, non-clickbait {negative:,})")

    lines.extend(["", "## Final metrics"])
    for result in sorted(results, key=lambda item: item.test_macro_f1, reverse=True):
        lines.append(
            f"- {result.label}: valid macro F1 {fmt(result.valid_macro_f1)}, "
            f"test macro F1 {fmt(result.test_macro_f1)}, test accuracy {fmt(result.test_accuracy)}"
        )

    lines.extend(
        [
            "",
            "## Why RoBERTa-base numbers differ",
            "- `0.8241`: fast subset validation result. This was a quick feasibility run on `120k/20k/20k`, `epochs=1`, `max_length=96`.",
            "- `0.8194`: full-test check of the same fast-run checkpoint. This confirmed the fast result was not only a validation artifact.",
            "- `0.8028`: original baseline rerun. This used `epochs=2`, `max_length=128`, and `learning_rate=2e-5` on the final shared split.",
            "- `0.8145`: current best rerun (Run A). This kept the same split and length, but lowered the learning rate to `1e-5`.",
            "- The latest reported score now uses Run A because it outperformed both the original baseline and Run B on the same reduced split.",
            "",
            "## KLUE RoBERTa base rerun comparison",
            "- Baseline: lr `2e-5`, epochs `2`, max_length `128`, test macro F1 `0.8028`",
            "- Run A: lr `1e-5`, epochs `2`, max_length `128`, test macro F1 `0.8145`",
            "- Run B: lr `2e-5`, epochs `3`, max_length `128`, test macro F1 `0.8054`",
        ]
    )

    return "\n".join(lines) + "\n"


def build_roberta_note_markdown() -> str:
    return "\n".join(
        [
            "# KLUE RoBERTa Base Result Interpretation",
            "",
            "## Why there are three different numbers",
            "- `82.41` (`0.8241` Macro F1): fast subset validation result.",
            "- `81.94` (`0.8194` Macro F1): full-test check of the same fast checkpoint.",
            "- `80.28` (`0.8028` Macro F1): original baseline rerun test result.",
            "- `81.45` (`0.8145` Macro F1): current best rerun result from Run A.",
            "",
            "## What changed between runs",
            "- Exploratory checkpoints with `0.8241` and `0.8194` were kept only for internal validation and are not shown as final timeline stages.",
            "- Fast check: `train 120k / valid 20k / test 20k`",
            "- Fast check: `epochs=1`, `max_length=96`, `batch_size=8`, `learning_rate=2e-5`",
            "- Baseline rerun: `train 200k / valid 25k / test 25k`",
            "- Baseline rerun: `epochs=2`, `max_length=128`, `batch_size=8`, `learning_rate=2e-5`",
            "- Run A rerun: `train 200k / valid 25k / test 25k`",
            "- Run A rerun: `epochs=2`, `max_length=128`, `batch_size=8`, `learning_rate=1e-5`",
            "",
            "## Why the final score is lower",
            "- The fast run was designed for quick feasibility checking, not final reporting.",
            "- The final rerun used a stricter shared split and a reporting-grade configuration.",
            "- Therefore the final reported score prioritizes reproducibility and fair comparison over the single highest exploratory value.",
            "",
            "## Presentation-ready wording",
            "- We first used a fast subset setting to verify that the transformer family clearly outperformed the linear baselines.",
            "- After confirming feasibility, we reran KLUE RoBERTa base on the final shared split and then compared multiple hyperparameter settings on that same split.",
            "- Run A became the new official result because it improved over the earlier baseline rerun while keeping the same data split and input length.",
        ]
    ) + "\n"


def roberta_base_stages() -> list[StageResult]:
    return [
        StageResult(
            label="Linear baseline",
            split_text="Reduced split 200k / 25k / 25k",
            settings_text="TF-IDF + Logistic Regression",
            metric_label="Test Macro F1",
            metric_value=0.6814,
            detail_text="Cheap baseline used to judge whether the transformer is worth deploying.",
        ),
        StageResult(
            label="Run A rerun",
            split_text="Reduced split 200k / 25k / 25k",
            settings_text="klue/roberta-base, max_length 128, epochs 2, batch 8, lr 1e-5",
            metric_label="Test Macro F1",
            metric_value=0.8145,
            detail_text="Current adopted setting. Same reduced split as the baseline rerun, but lower learning rate improved the final test score.",
        ),
    ]


def build_roberta_timeline_svg() -> str:
    stages = roberta_base_stages()
    width = 1800
    height = 1120
    palette = {
        "paper": "#f7f3eb",
        "panel": "#efe6d7",
        "ink": "#1f1b17",
        "muted": "#6f6358",
        "grid": "#d8ccbc",
        "accent": "#0f6670",
        "accent_soft": "#8ec5c0",
        "baseline": "#b5773b",
        "warn": "#b13a2f",
        "good": "#2f7d32",
    }

    chart_left = 120
    chart_right = 1650
    chart_top = 220
    chart_bottom = 520
    max_score = 0.85
    min_score = 0.65

    def x_pos(index: int) -> float:
        if len(stages) == 1:
            return float(chart_left)
        return chart_left + index * ((chart_right - chart_left) / (len(stages) - 1))

    def y_pos(score: float) -> float:
        usable = chart_bottom - chart_top
        ratio = (score - min_score) / (max_score - min_score)
        return chart_bottom - (ratio * usable)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{palette["paper"]}"/>',
        f'<text x="90" y="95" font-size="52" font-weight="700" fill="{palette["ink"]}" font-family="Georgia, Times New Roman, serif">KLUE RoBERTa Base Experiment Timeline</text>',
        f'<text x="90" y="136" font-size="24" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Shows the baseline-to-final rerun change that was used in the final presentation and report</text>',
        f'<rect x="70" y="170" width="1660" height="740" rx="24" fill="{palette["panel"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
    ]

    for tick in [0.65, 0.70, 0.75, 0.80, 0.85]:
        y = y_pos(tick)
        parts.append(f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="{palette["grid"]}" stroke-width="1"/>')
        parts.append(f'<text x="{chart_left - 20}" y="{y + 6:.1f}" text-anchor="end" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{tick:.2f}</text>')

    for idx in range(len(stages) - 1):
        x1 = x_pos(idx)
        x2 = x_pos(idx + 1)
        y1 = y_pos(stages[idx].metric_value)
        y2 = y_pos(stages[idx + 1].metric_value)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{palette["accent"]}" stroke-width="6" stroke-linecap="round"/>')

    for idx, stage in enumerate(stages):
        x = x_pos(idx)
        y = y_pos(stage.metric_value)
        color = palette["baseline"] if idx == 0 else (palette["warn"] if "Final" in stage.label else palette["accent"])
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="{color}" stroke="#ffffff" stroke-width="4"/>',
                f'<text x="{x:.1f}" y="{y - 24:.1f}" text-anchor="middle" font-size="24" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{stage.metric_value:.4f}</text>',
            ]
        )

    note_y = 545
    card_y = 690
    card_w = 720
    gap = 60
    for idx, stage in enumerate(stages):
        x = 150 + idx * (card_w + gap)
        border = palette["good"] if stage.label == "Final rerun" else palette["grid"]
        title_color = palette["good"] if stage.label == "Final rerun" else palette["ink"]
        parts.extend(
            [
                f'<rect x="{x}" y="{card_y}" width="{card_w}" height="220" rx="18" fill="#fbf8f2" stroke="{border}" stroke-width="2"/>',
                f'<text x="{x + 24}" y="{card_y + 40}" font-size="26" font-weight="700" fill="{title_color}" font-family="Arial, Helvetica, sans-serif">{escape(stage.label)}</text>',
            ]
        )
        line_y = card_y + 74
        for line in wrap_text(stage.split_text, 60):
            parts.append(
                f'<text x="{x + 24}" y="{line_y}" font-size="17" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>'
            )
            line_y += 22
        line_y += 8
        for line in wrap_text(stage.settings_text, 78):
            parts.append(
                f'<text x="{x + 24}" y="{line_y}" font-size="16" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>'
            )
            line_y += 20
        line_y += 10
        parts.append(
            f'<text x="{x + 24}" y="{line_y}" font-size="21" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{escape(stage.metric_label)} {stage.metric_value:.4f}</text>'
        )
        line_y += 28
        for line in wrap_text(stage.detail_text, 82):
            parts.append(
                f'<text x="{x + 24}" y="{line_y}" font-size="15" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{escape(line)}</text>'
            )
            line_y += 20

    parts.extend(
        [
            f'<rect x="90" y="{note_y}" width="1620" height="96" rx="18" fill="{palette["ink"]}"/>',
            f'<text x="120" y="{note_y + 52}" font-size="28" font-weight="700" fill="#fffaf3" font-family="Arial, Helvetica, sans-serif">Interpretation</text>',
            f'<text x="360" y="{note_y + 42}" font-size="19" fill="#fffaf3" font-family="Arial, Helvetica, sans-serif">Exploratory fast runs existed, but they are not shown as standalone stages because they were only used to verify feasibility.</text>',
            f'<text x="360" y="{note_y + 68}" font-size="19" fill="#e6d9c8" font-family="Arial, Helvetica, sans-serif">The timeline keeps only the baseline and the current adopted rerun that improved the score on the same reduced split.</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts)


def roberta_hparam_runs() -> list[HyperparamRun]:
    return [
        HyperparamRun(
            run_id="Baseline",
            label="KLUE RoBERTa base",
            learning_rate="2e-5",
            epochs=2,
            max_length=128,
            valid_macro_f1=0.8007271614970889,
            test_macro_f1=0.802754869015891,
            note="Original final rerun used in the previous report.",
        ),
        HyperparamRun(
            run_id="Run A",
            label="KLUE RoBERTa base",
            learning_rate="1e-5",
            epochs=2,
            max_length=128,
            valid_macro_f1=0.812592238894515,
            test_macro_f1=0.8144898932077058,
            note="Best result so far. Lower learning rate improved stability.",
        ),
        HyperparamRun(
            run_id="Run B",
            label="KLUE RoBERTa base",
            learning_rate="2e-5",
            epochs=3,
            max_length=128,
            valid_macro_f1=0.8050706492492592,
            test_macro_f1=0.8054167070934601,
            note="Extra epoch gave only a small gain over the original baseline.",
        ),
    ]


def build_roberta_hparam_svg() -> str:
    runs = roberta_hparam_runs()
    width = 1800
    height = 980
    left = 120
    bar_x = 860
    bar_w = 700
    top = 250
    row_h = 180
    max_score = 0.85
    palette = {
        "paper": "#f6f1e7",
        "panel": "#efe5d6",
        "ink": "#1f1b17",
        "muted": "#6e6257",
        "grid": "#d7cab8",
        "baseline": "#1f6f78",
        "run_a": "#2f7d32",
        "run_b": "#c27b2f",
    }
    color_map = {
        "Baseline": palette["baseline"],
        "Run A": palette["run_a"],
        "Run B": palette["run_b"],
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{palette["paper"]}"/>',
        f'<rect x="40" y="40" width="{width - 80}" height="{height - 80}" rx="26" fill="{palette["panel"]}" stroke="{palette["grid"]}" stroke-width="2"/>',
        f'<text x="90" y="110" font-size="50" font-weight="700" fill="{palette["ink"]}" font-family="Georgia, Times New Roman, serif">KLUE RoBERTa base Hyperparameter Comparison</text>',
        f'<text x="90" y="150" font-size="24" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Same reduced split, comparing the original setting against Run A and Run B</text>',
        f'<text x="{bar_x}" y="225" font-size="20" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Test Macro F1</text>',
    ]

    for tick in [0.70, 0.75, 0.80, 0.85]:
        x = bar_x + (tick / max_score) * bar_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 15}" x2="{x:.1f}" y2="{top + row_h * len(runs) - 35}" stroke="{palette["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top - 28}" text-anchor="middle" font-size="16" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{tick:.2f}</text>'
        )

    for idx, run in enumerate(runs):
        y = top + idx * row_h
        color = color_map[run.run_id]
        bar_len = (run.test_macro_f1 / max_score) * bar_w
        delta = run.test_macro_f1 - runs[0].test_macro_f1
        delta_text = "baseline" if idx == 0 else f"{delta:+.4f} vs baseline"
        parts.extend(
            [
                f'<text x="{left}" y="{y + 26}" font-size="28" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{run.run_id}</text>',
                f'<text x="{left}" y="{y + 56}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">lr {run.learning_rate} / epochs {run.epochs} / max_length {run.max_length}</text>',
                f'<text x="{left}" y="{y + 84}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">Valid F1 {fmt(run.valid_macro_f1)} / Test F1 {fmt(run.test_macro_f1)}</text>',
                f'<text x="{left}" y="{y + 112}" font-size="18" fill="{palette["muted"]}" font-family="Arial, Helvetica, sans-serif">{escape(run.note)}</text>',
                f'<rect x="{bar_x}" y="{y + 10}" width="{bar_w}" height="40" rx="20" fill="#fbf8f2"/>',
                f'<rect x="{bar_x}" y="{y + 10}" width="{bar_len:.1f}" height="40" rx="20" fill="{color}"/>',
                f'<text x="{min(bar_x + bar_len + 16, width - 180):.1f}" y="{y + 38}" font-size="22" font-weight="700" fill="{palette["ink"]}" font-family="Arial, Helvetica, sans-serif">{fmt(run.test_macro_f1)}</text>',
                f'<text x="{bar_x}" y="{y + 86}" font-size="18" fill="{color}" font-family="Arial, Helvetica, sans-serif">{delta_text}</text>',
            ]
        )

    parts.extend(
        [
            f'<rect x="90" y="810" width="1620" height="92" rx="18" fill="{palette["ink"]}"/>',
            f'<text x="120" y="862" font-size="28" font-weight="700" fill="#fffaf3" font-family="Arial, Helvetica, sans-serif">Summary</text>',
            f'<text x="330" y="852" font-size="20" fill="#fffaf3" font-family="Arial, Helvetica, sans-serif">Run A improved over the original baseline by lowering the learning rate from 2e-5 to 1e-5 while keeping epochs and max_length fixed.</text>',
            f'<text x="330" y="880" font-size="18" fill="#dfd2c3" font-family="Arial, Helvetica, sans-serif">Run B added one more epoch at the original learning rate, but its gain stayed below Run A.</text>',
        ]
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_results()
    split_counts = {
        "train": split_stats(ROOT / "data" / "train.csv"),
        "valid": split_stats(ROOT / "data" / "valid.csv"),
        "test": split_stats(ROOT / "data" / "test.csv"),
    }

    (OUT_DIR / "presentation_model_comparison.svg").write_text(
        build_presentation_svg(results),
        encoding="utf-8",
    )
    (OUT_DIR / "professor_report_summary.svg").write_text(
        build_report_svg(results, split_counts),
        encoding="utf-8",
    )
    (OUT_DIR / "roberta_base_experiment_timeline.svg").write_text(
        build_roberta_timeline_svg(),
        encoding="utf-8",
    )
    (OUT_DIR / "roberta_base_hparam_comparison.svg").write_text(
        build_roberta_hparam_svg(),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(
        build_summary_markdown(results, split_counts),
        encoding="utf-8",
    )
    (OUT_DIR / "roberta_base_experiment_notes.md").write_text(
        build_roberta_note_markdown(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
