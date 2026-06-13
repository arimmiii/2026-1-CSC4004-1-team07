from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = ROOT / "visualizations"
CHART_DIR = VIS_DIR / "charts"
DATA_DIR = VIS_DIR / "data"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value):
    if value is None:
        return None
    return float(value)


def parse_report(report):
    if isinstance(report, dict):
        return report
    rows = {}
    for line in str(report).splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            rows[parts[0]] = {
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1-score": float(parts[3]),
                "support": float(parts[4]),
            }
    return rows


def classification_5class_rows() -> list[dict]:
    return [
        {
            "group": "classification_5class",
            "model": "LogReg TF-IDF",
            "family": "baseline",
            "test_accuracy": 0.4620,
            "test_macro_f1": 0.4504,
            "test_top2_accuracy": None,
            "notes": "README benchmark",
        },
        {
            "group": "classification_5class",
            "model": "Linear SVM",
            "family": "baseline",
            "test_accuracy": 0.4360,
            "test_macro_f1": 0.4167,
            "test_top2_accuracy": None,
            "notes": "README benchmark",
        },
        {
            "group": "classification_5class",
            "model": "Kernel Approx SVM",
            "family": "baseline",
            "test_accuracy": 0.4460,
            "test_macro_f1": 0.4460,
            "test_top2_accuracy": None,
            "notes": "README benchmark",
        },
        {
            "group": "classification_5class",
            "model": "GBDT TF-IDF (XGB)",
            "family": "baseline",
            "test_accuracy": 0.3840,
            "test_macro_f1": 0.3434,
            "test_top2_accuracy": None,
            "notes": "README benchmark",
        },
    ]


def classification_3class_rows() -> list[dict]:
    rows = [
        {
            "group": "classification_3class",
            "model": "LogReg TF-IDF",
            "family": "baseline",
            "test_accuracy": 0.6520,
            "test_macro_f1": 0.6387,
            "test_top2_accuracy": 0.9200,
            "notes": "README benchmark",
        },
        {
            "group": "classification_3class",
            "model": "Linear SVM",
            "family": "baseline",
            "test_accuracy": 0.6740,
            "test_macro_f1": 0.6627,
            "test_top2_accuracy": 0.9320,
            "notes": "README benchmark",
        },
        {
            "group": "classification_3class",
            "model": "Kernel Approx SVM",
            "family": "baseline",
            "test_accuracy": 0.5720,
            "test_macro_f1": 0.5595,
            "test_top2_accuracy": 0.8600,
            "notes": "README benchmark",
        },
        {
            "group": "classification_3class",
            "model": "GBDT TF-IDF (XGB)",
            "family": "baseline",
            "test_accuracy": 0.6420,
            "test_macro_f1": 0.6333,
            "test_top2_accuracy": 0.9060,
            "notes": "README benchmark",
        },
    ]

    transformer = transformer_experiment_rows()
    best_test = max(transformer, key=lambda row: row["test_macro_f1"])
    rows.append(
        {
            "group": "classification_3class",
            "model": "Transformer Best Variant",
            "family": "transformer",
            "valid_accuracy": best_test["valid_accuracy"],
            "valid_macro_f1": best_test["valid_macro_f1"],
            "valid_top2_accuracy": best_test["valid_top2_accuracy"],
            "test_accuracy": best_test["test_accuracy"],
            "test_macro_f1": best_test["test_macro_f1"],
            "test_top2_accuracy": best_test["test_top2_accuracy"],
            "notes": f'Best transformer variant: {best_test["model"]}',
        }
    )
    return rows


def regression_rows() -> list[dict]:
    rows = [
        {
            "group": "regression_5class",
            "model": "Ridge TF-IDF",
            "family": "baseline",
            "test_rounded_accuracy": 0.2980,
            "test_rounded_macro_f1": 0.2158,
            "notes": "README benchmark",
        },
        {
            "group": "regression_5class",
            "model": "Linear SVR",
            "family": "baseline",
            "test_rounded_accuracy": 0.3260,
            "test_rounded_macro_f1": 0.2621,
            "notes": "README benchmark",
        },
        {
            "group": "regression_5class",
            "model": "GBDT TF-IDF Reg (XGB)",
            "family": "baseline",
            "test_rounded_accuracy": 0.3100,
            "test_rounded_macro_f1": 0.2503,
            "notes": "README benchmark",
        },
    ]
    metrics_path = ROOT / "regression" / "bias_transformer_regression" / "models" / "bias_transformer_regression_metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        rows.append(
            {
                "group": "regression_5class",
                "model": "Transformer Regression",
                "family": "transformer",
                "valid_mae": maybe_float(metrics["valid"]["mae"]),
                "valid_rmse": maybe_float(metrics["valid"]["rmse"]),
                "valid_rounded_accuracy": maybe_float(metrics["valid"]["rounded_accuracy"]),
                "valid_rounded_macro_f1": maybe_float(metrics["valid"]["rounded_macro_f1"]),
                "test_mae": maybe_float(metrics["test"]["mae"]),
                "test_rmse": maybe_float(metrics["test"]["rmse"]),
                "test_rounded_accuracy": maybe_float(metrics["test"]["rounded_accuracy"]),
                "test_rounded_macro_f1": maybe_float(metrics["test"]["rounded_macro_f1"]),
                "notes": "klue/roberta-base regression",
            }
        )
    return rows


def transformer_experiment_rows() -> list[dict]:
    metric_files = [
        (
            "Baseline",
            ROOT
            / "classification"
            / "bias_transformer_kopolitic_3class"
            / "models"
            / "bias_kopolitic_transformer_3class"
            / "bias_kopolitic_transformer_3class_metrics.json",
        ),
        (
            "Low LR + Longer",
            ROOT
            / "classification"
            / "bias_transformer_kopolitic_3class"
            / "models"
            / "bias_kopolitic_transformer_3class_low_lr_longer"
            / "bias_kopolitic_transformer_3class_low_lr_longer_metrics.json",
        ),
        (
            "Shorter Context + Bigger Batch",
            ROOT
            / "classification"
            / "bias_transformer_kopolitic_3class"
            / "models"
            / "checkpoints_shorter_context_bigger_batch"
            / "bias_kopolitic_transformer_3class_shorter_context_bigger_batch_metrics.json",
        ),
        (
            "Stronger Regularization",
            ROOT
            / "classification"
            / "bias_transformer_kopolitic_3class"
            / "models"
            / "checkpoints_stronger_regularization"
            / "bias_kopolitic_transformer_3class_stronger_regularization_metrics.json",
        ),
    ]
    rows = []
    for label, path in metric_files:
        metrics = load_json(path)
        report = parse_report(metrics["test"]["classification_report"])
        class2 = report.get("2", {})
        rows.append(
            {
                "group": "transformer_experiments_3class",
                "model": label,
                "family": "transformer",
                "path": str(path.relative_to(ROOT)),
                "base_model_name": metrics.get("base_model_name") or metrics.get("model_name"),
                "valid_accuracy": maybe_float(metrics["valid"]["accuracy"]),
                "valid_macro_f1": maybe_float(metrics["valid"]["macro_f1"]),
                "valid_top2_accuracy": maybe_float(metrics["valid"]["top2_accuracy"]),
                "test_accuracy": maybe_float(metrics["test"]["accuracy"]),
                "test_macro_f1": maybe_float(metrics["test"]["macro_f1"]),
                "test_top2_accuracy": maybe_float(metrics["test"]["top2_accuracy"]),
                "class2_precision": maybe_float(class2.get("precision")),
                "class2_recall": maybe_float(class2.get("recall")),
                "class2_f1": maybe_float(class2.get("f1-score")),
                "notes": label,
            }
        )
    return rows


def build_rows() -> list[dict]:
    return classification_5class_rows() + classification_3class_rows() + transformer_experiment_rows() + regression_rows()


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def svg_bar_chart(
    rows: list[dict],
    metric_keys: list[tuple[str, str, str]],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    width, left, right, top, row_h, bottom = 1480, 340, 80, 130, 78, 76
    chart_w = width - left - right
    height = top + row_h * len(rows) + bottom
    bg, fg, grid = "#f5f1e8", "#1b1b1b", "#d3c8b6"
    accent = ["#174c8f", "#d76a03", "#2d7a37"]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
        f'<text x="{left}" y="52" font-size="34" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#112b3c">{esc(title)}</text>',
        f'<text x="{left}" y="84" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="{fg}" opacity="0.78">{esc(subtitle)}</text>',
    ]
    for tick in range(0, 11):
        x = left + chart_w * tick / 10
        svg.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 10}" stroke="{grid}" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="{fg}" opacity="0.72">{tick/10:.1f}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        svg.append(f'<text x="{left - 20}" y="{y + 26}" text-anchor="end" font-size="20" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="{fg}">{esc(row["model"])}</text>')
        for metric_idx, (key, label, _) in enumerate(metric_keys):
            value = row.get(key)
            if value is None:
                continue
            bar_y = y + 8 + metric_idx * 18
            bar_w = chart_w * max(0.0, min(1.0, value))
            svg.append(f'<rect x="{left}" y="{bar_y}" width="{bar_w:.1f}" height="14" rx="7" fill="{accent[metric_idx % len(accent)]}" opacity="0.88"/>')
            text_x = min(left + bar_w + 10, width - right - 8)
            anchor = "start"
            if text_x >= width - right - 20:
                text_x = left + bar_w - 10
                anchor = "end"
            svg.append(f'<text x="{text_x:.1f}" y="{bar_y + 12}" text-anchor="{anchor}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="{fg}">{esc(label)} {fmt(value)}</text>')
    legend_y = height - 42
    for idx, (_, label, desc) in enumerate(metric_keys):
        lx = left + idx * 300
        svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="18" height="18" rx="4" fill="{accent[idx % len(accent)]}" opacity="0.88"/>')
        svg.append(f'<text x="{lx + 28}" y="{legend_y + 2}" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="{fg}">{esc(label)}: {esc(desc)}</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def svg_transformer_class2(rows: list[dict], out_path: Path) -> None:
    width, left, right, top, row_h, bottom = 1360, 320, 80, 130, 78, 76
    chart_w = width - left - right
    height = top + row_h * len(rows) + bottom
    bg, fg, grid = "#f5f1e8", "#1b1b1b", "#d3c8b6"
    colors = {"class2_precision": "#8b1e3f", "class2_recall": "#0f766e", "class2_f1": "#174c8f"}

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
        f'<text x="{left}" y="52" font-size="34" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="#112b3c">Transformer Class 2 Bottleneck Comparison</text>',
        f'<text x="{left}" y="84" font-size="18" font-family="Arial, Helvetica, sans-serif" fill="{fg}" opacity="0.78">Class 2 is the main ambiguity region. Precision recovery matters more than raw recall growth.</text>',
    ]
    for tick in range(0, 11):
        x = left + chart_w * tick / 10
        svg.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - bottom + 10}" stroke="{grid}" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-size="13" font-family="Arial, Helvetica, sans-serif" fill="{fg}" opacity="0.72">{tick/10:.1f}</text>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        svg.append(f'<text x="{left - 20}" y="{y + 26}" text-anchor="end" font-size="20" font-family="Arial, Helvetica, sans-serif" font-weight="700" fill="{fg}">{esc(row["model"])}</text>')
        for metric_idx, key in enumerate(["class2_precision", "class2_recall", "class2_f1"]):
            value = row.get(key)
            if value is None:
                continue
            bar_y = y + 8 + metric_idx * 18
            bar_w = chart_w * max(0.0, min(1.0, value))
            svg.append(f'<rect x="{left}" y="{bar_y}" width="{bar_w:.1f}" height="14" rx="7" fill="{colors[key]}" opacity="0.88"/>')
            label = key.replace("class2_", "")
            svg.append(f'<text x="{left + bar_w + 10:.1f}" y="{bar_y + 12}" font-size="14" font-family="Arial, Helvetica, sans-serif" fill="{fg}">{esc(label)} {fmt(value)}</text>')
    legend_y = height - 42
    legend = [
        ("class2_precision", "precision"),
        ("class2_recall", "recall"),
        ("class2_f1", "f1"),
    ]
    for idx, (key, label) in enumerate(legend):
        lx = left + idx * 200
        svg.append(f'<rect x="{lx}" y="{legend_y - 12}" width="18" height="18" rx="4" fill="{colors[key]}" opacity="0.88"/>')
        svg.append(f'<text x="{lx + 28}" y="{legend_y + 2}" font-size="15" font-family="Arial, Helvetica, sans-serif" fill="{fg}">Class 2 {esc(label)}</text>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def write_notes(transformer_rows: list[dict], out_path: Path) -> None:
    best_test = max(transformer_rows, key=lambda row: row["test_macro_f1"])
    best_top2 = max(transformer_rows, key=lambda row: row["test_top2_accuracy"])
    best_class2 = max(transformer_rows, key=lambda row: row["class2_f1"])

    lines = [
        "# Presentation Notes",
        "",
        "## Transformer 4-Experiment Summary",
        "",
        f"- Test accuracy / macro F1 최고: `{best_test['model']}` (`{best_test['test_accuracy']:.3f}`, `{best_test['test_macro_f1']:.3f}`)",
        f"- Test top2 최고: `{best_top2['model']}` (`{best_top2['test_top2_accuracy']:.3f}`)",
        f"- Class 2 F1 최고: `{best_class2['model']}` (`{best_class2['class2_f1']:.3f}`)",
        "",
        "## Interpretation",
        "",
        "- `Shorter Context + Bigger Batch`는 test generalization이 가장 좋았다.",
        "- `Stronger Regularization`은 정확도는 baseline과 비슷하지만, top2와 class 2 안정성이 가장 좋았다.",
        "- `Low LR + Longer`는 class 2 recall은 끌어올렸지만 precision 손실이 커서 전체 macro F1이 내려갔다.",
        "- baseline은 validation 수치는 가장 좋았지만 test에서는 `Shorter Context + Bigger Batch`가 앞섰다.",
        "",
        "## Slide Order",
        "",
        "- `charts/transformer_experiments_test.svg`: 발표 메인 비교",
        "- `charts/transformer_experiments_valid.svg`: 과적합/일반화 비교",
        "- `charts/transformer_experiments_class2.svg`: 중도(class 2) 병목 설명",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    transformer_rows = transformer_experiment_rows()

    write_csv(rows, DATA_DIR / "results_summary.csv")
    write_csv(transformer_rows, DATA_DIR / "transformer_experiment_summary.csv")

    svg_bar_chart(
        classification_5class_rows(),
        [
            ("test_accuracy", "Acc", "5-class accuracy"),
            ("test_macro_f1", "Macro F1", "5-class macro F1"),
        ],
        "Political Bias 5-Class Test Comparison",
        "CPU baseline models. Transformer section is evaluated separately in 3-class mode.",
        CHART_DIR / "classification_5class_test.svg",
    )
    svg_bar_chart(
        classification_3class_rows(),
        [
            ("test_accuracy", "Acc", "3-class accuracy"),
            ("test_macro_f1", "Macro F1", "3-class macro F1"),
            ("test_top2_accuracy", "Top2", "3-class top-2 accuracy"),
        ],
        "Political Bias 3-Class Test Comparison",
        "3-class remapping improves stability. Best transformer variant is included as a single comparison point.",
        CHART_DIR / "classification_3class_test.svg",
    )
    svg_bar_chart(
        regression_rows(),
        [
            ("test_rounded_accuracy", "Rounded Acc", "rounded 5-class accuracy"),
            ("test_rounded_macro_f1", "Rounded F1", "rounded 5-class macro F1"),
        ],
        "Political Bias Regression Test Comparison",
        "Regression outputs are rounded back to 1~5 for comparison.",
        CHART_DIR / "regression_5class_test.svg",
    )
    svg_bar_chart(
        transformer_rows,
        [
            ("test_accuracy", "Test Acc", "held-out accuracy"),
            ("test_macro_f1", "Test F1", "held-out macro F1"),
            ("test_top2_accuracy", "Top2", "held-out top-2 accuracy"),
        ],
        "Transformer 4-Experiment Test Comparison",
        "Main 발표용 차트. `Shorter Context + Bigger Batch` wins on held-out accuracy/F1, while `Stronger Regularization` leads top-2.",
        CHART_DIR / "transformer_experiments_test.svg",
    )
    svg_bar_chart(
        transformer_rows,
        [
            ("valid_accuracy", "Valid Acc", "validation accuracy"),
            ("valid_macro_f1", "Valid F1", "validation macro F1"),
            ("valid_top2_accuracy", "Valid Top2", "validation top-2 accuracy"),
        ],
        "Transformer 4-Experiment Validation Comparison",
        "Validation favors baseline slightly more than test does. This is useful for explaining generalization shift.",
        CHART_DIR / "transformer_experiments_valid.svg",
    )
    svg_transformer_class2(transformer_rows, CHART_DIR / "transformer_experiments_class2.svg")
    write_notes(transformer_rows, VIS_DIR / "presentation_notes.md")


if __name__ == "__main__":
    main()
