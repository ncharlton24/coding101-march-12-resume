from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

TARGET_COLUMN = "actual"
LINE_COLUMN = "line"
DATE_COLUMN = "game_date"
PLAYER_COLUMN = "player"
STAT_COLUMN = "stat_type"
OPPONENT_COLUMN = "opponent"


class MissingColumnsError(ValueError):
    """Raised when the input dataset does not have required columns."""


@dataclass
class MeanDiffModel:
    global_mean_diff: float
    player_stat_mean_diff: dict[str, float]
    opponent_mean_diff: dict[str, float]
    blend_weight_player: float = 0.8
    blend_weight_opp: float = 0.2

    def predict_row(self, row: dict[str, str]) -> float:
        line = parse_float(row.get(LINE_COLUMN, "0"))
        key_player = make_player_stat_key(row)
        key_opp = row.get(OPPONENT_COLUMN, "UNKNOWN")

        player_diff = self.player_stat_mean_diff.get(key_player, self.global_mean_diff)
        opp_diff = self.opponent_mean_diff.get(key_opp, self.global_mean_diff)

        expected_diff = (
            self.blend_weight_player * player_diff
            + self.blend_weight_opp * opp_diff
        )
        return line + expected_diff

    def predict_rows(self, rows: List[dict[str, str]]) -> List[float]:
        return [self.predict_row(row) for row in rows]


def parse_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def make_player_stat_key(row: dict[str, str]) -> str:
    return f"{row.get(PLAYER_COLUMN, 'UNKNOWN')}::{row.get(STAT_COLUMN, 'points')}"


def validate_columns(headers: Iterable[str], required: Iterable[str]) -> None:
    missing = [column for column in required if column not in headers]
    if missing:
        raise MissingColumnsError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )


def read_csv_rows(path: Path) -> List[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def sort_rows(rows: List[dict[str, str]]) -> List[dict[str, str]]:
    if not rows or DATE_COLUMN not in rows[0]:
        return rows

    def parse_date(row: dict[str, str]) -> datetime:
        raw = row.get(DATE_COLUMN, "")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return datetime.min

    return sorted(rows, key=parse_date)


def fit_model(rows: List[dict[str, str]]) -> MeanDiffModel:
    global_diffs: List[float] = []
    player_stat_diffs = defaultdict(list)
    opponent_diffs = defaultdict(list)

    for row in rows:
        actual = parse_float(row.get(TARGET_COLUMN, ""), fallback=float("nan"))
        line = parse_float(row.get(LINE_COLUMN, ""), fallback=float("nan"))
        if actual != actual or line != line:
            continue

        diff = actual - line
        global_diffs.append(diff)
        player_stat_diffs[make_player_stat_key(row)].append(diff)
        opponent_diffs[row.get(OPPONENT_COLUMN, "UNKNOWN")].append(diff)

    if len(global_diffs) < 10:
        raise ValueError("Need at least 10 valid rows to train a model.")

    global_mean = sum(global_diffs) / len(global_diffs)
    player_means = {k: sum(v) / len(v) for k, v in player_stat_diffs.items()}
    opponent_means = {k: sum(v) / len(v) for k, v in opponent_diffs.items()}

    return MeanDiffModel(
        global_mean_diff=global_mean,
        player_stat_mean_diff=player_means,
        opponent_mean_diff=opponent_means,
    )


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    squared = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    return (sum(squared) / len(squared)) ** 0.5


def directional_accuracy(y_true: List[float], y_pred: List[float], lines: List[float]) -> float:
    hits = 0
    for actual, pred, line in zip(y_true, y_pred, lines):
        if (actual >= line) == (pred >= line):
            hits += 1
    return hits / len(y_true)


def train_and_evaluate(rows: List[dict[str, str]], holdout_ratio: float = 0.2) -> tuple[MeanDiffModel, dict]:
    rows = sort_rows(rows)
    split_idx = max(1, min(int(len(rows) * (1 - holdout_ratio)), len(rows) - 1))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    model = fit_model(train_rows)

    y_test = [parse_float(r[TARGET_COLUMN]) for r in test_rows]
    preds = model.predict_rows(test_rows)
    lines = [parse_float(r[LINE_COLUMN]) for r in test_rows]

    metrics = {
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "mae": mae(y_test, preds),
        "rmse": rmse(y_test, preds),
        "directional_accuracy": directional_accuracy(y_test, preds, lines),
    }
    return model, metrics


def save_model(model: MeanDiffModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: Path) -> MeanDiffModel:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_predictions(rows: List[dict[str, str]], preds: List[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []

    for row, pred in zip(rows, preds):
        line = parse_float(row.get(LINE_COLUMN, "0"))
        edge = pred - line
        out = dict(row)
        out["predicted"] = f"{pred:.3f}"
        out["edge"] = f"{edge:.3f}"
        out["pick"] = "OVER" if edge >= 0 else "UNDER"
        out_rows.append(out)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)




def format_picks_table(rows: List[dict[str, str]], preds: List[float]) -> str:
    enriched = []
    for row, pred in zip(rows, preds):
        line = parse_float(row.get(LINE_COLUMN, "0"))
        edge = pred - line
        enriched.append({
            "game_date": row.get("game_date", ""),
            "player": row.get("player", ""),
            "stat_type": row.get("stat_type", ""),
            "line": line,
            "predicted": pred,
            "edge": edge,
            "pick": "OVER" if edge >= 0 else "UNDER",
        })

    enriched.sort(key=lambda r: abs(r["edge"]), reverse=True)

    header = "| Date | Player | Prop | Line | Predicted | Edge | Pick |\n|---|---|---|---:|---:|---:|---|"
    lines = [header]
    for item in enriched:
        lines.append(
            f"| {item['game_date']} | {item['player']} | {item['stat_type']} | "
            f"{item['line']:.1f} | {item['predicted']:.2f} | {item['edge']:+.2f} | {item['pick']} |"
        )

    return "\n".join(lines)


def write_markdown_report(rows: List[dict[str, str]], preds: List[float], out_path: Path) -> None:
    report = ["# Daily NBA Prop Picks", "", format_picks_table(rows, preds), ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run an NBA player prop prediction model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model on historical props data.")
    train_parser.add_argument("--train-csv", required=True, type=Path)
    train_parser.add_argument("--model-out", required=True, type=Path)
    train_parser.add_argument("--metrics-out", type=Path)

    predict_parser = subparsers.add_parser("predict", help="Generate over/under picks.")
    predict_parser.add_argument("--model-path", required=True, type=Path)
    predict_parser.add_argument("--input-csv", required=True, type=Path)
    predict_parser.add_argument("--output-csv", required=True, type=Path)

    report_parser = subparsers.add_parser("report", help="Generate readable markdown picks report.")
    report_parser.add_argument("--model-path", required=True, type=Path)
    report_parser.add_argument("--input-csv", required=True, type=Path)
    report_parser.add_argument("--output-md", required=True, type=Path)

    args = parser.parse_args()

    if args.command == "train":
        rows = read_csv_rows(args.train_csv)
        if not rows:
            raise ValueError("Training CSV is empty.")
        validate_columns(rows[0].keys(), [TARGET_COLUMN, LINE_COLUMN, PLAYER_COLUMN, STAT_COLUMN, OPPONENT_COLUMN])

        model, metrics = train_and_evaluate(rows)
        save_model(model, args.model_out)

        if args.metrics_out:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            print(json.dumps(metrics, indent=2))

    if args.command == "predict":
        rows = read_csv_rows(args.input_csv)
        if not rows:
            raise ValueError("Prediction CSV is empty.")
        validate_columns(rows[0].keys(), [LINE_COLUMN, PLAYER_COLUMN, STAT_COLUMN, OPPONENT_COLUMN])

        model = load_model(args.model_path)
        preds = model.predict_rows(rows)
        write_predictions(rows, preds, args.output_csv)
        print(f"Wrote {len(rows)} predictions to {args.output_csv}")

    if args.command == "report":
        rows = read_csv_rows(args.input_csv)
        if not rows:
            raise ValueError("Report CSV is empty.")
        validate_columns(rows[0].keys(), [LINE_COLUMN, PLAYER_COLUMN, STAT_COLUMN, OPPONENT_COLUMN])

        model = load_model(args.model_path)
        preds = model.predict_rows(rows)
        write_markdown_report(rows, preds, args.output_md)
        print(format_picks_table(rows, preds))
        print(f"Wrote readable report to {args.output_md}")


if __name__ == "__main__":
    main()
