"""Evaluate predicted DelVG breakpoints against ground truth."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DelVGEvent:
    """Container for one DelVG deletion event."""

    genome_id: str
    start: int
    end: int


@dataclass(frozen=True)
class EvaluationResult:
    """Precision/Recall/F1 metrics and confusion counts."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


def _to_int(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    return int(float(text))


def _normalize_event(genome_id: str, start: int, end: int) -> DelVGEvent:
    left = min(start, end)
    right = max(start, end)
    return DelVGEvent(genome_id=genome_id, start=left, end=right)


def load_ground_truth_events(ground_truth_csv: Path) -> list[DelVGEvent]:
    """Load deletion events from ground truth CSV and exclude WT rows."""

    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {ground_truth_csv}")

    events: list[DelVGEvent] = []
    with ground_truth_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            deletion_size = _to_int(row.get("deletion_size"))
            if deletion_size in (None, 0):
                continue

            start = _to_int(row.get("deletion_start"))
            end = _to_int(row.get("deletion_end"))
            if start is None or end is None:
                continue

            raw_id = (row.get("sequence_id") or "").strip()
            genome_id = raw_id.split("|", 1)[0] if "|" in raw_id else raw_id
            events.append(_normalize_event(genome_id=genome_id, start=start, end=end))

    return events


def load_predicted_events(predicted_csv: Path) -> list[DelVGEvent]:
    """Load predicted deletion events from standardized prediction CSV."""

    if not predicted_csv.exists():
        raise FileNotFoundError(f"Predicted CSV not found: {predicted_csv}")

    events: list[DelVGEvent] = []
    with predicted_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            genome_id = (row.get("genome_id") or "").strip()
            if not genome_id:
                continue

            start = _to_int(row.get("predicted_start"))
            end = _to_int(row.get("predicted_end"))
            if start is None or end is None:
                continue
            events.append(_normalize_event(genome_id=genome_id, start=start, end=end))

    return events


def _is_match(predicted: DelVGEvent, truth: DelVGEvent, tolerance_window: int) -> bool:
    if predicted.genome_id != truth.genome_id:
        return False
    return abs(predicted.start - truth.start) <= tolerance_window and abs(predicted.end - truth.end) <= tolerance_window


def evaluate_predictions(
    predicted_events: list[DelVGEvent],
    truth_events: list[DelVGEvent],
    tolerance_window: int = 5,
) -> EvaluationResult:
    """Compute precision/recall/F1 using one-to-one event matching."""

    if tolerance_window < 0:
        raise ValueError("tolerance_window must be >= 0.")

    unmatched_truth = set(range(len(truth_events)))
    true_positives = 0

    for predicted in predicted_events:
        candidate_idx = None
        candidate_distance = None
        for idx in list(unmatched_truth):
            truth = truth_events[idx]
            if not _is_match(predicted, truth, tolerance_window):
                continue

            distance = abs(predicted.start - truth.start) + abs(predicted.end - truth.end)
            if candidate_distance is None or distance < candidate_distance:
                candidate_distance = distance
                candidate_idx = idx

        if candidate_idx is not None:
            true_positives += 1
            unmatched_truth.remove(candidate_idx)

    false_positives = len(predicted_events) - true_positives
    false_negatives = len(truth_events) - true_positives

    precision = true_positives / len(predicted_events) if predicted_events else 0.0
    recall = true_positives / len(truth_events) if truth_events else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return EvaluationResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )


def evaluate_from_csv(
    predicted_csv: Path,
    ground_truth_csv: Path,
    tolerance_window: int = 5,
) -> EvaluationResult:
    predicted = load_predicted_events(predicted_csv)
    truth = load_ground_truth_events(ground_truth_csv)
    return evaluate_predictions(predicted, truth, tolerance_window=tolerance_window)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted DelVGs against synthetic ground truth.")
    parser.add_argument(
        "--predicted-csv",
        type=Path,
        default=Path("output/predicted_delvgs.csv"),
        help="Path to standardized predicted DelVG CSV.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=Path("output/ground_truth.csv"),
        help="Path to synthetic ground-truth CSV.",
    )
    parser.add_argument(
        "--tolerance-window",
        type=int,
        default=5,
        help="Allowed breakpoint deviation in bp (+/- window) for TP matching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_from_csv(
        predicted_csv=args.predicted_csv,
        ground_truth_csv=args.ground_truth_csv,
        tolerance_window=args.tolerance_window,
    )
    print(f"Tolerance Window: +/-{args.tolerance_window} bp")
    print(f"True Positives: {result.true_positives}")
    print(f"False Positives: {result.false_positives}")
    print(f"False Negatives: {result.false_negatives}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall: {result.recall:.4f}")
    print(f"F1-Score: {result.f1_score:.4f}")


if __name__ == "__main__":
    main()
