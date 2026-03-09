"""Tests for DelVG evaluation logic (Checkpoint 4)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_from_csv  # noqa: E402


def _write_ground_truth(path: Path) -> None:
    rows = [
        {"sequence_id": "PB2|WT", "deletion_start": "NA", "deletion_end": "NA", "deletion_size": "0"},
        {"sequence_id": "PB2|DelVG_1", "deletion_start": "100", "deletion_end": "300", "deletion_size": "201"},
        {"sequence_id": "PB2|DelVG_2", "deletion_start": "500", "deletion_end": "800", "deletion_size": "301"},
        {"sequence_id": "PB2|DelVG_3", "deletion_start": "1200", "deletion_end": "1500", "deletion_size": "301"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_id", "deletion_start", "deletion_end", "deletion_size"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_predicted(path: Path, events: list[tuple[str, int, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["genome_id", "predicted_start", "predicted_end"])
        writer.writeheader()
        for genome_id, start, end in events:
            writer.writerow(
                {
                    "genome_id": genome_id,
                    "predicted_start": str(start),
                    "predicted_end": str(end),
                }
            )


def test_perfect_matches_have_f1_of_one(tmp_path: Path) -> None:
    truth_csv = tmp_path / "ground_truth.csv"
    pred_csv = tmp_path / "predicted.csv"
    _write_ground_truth(truth_csv)
    _write_predicted(
        pred_csv,
        [
            ("PB2", 100, 300),
            ("PB2", 500, 800),
            ("PB2", 1200, 1500),
        ],
    )

    result = evaluate_from_csv(pred_csv, truth_csv, tolerance_window=5)
    assert result.true_positives == 3
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0


def test_fuzzy_matches_within_tolerance_are_true_positives(tmp_path: Path) -> None:
    truth_csv = tmp_path / "ground_truth.csv"
    pred_csv = tmp_path / "predicted.csv"
    _write_ground_truth(truth_csv)
    _write_predicted(
        pred_csv,
        [
            ("PB2", 103, 297),  # within +/-5
            ("PB2", 496, 804),  # within +/-5
            ("PB2", 1198, 1502),  # within +/-5
        ],
    )

    result = evaluate_from_csv(pred_csv, truth_csv, tolerance_window=5)
    assert result.true_positives == 3
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0


def test_wrong_matches_outside_tolerance_have_zero_scores(tmp_path: Path) -> None:
    truth_csv = tmp_path / "ground_truth.csv"
    pred_csv = tmp_path / "predicted.csv"
    _write_ground_truth(truth_csv)
    _write_predicted(
        pred_csv,
        [
            ("PB2", 50, 200),  # outside tolerance
            ("PB2", 900, 1100),  # outside tolerance
            ("PB2", 1600, 1900),  # outside tolerance
        ],
    )

    result = evaluate_from_csv(pred_csv, truth_csv, tolerance_window=5)
    assert result.true_positives == 0
    assert result.false_positives == 3
    assert result.false_negatives == 3
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1_score == 0.0


def test_mixed_predictions_compute_expected_metrics(tmp_path: Path) -> None:
    truth_csv = tmp_path / "ground_truth.csv"
    pred_csv = tmp_path / "predicted.csv"
    _write_ground_truth(truth_csv)
    _write_predicted(
        pred_csv,
        [
            ("PB2", 101, 299),  # TP
            ("PB2", 498, 802),  # TP
            ("PB2", 50, 150),  # FP
        ],
    )

    result = evaluate_from_csv(pred_csv, truth_csv, tolerance_window=5)
    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert abs(result.precision - (2 / 3)) < 1e-9
    assert abs(result.recall - (2 / 3)) < 1e-9
    assert abs(result.f1_score - (2 / 3)) < 1e-9
