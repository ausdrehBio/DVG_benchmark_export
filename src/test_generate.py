"""Tests for synthetic DelVG generation (Checkpoint 1)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_delvgs import run_pipeline  # noqa: E402


def _read_fasta_records(fasta_path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        records[current_id] = "".join(chunks)

    return records


def test_delvg_length_and_coordinate_mapping(tmp_path: Path) -> None:
    """Generated DelVG length and coordinates must map exactly to WT."""

    input_fasta = tmp_path / "PB2.fasta"
    output_fasta = tmp_path / "synthetic_delvgs.fasta"
    output_csv = tmp_path / "ground_truth.csv"

    wt_sequence = "ACGT" * 300  # 1200 bp
    wt_id = "PB2_WT"
    input_fasta.write_text(f">{wt_id}\n{wt_sequence}\n", encoding="utf-8")

    run_pipeline(
        input_fasta=input_fasta,
        output_fasta=output_fasta,
        output_csv=output_csv,
        number_of_sequences=15,
        min_deletion_size=20,
        max_deletion_size=120,
        seed=17,
    )

    assert output_fasta.exists() and output_fasta.stat().st_size > 0
    assert output_csv.exists() and output_csv.stat().st_size > 0

    sequences = _read_fasta_records(output_fasta)
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8")))
    assert len(rows) == 16  # 1 WT + 15 DelVGs

    for row in rows:
        seq_id = row["sequence_id"]
        del_seq = sequences[seq_id]
        deletion_size = int(row["deletion_size"])
        deletion_start = row["deletion_start"]
        deletion_end = row["deletion_end"]

        if deletion_size == 0:
            assert deletion_start == "NA"
            assert deletion_end == "NA"
            assert del_seq == wt_sequence
            continue

        start = int(deletion_start)
        end = int(deletion_end)
        assert end - start + 1 == deletion_size
        assert len(del_seq) + deletion_size == len(wt_sequence)

        expected_sequence = wt_sequence[: start - 1] + wt_sequence[end:]
        assert del_seq == expected_sequence


@pytest.mark.parametrize(
    "min_del,max_del",
    [
        (0, 10),
        (50, 10),
    ],
)
def test_invalid_deletion_parameters_raise(tmp_path: Path, min_del: int, max_del: int) -> None:
    input_fasta = tmp_path / "PB2.fasta"
    input_fasta.write_text(">PB2_WT\nACGTACGTACGT\n", encoding="utf-8")

    with pytest.raises(ValueError):
        run_pipeline(
            input_fasta=input_fasta,
            output_fasta=tmp_path / "synthetic_delvgs.fasta",
            output_csv=tmp_path / "ground_truth.csv",
            number_of_sequences=2,
            min_deletion_size=min_del,
            max_deletion_size=max_del,
            seed=1,
        )
