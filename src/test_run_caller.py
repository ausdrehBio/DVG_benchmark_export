"""Tests for run_caller normalization logic (Checkpoint 3)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_virema  # noqa: E402
from run_virema import parse_raw_events, run_virema, standardize_predictions  # noqa: E402


def test_standardize_predictions_from_dummy_virema_output(tmp_path: Path) -> None:
    raw_events = tmp_path / "Virus_Recombination_Results.BED"
    output_csv = tmp_path / "predicted_delvgs.csv"

    raw_events.write_text(
        "\n".join(
            [
                "Reference_1\tStop\tReference_2\tStart\tCount",
                "PB2\t500\tPB2\t900\t14",
                "PB2\t1300\tPB2\t1700\t7",
                "PB2\t1300\tPB2\t1700\t5",  # duplicate event -> deduplicated
                "PB2\t2000\tPB2\t2000\t3",  # zero-length event -> filtered out
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events)
    assert len(parsed) == 4

    standardized = standardize_predictions(
        raw_events_path=raw_events,
        output_csv=output_csv,
        genome_filter="PB2",
        min_deletion_size=1,
    )

    assert output_csv.exists()
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8")))

    assert [event.genome_id for event in standardized] == ["PB2", "PB2"]
    assert [(event.predicted_start, event.predicted_end) for event in standardized] == [
        (500, 900),
        (1300, 1700),
    ]
    assert rows == [
        {"genome_id": "PB2", "predicted_start": "500", "predicted_end": "900"},
        {"genome_id": "PB2", "predicted_start": "1300", "predicted_end": "1700"},
    ]


def test_parse_raw_events_bedpe_without_header(tmp_path: Path) -> None:
    raw_events = tmp_path / "events_no_header.bedpe"
    raw_events.write_text(
        "\n".join(
            [
                "PB2\t100\t250\tPB2\t600\t700\tevent1\t10\t+\t+",
                "PB2\t200\t280\tPB2\t900\t980\tevent2\t8\t+\t+",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events)
    assert [(item.genome_id, item.predicted_start, item.predicted_end) for item in parsed] == [
        ("PB2", 250, 600),
        ("PB2", 280, 900),
    ]


def test_parse_virema_csv_start_end_reference_format(tmp_path: Path) -> None:
    raw_events = tmp_path / "pb2_sim_Virus_Recombination_Results.csv"
    raw_events.write_text(
        "\n".join(
            [
                "Start,End,Reference,NGS_read_count,Sequence",
                "544,1885,PB2,12,AAAA-CCCC",
                "600,1900,PB2,5,TTTT-GGGG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events)
    assert [(item.genome_id, item.predicted_start, item.predicted_end) for item in parsed] == [
        ("PB2", 544, 1885),
        ("PB2", 600, 1900),
    ]


def test_run_virema_uses_sam_filename_and_overwrite_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "virema"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def fake_run_command(cmd: list[str]) -> None:
        captured["cmd"] = cmd
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pb2_sim_Virus_Recombination_Results.csv").write_text(
            "Start,End,Reference,NGS_read_count,Sequence\n10,20,PB2,1,AAAA-TTTT\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(run_virema, "_run_command", fake_run_command)

    raw_output = run_virema(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        virema_script=str(Path("src/ViReMa/ViReMa.py")),
        python_executable="python3",
    )

    assert raw_output == output_dir / "pb2_sim_Virus_Recombination_Results.csv"
    assert "cmd" in captured
    assert captured["cmd"][4] == "pb2_sim.sam"
    assert str(output_dir / "pb2_sim.sam") not in captured["cmd"]
    assert "-Overwrite" in captured["cmd"]
