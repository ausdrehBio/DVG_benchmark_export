"""Tests for DI-tector wrapper + normalization logic."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_DItector  # noqa: E402
from run_DItector import parse_raw_events, run_ditector, standardize_predictions  # noqa: E402


def _touch_bwa_index_files(reference_fasta: Path) -> None:
    for suffix in (".amb", ".ann", ".bwt", ".pac", ".sa"):
        Path(f"{reference_fasta}{suffix}").write_text("idx\n", encoding="utf-8")


def test_standardize_predictions_from_ditector_output_sorted(tmp_path: Path) -> None:
    raw_events = tmp_path / "pb2_sim_output_sorted.txt"
    output_csv = tmp_path / "predicted_delvgs.csv"

    raw_events.write_text(
        "\n".join(
            [
                "DVG's type\tLength\tBP_Pos\tRI_Pos\tDelta_Positions\tSegmentation\tMAPQ_F\tMAPQ_L\tRNAME_F\tRNAME_L\tCIGAR_F\tCIGAR_L\tMD_CIGAR_F\tMD_CIGAR_L\tPOS_F\tPOS_L\tQNAME_F\tSEQ_FL_ori",
                "Deletion DVG (Fwd. strand)\t100\t500\t900\t399\t10|90\t60\t60\tPB2\tPB2\t50M\t50M\t50\t50\t451\t851\tread1\tACTG",
                "Insertion DVG (Fwd. strand)\t10\t700\t705\t4\t10|90\t60\t60\tPB2\tPB2\t50M\t50M\t50\t50\t651\t701\tread2\tACTG",
                "Deletion DVG (Rev. strand)\t120\t1300\t1700\t399\t10|90\t60\t60\tPB2\tPB2\t50M\t50M\t50\t50\t1251\t1651\tread3\tACTG",
                "Deletion DVG (Rev. strand)\t120\t1300\t1700\t399\t10|90\t60\t60\tPB2\tPB2\t50M\t50M\t50\t50\t1251\t1651\tread4\tACTG",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events)
    assert len(parsed) == 3  # insertion filtered out

    standardized = standardize_predictions(
        raw_events_path=raw_events,
        output_csv=output_csv,
        genome_filter="PB2",
        min_deletion_size=1,
    )

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8")))
    assert [(event.genome_id, event.predicted_start, event.predicted_end) for event in standardized] == [
        ("PB2", 500, 900),
        ("PB2", 1300, 1700),
    ]
    assert rows == [
        {"genome_id": "PB2", "predicted_start": "500", "predicted_end": "900"},
        {"genome_id": "PB2", "predicted_start": "1300", "predicted_end": "1700"},
    ]


def test_parse_ditector_counts_without_header_sections(tmp_path: Path) -> None:
    raw_events = tmp_path / "pb2_sim_counts.txt"
    raw_events.write_text(
        "\n".join(
            [
                "=================================",
                "= Deletion DVG (Fwd. strand)",
                "=================================",
                "DVG's type\tLength\tBP_Pos\tRI_Pos\tDelta_Positions\tRef\tCounts\t%_to_Virus",
                "Deletion DVG (Fwd. strand)\t100\t500\t900\t399\tPB2|PB2\t11\t8.0%|7.0%",
                "Insertion DVG (Fwd. strand)\t10\t700\t705\t4\tPB2|PB2\t3\t2.0%|2.0%",
                "Deletion DVG (Rev. strand)\t120\t1300\t1700\t399\tPB2|PB2\t9\t5.0%|4.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events)
    assert [(item.genome_id, item.predicted_start, item.predicted_end) for item in parsed] == [
        ("PB2", 500, 900),
        ("PB2", 1300, 1700),
    ]


def test_run_ditector_builds_expected_command_and_discovers_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    host_fasta = tmp_path / "host.fasta"
    output_dir = tmp_path / "ditector"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    host_fasta.write_text(">HOST\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")
    _touch_bwa_index_files(reference_fasta)
    _touch_bwa_index_files(host_fasta)

    captured: dict[str, list[str]] = {}

    def fake_run_command(cmd: list[str]) -> None:
        captured["cmd"] = cmd
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pb2_sim_counts.txt").write_text(
            "DVG's type\tLength\tBP_Pos\tRI_Pos\tDelta_Positions\tRef\tCounts\t%_to_Virus\n"
            "Deletion DVG (Fwd. strand)\t100\t10\t30\t19\tPB2|PB2\t4\tND\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(run_DItector, "_run_command", fake_run_command)

    raw_output = run_ditector(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        ditector_script=str(Path("src/DI-tector/DI-tector_06.py")),
        python_executable="python3",
        microindel_length=7,
        threads=8,
        host_reference=host_fasta,
        no_quantification=True,
        min_reads=3,
    )

    assert raw_output == output_dir / "pb2_sim_counts.txt"
    assert "cmd" in captured
    assert captured["cmd"][:3] == [
        "python3",
        "src/DI-tector/DI-tector_06.py",
        str(reference_fasta),
    ]
    assert str(output_dir / "pb2_sim.merged.fastq") in captured["cmd"]
    assert "-o" in captured["cmd"]
    assert str(output_dir) in captured["cmd"]
    assert "-t" in captured["cmd"]
    assert "pb2_sim" in captured["cmd"]
    assert "-l" in captured["cmd"]
    assert "7" in captured["cmd"]
    assert "-x" in captured["cmd"]
    assert "8" in captured["cmd"]
    assert "-n" in captured["cmd"]
    assert "3" in captured["cmd"]
    assert "-q" in captured["cmd"]
    assert "-g" in captured["cmd"]
    assert str(host_fasta) in captured["cmd"]
    assert not (output_dir / "pb2_sim.merged.fastq").exists()


def test_run_ditector_builds_bwa_index_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "ditector"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")

    captured_cmds: list[list[str]] = []

    def fake_run_command(cmd: list[str]) -> None:
        captured_cmds.append(cmd)
        if cmd[:2] == ["bwa", "index"]:
            _touch_bwa_index_files(reference_fasta)
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pb2_sim_output_sorted.txt").write_text(
            "DVG's type\tLength\tBP_Pos\tRI_Pos\tDelta_Positions\tSegmentation\tMAPQ_F\tMAPQ_L\tRNAME_F\tRNAME_L\tCIGAR_F\tCIGAR_L\tMD_CIGAR_F\tMD_CIGAR_L\tPOS_F\tPOS_L\tQNAME_F\tSEQ_FL_ori\n"
            "Deletion DVG (Fwd. strand)\t100\t10\t30\t19\t10|90\t60\t60\tPB2\tPB2\t50M\t50M\t50\t50\t1\t2\tread1\tACTG\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(run_DItector, "_run_command", fake_run_command)
    monkeypatch.setattr(run_DItector.shutil, "which", lambda executable: "/mock/bin/bwa" if executable == "bwa" else None)

    raw_output = run_ditector(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        ditector_script=str(Path("src/DI-tector/DI-tector_06.py")),
        python_executable="python3",
    )

    assert raw_output == output_dir / "pb2_sim_output_sorted.txt"
    assert captured_cmds[0][:2] == ["bwa", "index"]
    assert captured_cmds[0][2] == str(reference_fasta)
    assert captured_cmds[1][:2] == ["python3", "src/DI-tector/DI-tector_06.py"]


def test_run_ditector_no_overwrite_raises_if_outputs_exist(tmp_path: Path) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "ditector"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pb2_sim_output_sorted.txt").write_text("dummy\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_ditector(
            reference_fasta=reference_fasta,
            reads_r1=reads_r1,
            reads_r2=reads_r2,
            output_dir=output_dir,
            output_tag="pb2_sim",
            overwrite_output_dir=False,
        )
