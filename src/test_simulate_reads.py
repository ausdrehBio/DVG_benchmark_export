"""Tests for read simulation wrapper (Checkpoint 2)."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_reads import is_valid_fastq, simulate_reads  # noqa: E402


def _write_dummy_art_executable(path: Path) -> None:
    script = """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
if "-o" not in args:
    raise SystemExit("missing -o")

prefix = pathlib.Path(args[args.index("-o") + 1])
prefix.parent.mkdir(parents=True, exist_ok=True)
r1 = prefix.with_name(prefix.name + "1.fq")
r2 = prefix.with_name(prefix.name + "2.fq")

r1.write_text("@read1/1\\nACGTACGT\\n+\\nFFFFFFFF\\n", encoding="utf-8")
r2.write_text("@read1/2\\nTGCATGCA\\n+\\nFFFFFFFF\\n", encoding="utf-8")
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_failing_art_executable(path: Path) -> None:
    script = """#!/usr/bin/env python3
import sys
sys.stderr.write("dyld: Library not loaded: @rpath/libgsl.25.dylib\\n")
raise SystemExit(1)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_simulate_reads_creates_nonempty_valid_fastq(tmp_path: Path) -> None:
    input_fasta = tmp_path / "synthetic_delvgs.fasta"
    input_fasta.write_text(">seq1\nACGTACGTACGTACGT\n", encoding="utf-8")

    dummy_sim = tmp_path / "dummy_art.py"
    _write_dummy_art_executable(dummy_sim)

    output_r1 = tmp_path / "reads_R1.fastq"
    output_r2 = tmp_path / "reads_R2.fastq"

    r1, r2 = simulate_reads(
        input_fasta=input_fasta,
        output_r1=output_r1,
        output_r2=output_r2,
        read_length=8,
        coverage_depth=10,
        error_rate=0.01,
        simulator_cmd=str(dummy_sim),
    )

    assert r1.exists() and r1.stat().st_size > 0
    assert r2.exists() and r2.stat().st_size > 0
    assert is_valid_fastq(r1)
    assert is_valid_fastq(r2)


@pytest.mark.parametrize(
    "fastq_text",
    [
        "@r1\nACGT\n-\nFFFF\n",  # third line must start with +
        "@r1\nACGT\n+\nFFF\n",  # sequence and quality length mismatch
        "@r1\nACGT\n+\n",  # incomplete record
    ],
)
def test_is_valid_fastq_rejects_invalid_format(tmp_path: Path, fastq_text: str) -> None:
    bad_fastq = tmp_path / "bad.fastq"
    bad_fastq.write_text(fastq_text, encoding="utf-8")
    assert not is_valid_fastq(bad_fastq)


def test_simulate_reads_missing_simulator_raises_clear_error(tmp_path: Path) -> None:
    input_fasta = tmp_path / "synthetic_delvgs.fasta"
    input_fasta.write_text(">seq1\nACGTACGTACGTACGT\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        simulate_reads(
            input_fasta=input_fasta,
            output_r1=tmp_path / "reads_R1.fastq",
            output_r2=tmp_path / "reads_R2.fastq",
            read_length=8,
            coverage_depth=10,
            error_rate=0.01,
            simulator_cmd="definitely_not_a_real_simulator",
        )


def test_simulate_reads_surfaces_dependency_hint(tmp_path: Path) -> None:
    input_fasta = tmp_path / "synthetic_delvgs.fasta"
    input_fasta.write_text(">seq1\nACGTACGTACGTACGT\n", encoding="utf-8")

    failing_sim = tmp_path / "failing_art.py"
    _write_failing_art_executable(failing_sim)

    with pytest.raises(RuntimeError, match="runtime dependency error"):
        simulate_reads(
            input_fasta=input_fasta,
            output_r1=tmp_path / "reads_R1.fastq",
            output_r2=tmp_path / "reads_R2.fastq",
            read_length=8,
            coverage_depth=10,
            error_rate=0.01,
            simulator_cmd=str(failing_sim),
        )
