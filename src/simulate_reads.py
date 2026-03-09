"""Simulate paired-end Illumina reads from synthetic DelVG FASTA using ART."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path


def estimate_q_shift_from_error_rate(error_rate: float, baseline_q: int = 30) -> int:
    """Approximate ART quality shift from a target per-base error rate.

    ART does not take a direct error-rate argument. This helper converts
    `error_rate` to an approximate Phred quality and then computes a quality
    shift relative to a baseline quality (default Q30).
    """

    if error_rate <= 0 or error_rate >= 1:
        raise ValueError("error_rate must be in (0, 1).")

    target_q = int(round(-10 * math.log10(error_rate)))
    return target_q - baseline_q


def build_art_command(
    simulator_cmd: str,
    input_fasta: Path,
    output_prefix: Path,
    read_length: int,
    coverage_depth: float,
    error_rate: float | None,
    sequencer_profile: str = "HS25",
    fragment_length: int = 250,
    fragment_std: int = 25,
) -> list[str]:
    """Build ART command for paired-end simulation."""

    if read_length < 1:
        raise ValueError("read_length must be >= 1.")
    if coverage_depth <= 0:
        raise ValueError("coverage_depth must be > 0.")
    if fragment_length < read_length:
        raise ValueError("fragment_length must be >= read_length.")
    if fragment_std < 0:
        raise ValueError("fragment_std must be >= 0.")

    cmd = [
        simulator_cmd,
        "-ss",
        sequencer_profile,
        "-i",
        str(input_fasta),
        "-p",
        "-l",
        str(read_length),
        "-f",
        str(coverage_depth),
        "-m",
        str(fragment_length),
        "-s",
        str(fragment_std),
        "-o",
        str(output_prefix),
    ]

    if error_rate is not None:
        q_shift = estimate_q_shift_from_error_rate(error_rate)
        cmd.extend(["-qs", str(q_shift), "-qs2", str(q_shift)])

    return cmd


def _locate_art_outputs(output_prefix: Path) -> tuple[Path, Path]:
    r1_candidates = [output_prefix.with_name(output_prefix.name + suffix) for suffix in ("1.fq", "1.fastq")]
    r2_candidates = [output_prefix.with_name(output_prefix.name + suffix) for suffix in ("2.fq", "2.fastq")]

    r1 = next((p for p in r1_candidates if p.exists()), None)
    r2 = next((p for p in r2_candidates if p.exists()), None)

    if r1 is None or r2 is None:
        raise FileNotFoundError(
            "ART output FASTQ files were not found. "
            f"Expected one of {[str(p) for p in r1_candidates + r2_candidates]}"
        )

    return r1, r2


def is_valid_fastq(path: Path) -> bool:
    """Return True if file appears to be valid FASTQ with complete records."""

    if not path.exists() or path.stat().st_size == 0:
        return False

    with path.open("r", encoding="utf-8") as handle:
        line_index = 0
        records = 0
        seq_len = 0
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            mod = line_index % 4

            if mod == 0:
                if not line.startswith("@"):
                    return False
            elif mod == 1:
                if not line:
                    return False
                seq_len = len(line)
            elif mod == 2:
                if not line.startswith("+"):
                    return False
            else:
                if len(line) != seq_len:
                    return False
                records += 1

            line_index += 1

    if line_index == 0:
        return False
    if line_index % 4 != 0:
        return False
    return records > 0


def simulate_reads(
    input_fasta: Path,
    output_r1: Path,
    output_r2: Path,
    read_length: int,
    coverage_depth: float,
    error_rate: float | None = None,
    simulator_cmd: str = "art_illumina",
    sequencer_profile: str = "HS25",
    fragment_length: int = 250,
    fragment_std: int = 25,
    keep_intermediate: bool = False,
) -> tuple[Path, Path]:
    """Run ART and return standardized FASTQ outputs (R1, R2)."""

    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")
    if "/" not in simulator_cmd and shutil.which(simulator_cmd) is None:
        raise FileNotFoundError(
            f"Simulator executable '{simulator_cmd}' was not found on PATH. "
            "Install ART or pass --simulator-cmd with an absolute executable path."
        )

    output_r1.parent.mkdir(parents=True, exist_ok=True)
    output_r2.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = output_r1.parent / "_art_tmp_"

    cmd = build_art_command(
        simulator_cmd=simulator_cmd,
        input_fasta=input_fasta,
        output_prefix=output_prefix,
        read_length=read_length,
        coverage_depth=coverage_depth,
        error_rate=error_rate,
        sequencer_profile=sequencer_profile,
        fragment_length=fragment_length,
        fragment_std=fragment_std,
    )

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr if stderr else stdout
        dependency_hint = ""
        if "dyld" in details or "Library not loaded" in details or "libgsl" in details:
            dependency_hint = (
                " ART runtime dependency error detected. If using conda, install "
                "matching runtime libraries in the same environment, e.g. "
                "`conda install -n <env> -c conda-forge gsl` and reinstall ART "
                "with `conda install -n <env> -c bioconda art`."
            )
        raise RuntimeError(
            f"ART simulation failed with exit code {result.returncode}: {details}{dependency_hint}"
        )

    tmp_r1, tmp_r2 = _locate_art_outputs(output_prefix)
    shutil.move(str(tmp_r1), str(output_r1))
    shutil.move(str(tmp_r2), str(output_r2))

    if not keep_intermediate:
        for aln_ext in (".aln", "1.aln", "2.aln"):
            aln_path = output_prefix.with_name(output_prefix.name + aln_ext)
            if aln_path.exists():
                aln_path.unlink()

    if not is_valid_fastq(output_r1) or not is_valid_fastq(output_r2):
        raise ValueError("Generated FASTQ output failed validation.")

    return output_r1, output_r2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate paired-end reads from synthetic DelVG FASTA.")
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=Path("output/synthetic_delvgs.fasta"),
        help="Input FASTA of synthetic DelVG sequences.",
    )
    parser.add_argument(
        "--output-r1",
        type=Path,
        default=Path("output/reads_R1.fastq"),
        help="Output FASTQ for R1.",
    )
    parser.add_argument(
        "--output-r2",
        type=Path,
        default=Path("output/reads_R2.fastq"),
        help="Output FASTQ for R2.",
    )
    parser.add_argument("--read-length", type=int, default=150, help="Read length for each mate.")
    parser.add_argument("--coverage-depth", type=float, default=200.0, help="Target sequencing depth.")
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.001,
        help="Approximate per-base error rate (mapped to ART quality shift).",
    )
    parser.add_argument(
        "--simulator-cmd",
        type=str,
        default="art_illumina",
        help="Simulator executable (default: art_illumina).",
    )
    parser.add_argument(
        "--sequencer-profile",
        type=str,
        default="HS25",
        help="ART sequencing profile (e.g. HS25, HS20, MSv3).",
    )
    parser.add_argument("--fragment-length", type=int, default=250, help="Mean fragment length.")
    parser.add_argument("--fragment-std", type=int, default=25, help="Fragment length standard deviation.")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate ART output files if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulate_reads(
        input_fasta=args.input_fasta,
        output_r1=args.output_r1,
        output_r2=args.output_r2,
        read_length=args.read_length,
        coverage_depth=args.coverage_depth,
        error_rate=args.error_rate,
        simulator_cmd=args.simulator_cmd,
        sequencer_profile=args.sequencer_profile,
        fragment_length=args.fragment_length,
        fragment_std=args.fragment_std,
        keep_intermediate=args.keep_intermediate,
    )


if __name__ == "__main__":
    main()
