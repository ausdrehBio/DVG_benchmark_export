"""Run DVG caller Virema and normalize raw output to predicted_delvgs.csv."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PredictedDelVG:
    """Container for one predicted deletion event."""

    genome_id: str
    predicted_start: int
    predicted_end: int


def _run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr if stderr else stdout
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{details}")


def _merge_fastq_files(reads_r1: Path, reads_r2: Path, merged_output: Path) -> None:
    """Concatenate R1 and R2 into one FASTQ file for tools that expect single input."""

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    with merged_output.open("w", encoding="utf-8") as out_handle:
        for read_path in (reads_r1, reads_r2):
            with read_path.open("r", encoding="utf-8") as in_handle:
                out_handle.write(in_handle.read())


def _candidate_virema_outputs(output_dir: Path, output_tag: str) -> list[Path]:
    return [
        output_dir / f"{output_tag}_Virus_Recombination_Results.csv",
        output_dir / "Virus_Recombination_Results.csv",
        output_dir / f"{output_tag}_Virus_Recombination_Results.txt",
        output_dir / "Virus_Recombination_Results.txt",
        output_dir / "BED_Files" / "Virus_Recombination_Results.BED",
        output_dir / "BED_Files" / "Virus_Recombination_Results.BEDPE",
        output_dir / "BED_Files" / "Virus_Recombination_Results.bed",
        output_dir / "BED_Files" / "Virus_Recombination_Results.BEDPE",
        output_dir / "Virus_Recombination_Results.BED",
        output_dir / "Virus_Recombination_Results.BEDPE",
        output_dir / f"{output_tag}_Virus_Recombination_Results.BED",
        output_dir / f"{output_tag}_Virus_Recombination_Results.BEDPE",
        output_dir / f"{output_tag}_recombinations.txt",
    ]


def run_virema(
    reference_fasta: Path,
    reads_r1: Path,
    reads_r2: Path,
    output_dir: Path,
    output_tag: str = "pb2_sim",
    virema_script: str = "src/ViReMa/ViReMa.py",
    python_executable: str = "python3",
    microindel_length: int = 5,
    threads: int = 4,
    overwrite_output_dir: bool = True,
    keep_merged_fastq: bool = False,
    additional_args: list[str] | None = None,
) -> Path:
    """Run ViReMa and return path to its raw recombination output file.

    This wrapper merges paired-end FASTQ into one input file because ViReMa's
    canonical CLI expects a single FASTA/FASTQ input stream.
    """

    for path in (reference_fasta, reads_r1, reads_r2):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_fastq = output_dir / f"{output_tag}.merged.fastq"
    output_sam_name = f"{output_tag}.sam"

    _merge_fastq_files(reads_r1, reads_r2, merged_fastq)

    cmd = [
        python_executable,
        virema_script,
        str(reference_fasta),
        str(merged_fastq),
        output_sam_name,
        "--Output_Dir",
        str(output_dir),
        "--Output_Tag",
        output_tag,
        "--MicroInDel_Length",
        str(microindel_length),
        "--p",
        str(threads),
        "-BED",
    ]
    if overwrite_output_dir:
        cmd.append("-Overwrite")

    if additional_args:
        cmd.extend(additional_args)

    try:
        _run_command(cmd)
    finally:
        if not keep_merged_fastq and merged_fastq.exists():
            merged_fastq.unlink()

    candidates = _candidate_virema_outputs(output_dir, output_tag)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "ViReMa completed but no known raw output file was found. "
        f"Checked: {[str(p) for p in candidates]}"
    )


def _normalize_header(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")


def _to_int(value: str) -> int:
    return int(float(value))


def _looks_like_header(row: list[str]) -> bool:
    normalized = {_normalize_header(item) for item in row}
    known_tokens = {
        "reference_1",
        "reference_2",
        "reference",
        "chrom1",
        "chrom2",
        "start",
        "end",
        "stop",
        "count",
    }
    return bool(normalized.intersection(known_tokens))


def _read_noncomment_rows(raw_events_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with raw_events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            delimiter = "\t" if "\t" in stripped else ","
            rows.append([part.strip() for part in stripped.split(delimiter)])
    return rows


def _parse_with_header(rows: list[list[str]]) -> list[PredictedDelVG]:
    headers = [_normalize_header(col) for col in rows[0]]
    data_rows = rows[1:]

    header_to_index = {name: idx for idx, name in enumerate(headers)}

    ref_col = next(
        (c for c in ("reference_1", "reference", "chrom1", "chrom", "genome", "sequence") if c in header_to_index),
        None,
    )
    stop_col = next(
        (
            c
            for c in (
                "stop",
                "stop_1",
                "end1",
                "breakpoint_start",
                "bp_left",
                "predicted_start",
                "deletion_start",
            )
            if c in header_to_index
        ),
        None,
    )
    start_col = next(
        (
            c
            for c in (
                "start",
                "end",
                "start_2",
                "start2",
                "breakpoint_end",
                "bp_right",
                "predicted_end",
                "deletion_end",
            )
            if c in header_to_index
        ),
        None,
    )

    # BEDPE-style fallback with explicit column names.
    if stop_col is None and start_col is None:
        if {"end1", "start2"}.issubset(header_to_index):
            stop_col = "end1"
            start_col = "start2"
    # ViReMa CSV fallback: Start/End/Reference
    if stop_col is None and {"start", "end"}.issubset(header_to_index):
        stop_col = "start"
        start_col = "end"

    if ref_col is None or stop_col is None or start_col is None:
        raise ValueError(
            "Unsupported raw output header. "
            "Need columns describing reference, left breakpoint, and right breakpoint."
        )

    predictions: list[PredictedDelVG] = []
    for row in data_rows:
        if len(row) < len(headers):
            continue
        genome_id = row[header_to_index[ref_col]]
        predicted_start = _to_int(row[header_to_index[stop_col]])
        predicted_end = _to_int(row[header_to_index[start_col]])
        predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))

    return predictions


def _parse_without_header(rows: list[list[str]]) -> list[PredictedDelVG]:
    predictions: list[PredictedDelVG] = []
    for row in rows:
        if len(row) >= 6:
            # BEDPE-like: chrom1 start1 end1 chrom2 start2 end2 ...
            genome_id = row[0]
            predicted_start = _to_int(row[2])  # end1
            predicted_end = _to_int(row[4])  # start2
            predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))
        elif len(row) >= 3:
            # Generic fallback: genome/start/end
            genome_id = row[0]
            predicted_start = _to_int(row[1])
            predicted_end = _to_int(row[2])
            predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))

    return predictions


def parse_raw_events(raw_events_path: Path) -> list[PredictedDelVG]:
    """Parse raw caller output into normalized prediction objects."""

    if not raw_events_path.exists():
        raise FileNotFoundError(f"Raw events file not found: {raw_events_path}")

    rows = _read_noncomment_rows(raw_events_path)
    if not rows:
        return []

    if _looks_like_header(rows[0]):
        return _parse_with_header(rows)
    return _parse_without_header(rows)


def write_predictions_csv(predictions: Iterable[PredictedDelVG], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["genome_id", "predicted_start", "predicted_end"])
        writer.writeheader()
        for event in predictions:
            writer.writerow(
                {
                    "genome_id": event.genome_id,
                    "predicted_start": event.predicted_start,
                    "predicted_end": event.predicted_end,
                }
            )


def standardize_predictions(
    raw_events_path: Path,
    output_csv: Path,
    genome_filter: str | None = None,
    min_deletion_size: int = 1,
) -> list[PredictedDelVG]:
    """Convert raw caller output to standardized predicted_delvgs.csv."""

    if min_deletion_size < 1:
        raise ValueError("min_deletion_size must be >= 1.")

    events = parse_raw_events(raw_events_path)

    normalized: list[PredictedDelVG] = []
    seen: set[tuple[str, int, int]] = set()
    for event in events:
        if genome_filter and event.genome_id != genome_filter:
            continue
        start = min(event.predicted_start, event.predicted_end)
        end = max(event.predicted_start, event.predicted_end)
        if (end - start) < min_deletion_size:
            continue
        key = (event.genome_id, start, end)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(PredictedDelVG(event.genome_id, start, end))

    normalized.sort(key=lambda x: (x.genome_id, x.predicted_start, x.predicted_end))
    write_predictions_csv(normalized, output_csv)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViReMa and standardize predicted DelVG breakpoints.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--reference-fasta", type=Path, default=Path("data/PB2.fasta"))
    parser.add_argument("--reads-r1", type=Path, default=Path("output/reads_R1.fastq"))
    parser.add_argument("--reads-r2", type=Path, default=Path("output/reads_R2.fastq"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/virema"))
    parser.add_argument("--output-csv", type=Path, default=Path("output/predicted_delvgs.csv"))
    parser.add_argument(
        "--raw-events",
        type=Path,
        default=None,
        help="Existing raw caller output to parse. If provided, ViReMa is not executed.",
    )
    parser.add_argument("--output-tag", type=str, default="pb2_sim")
    parser.add_argument(
        "--virema-script",
        type=str,
        default=str(script_dir / "ViReMa" / "ViReMa.py"),
        help="Path to ViReMa.py script (default: src/ViReMa/ViReMa.py).",
    )
    parser.add_argument("--python-executable", type=str, default="python3")
    parser.add_argument("--microindel-length", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not pass -Overwrite to ViReMa (ViReMa may append a timestamp to output directory).",
    )
    parser.add_argument("--min-deletion-size", type=int, default=1)
    parser.add_argument("--genome-filter", type=str, default=None)
    parser.add_argument(
        "--keep-merged-fastq",
        action="store_true",
        help="Keep intermediate merged FASTQ used for ViReMa input.",
    )
    parser.add_argument(
        "--additional-virema-arg",
        action="append",
        default=[],
        help="Additional argument to append to ViReMa command (can be repeated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_events_path = args.raw_events
    if raw_events_path is None:
        raw_events_path = run_virema(
            reference_fasta=args.reference_fasta,
            reads_r1=args.reads_r1,
            reads_r2=args.reads_r2,
            output_dir=args.output_dir,
            output_tag=args.output_tag,
            virema_script=args.virema_script,
            python_executable=args.python_executable,
            microindel_length=args.microindel_length,
            threads=args.threads,
            overwrite_output_dir=not args.no_overwrite,
            keep_merged_fastq=args.keep_merged_fastq,
            additional_args=args.additional_virema_arg,
        )

    standardize_predictions(
        raw_events_path=raw_events_path,
        output_csv=args.output_csv,
        genome_filter=args.genome_filter,
        min_deletion_size=args.min_deletion_size,
    )


if __name__ == "__main__":
    main()
