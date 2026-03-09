"""Run DVG caller DI-tector and normalize raw output to predicted_delvgs.csv."""

from __future__ import annotations

import argparse
import csv
import shutil
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


_BWA_INDEX_SUFFIXES = (".amb", ".ann", ".bwt", ".pac", ".sa")


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


def _candidate_ditector_outputs(output_dir: Path, output_tag: str) -> list[Path]:
    return [
        output_dir / f"{output_tag}_output_sorted.txt",
        output_dir / f"{output_tag}_counts.txt",
        output_dir / f"{output_tag}_output.txt",
    ]


def _expected_bwa_index_files(reference_fasta: Path) -> list[Path]:
    return [Path(f"{reference_fasta}{suffix}") for suffix in _BWA_INDEX_SUFFIXES]


def _ensure_bwa_index(reference_fasta: Path, bwa_executable: str = "bwa") -> None:
    """Ensure BWA index files exist for a reference FASTA, building them if needed."""

    missing_before = [path for path in _expected_bwa_index_files(reference_fasta) if not path.exists()]
    if not missing_before:
        return

    if shutil.which(bwa_executable) is None:
        raise FileNotFoundError(
            f"'{bwa_executable}' was not found on PATH and BWA index files are missing for {reference_fasta}."
        )

    _run_command([bwa_executable, "index", str(reference_fasta)])

    missing_after = [path for path in _expected_bwa_index_files(reference_fasta) if not path.exists()]
    if missing_after:
        raise FileNotFoundError(
            "BWA indexing was attempted but required index files are still missing. "
            f"Missing: {[str(path) for path in missing_after]}"
        )


def _resolve_ditector_script(
    ditector_script: str | None,
    legacy_virema_script: str | None,
    default_script: Path,
) -> str:
    """Prefer DI-tector script path while accepting legacy pipeline args."""

    if ditector_script:
        return ditector_script

    if legacy_virema_script:
        legacy_name = Path(legacy_virema_script).name.lower()
        if "ditector" in legacy_name:
            return legacy_virema_script

    return str(default_script)


def run_ditector(
    reference_fasta: Path,
    reads_r1: Path,
    reads_r2: Path,
    output_dir: Path,
    output_tag: str = "pb2_sim",
    ditector_script: str = "src/DI-tector/DI-tector_06.py",
    python_executable: str = "python3",
    microindel_length: int = 5,
    threads: int = 4,
    overwrite_output_dir: bool = True,
    keep_merged_fastq: bool = False,
    host_reference: Path | None = None,
    no_quantification: bool = False,
    min_reads: int = 1,
    additional_args: list[str] | None = None,
) -> Path:
    """Run DI-tector and return path to its raw output file."""

    for path in (reference_fasta, reads_r1, reads_r2):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if host_reference is not None and not host_reference.exists():
        raise FileNotFoundError(f"Host reference not found: {host_reference}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_fastq = output_dir / f"{output_tag}.merged.fastq"

    if not overwrite_output_dir:
        existing_outputs = [path for path in _candidate_ditector_outputs(output_dir, output_tag) if path.exists()]
        if existing_outputs:
            raise FileExistsError(
                "Output already exists and --no-overwrite was requested. "
                f"Found: {[str(path) for path in existing_outputs]}"
            )

    _ensure_bwa_index(reference_fasta)
    if host_reference is not None:
        _ensure_bwa_index(host_reference)

    _merge_fastq_files(reads_r1, reads_r2, merged_fastq)

    cmd = [
        python_executable,
        ditector_script,
        str(reference_fasta),
        str(merged_fastq),
        "-o",
        str(output_dir),
        "-t",
        output_tag,
        "-l",
        str(microindel_length),
        "-x",
        str(threads),
        "-n",
        str(min_reads),
    ]

    if host_reference is not None:
        cmd.extend(["-g", str(host_reference)])
    if no_quantification:
        cmd.append("-q")

    if additional_args:
        cmd.extend(additional_args)

    try:
        _run_command(cmd)
    finally:
        if not keep_merged_fastq and merged_fastq.exists():
            merged_fastq.unlink()

    candidates = _candidate_ditector_outputs(output_dir, output_tag)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "DI-tector completed but no known raw output file was found. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def _normalize_header(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")


def _to_int(value: str) -> int:
    return int(float(value))


def _is_deletion_type(raw_type: str) -> bool:
    return "deletion dvg" in raw_type.lower()


def _extract_genome_id(value: str) -> str:
    # DI-tector counts rows store references as "refA|refB".
    if "|" in value:
        return value.split("|", 1)[0].strip()
    return value.strip()


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


def _looks_like_header(row: list[str]) -> bool:
    normalized = {_normalize_header(item) for item in row}
    known_tokens = {
        "dvg_s_type",
        "dvg_type",
        "bp_pos",
        "ri_pos",
        "rname_f",
        "ref",
        "reference",
        "start",
        "end",
    }
    return bool(normalized.intersection(known_tokens))


def _parse_with_header(rows: list[list[str]]) -> list[PredictedDelVG]:
    headers = [_normalize_header(col) for col in rows[0]]
    data_rows = rows[1:]
    header_to_index = {name: idx for idx, name in enumerate(headers)}

    type_col = next((c for c in ("dvg_s_type", "dvg_type", "type") if c in header_to_index), None)
    ref_col = next(
        (c for c in ("rname_f", "reference_1", "reference", "chrom1", "chrom", "genome", "ref") if c in header_to_index),
        None,
    )
    left_col = next((c for c in ("bp_pos", "stop", "stop_1", "end1", "start", "predicted_start") if c in header_to_index), None)
    right_col = next((c for c in ("ri_pos", "start_2", "start2", "end", "predicted_end") if c in header_to_index), None)

    if ref_col is None or left_col is None or right_col is None:
        raise ValueError(
            "Unsupported raw output header. "
            "Need columns describing reference and two breakpoint positions."
        )

    predictions: list[PredictedDelVG] = []
    for row in data_rows:
        if len(row) < len(headers):
            continue
        if type_col is not None and not _is_deletion_type(row[header_to_index[type_col]]):
            continue
        try:
            genome_id = _extract_genome_id(row[header_to_index[ref_col]])
            predicted_start = _to_int(row[header_to_index[left_col]])
            predicted_end = _to_int(row[header_to_index[right_col]])
        except ValueError:
            continue
        if not genome_id:
            continue
        predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))

    return predictions


def _parse_without_header(rows: list[list[str]]) -> list[PredictedDelVG]:
    predictions: list[PredictedDelVG] = []

    for row in rows:
        if not row:
            continue
        raw_type = row[0].strip()
        if not raw_type or raw_type.startswith("=") or raw_type.lower().startswith("none or reads") or raw_type.lower().startswith("no data"):
            continue

        # DI-tector detailed row from *_output_sorted.txt
        if len(row) >= 18:
            if not _is_deletion_type(raw_type):
                continue
            genome_id = row[8].strip() or row[9].strip()
            try:
                predicted_start = _to_int(row[2])
                predicted_end = _to_int(row[3])
            except ValueError:
                continue
            if not genome_id:
                continue
            predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))
            continue

        # DI-tector aggregated row from *_counts.txt
        if len(row) >= 8 and _is_deletion_type(raw_type):
            try:
                genome_id = _extract_genome_id(row[5])
                predicted_start = _to_int(row[2])
                predicted_end = _to_int(row[3])
            except ValueError:
                continue
            if not genome_id:
                continue
            predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))
            continue

        # Generic fallback for simple 3-column rows: genome/start/end
        if len(row) == 3:
            try:
                genome_id = row[0].strip()
                predicted_start = _to_int(row[1])
                predicted_end = _to_int(row[2])
            except ValueError:
                continue
            if not genome_id:
                continue
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
    parser = argparse.ArgumentParser(description="Run DI-tector and standardize predicted DelVG breakpoints.")
    script_dir = Path(__file__).resolve().parent
    default_ditector_script = script_dir / "DI-tector" / "DI-tector_06.py"

    parser.add_argument("--reference-fasta", type=Path, default=Path("data/PB2.fasta"))
    parser.add_argument("--reads-r1", type=Path, default=Path("output/reads_R1.fastq"))
    parser.add_argument("--reads-r2", type=Path, default=Path("output/reads_R2.fastq"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/virema"))
    parser.add_argument("--output-csv", type=Path, default=Path("output/predicted_delvgs.csv"))
    parser.add_argument(
        "--raw-events",
        type=Path,
        default=None,
        help="Existing raw caller output to parse. If provided, DI-tector is not executed.",
    )
    parser.add_argument("--output-tag", type=str, default="pb2_sim")
    parser.add_argument(
        "--ditector-script",
        type=str,
        default=None,
        help=f"Path to DI-tector script (default: {default_ditector_script}).",
    )
    parser.add_argument(
        "--virema-script",
        type=str,
        default=None,
        help="Legacy argument accepted for pipeline compatibility. Prefer --ditector-script.",
    )
    parser.add_argument("--python-executable", type=str, default="python3")
    parser.add_argument("--microindel-length", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if target DI-tector output files already exist.",
    )
    parser.add_argument("--min-deletion-size", type=int, default=1)
    parser.add_argument("--genome-filter", type=str, default=None)
    parser.add_argument(
        "--keep-merged-fastq",
        action="store_true",
        help="Keep intermediate merged FASTQ used for DI-tector input.",
    )
    parser.add_argument(
        "--host-reference",
        type=Path,
        default=None,
        help="Optional host reference FASTA for DI-tector -g/--Host_Ref.",
    )
    parser.add_argument(
        "--no-quantification",
        action="store_true",
        help="Pass -q to DI-tector to disable percentage quantification.",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=1,
        help="Pass -n to DI-tector (minimum supporting reads per event).",
    )
    parser.add_argument(
        "--additional-ditector-arg",
        "--additional-virema-arg",
        dest="additional_ditector_arg",
        action="append",
        default=[],
        help="Additional argument to append to DI-tector command (can be repeated).",
    )

    args = parser.parse_args()
    args.ditector_script = _resolve_ditector_script(args.ditector_script, args.virema_script, default_ditector_script)
    return args


def main() -> None:
    args = parse_args()

    raw_events_path = args.raw_events
    if raw_events_path is None:
        raw_events_path = run_ditector(
            reference_fasta=args.reference_fasta,
            reads_r1=args.reads_r1,
            reads_r2=args.reads_r2,
            output_dir=args.output_dir,
            output_tag=args.output_tag,
            ditector_script=args.ditector_script,
            python_executable=args.python_executable,
            microindel_length=args.microindel_length,
            threads=args.threads,
            overwrite_output_dir=not args.no_overwrite,
            keep_merged_fastq=args.keep_merged_fastq,
            host_reference=args.host_reference,
            no_quantification=args.no_quantification,
            min_reads=args.min_reads,
            additional_args=args.additional_ditector_arg,
        )

    standardize_predictions(
        raw_events_path=raw_events_path,
        output_csv=args.output_csv,
        genome_filter=args.genome_filter,
        min_deletion_size=args.min_deletion_size,
    )


if __name__ == "__main__":
    main()
