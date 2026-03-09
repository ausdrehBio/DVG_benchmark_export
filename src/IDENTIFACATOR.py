"""Run DelVG identification on real FASTQ input data.

Workflow:
1) Read input data
2) Run caller_cmd (kept identical to run_pipeline.py)
3) Save results
4) Visualize results (and save plots)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from tqdm import tqdm


def run_step(name: str, cmd: list[str]) -> None:
    """Execute one command step and stop on failure."""

    print(f"\n[{name}] Running:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}.")


def run_action_step(name: str, action: Callable[[], None]) -> None:
    """Execute one python action step and stop on failure."""

    print(f"\n[{name}] Running:")
    action()


def _discover_single_fastq(input_dir: Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("*.fastq", "*.fq", "*.fastq.gz", "*.fq.gz"):
        candidates.extend(sorted(input_dir.glob(pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"No FASTQ file found in {input_dir}. Provide one via --input-fastq."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple FASTQ files found in input directory. "
            f"Use --input-fastq explicitly. Found: {[str(p) for p in candidates]}"
        )
    return candidates[0]


def _copy_or_decompress_fastq(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8", errors="replace") as in_handle:
            with destination.open("w", encoding="utf-8") as out_handle:
                shutil.copyfileobj(in_handle, out_handle)
        return
    shutil.copy2(source, destination)


def _count_fastq_reads(fastq_path: Path) -> int:
    line_count = 0
    with fastq_path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            line_count += 1
    return line_count // 4


def _sanitize_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value.strip())


def _discover_reference_fastas(reference_dir: Path, accession_file: Path | None) -> list[Path]:
    """Discover reference FASTA files.

    Priority:
    1) If accession file exists, use its order and map '#LABEL' to LABEL.fasta
       (fallback to ACCESSION.fasta if no label file exists).
    2) Otherwise, use all '*.fasta' files in reference_dir.
    """

    discovered: list[Path] = []
    seen: set[Path] = set()

    if accession_file is not None and accession_file.exists():
        with accession_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue

                payload, comment = (raw.split("#", maxsplit=1) + [""])[:2]
                tokens = payload.strip().split()
                if not tokens:
                    continue

                accession = _sanitize_stem(tokens[0])
                label = _sanitize_stem(comment.strip()) if comment.strip() else None

                candidates: list[Path] = []
                if label:
                    candidates.append(reference_dir / f"{label}.fasta")
                candidates.append(reference_dir / f"{accession}.fasta")

                chosen = next((p for p in candidates if p.exists()), None)
                if chosen is None:
                    continue
                resolved = chosen.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                discovered.append(chosen)

    if discovered:
        return discovered

    for fasta in sorted(reference_dir.glob("*.fasta")):
        resolved = fasta.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        discovered.append(fasta)

    if not discovered:
        raise FileNotFoundError(
            f"No reference FASTA files found in {reference_dir}. "
            "Expected files like PB2.fasta or accessions listed in reference_accessions.txt."
        )
    return discovered


def prepare_input_data(
    input_fastq: Path,
    output_r1: Path,
    output_r2: Path,
    input_fastq_r2: Path | None = None,
) -> dict[str, object]:
    """Prepare caller input FASTQ files expected by caller_cmd."""

    if not input_fastq.exists():
        raise FileNotFoundError(f"Input FASTQ not found: {input_fastq}")
    if input_fastq_r2 is not None and not input_fastq_r2.exists():
        raise FileNotFoundError(f"Input FASTQ R2 not found: {input_fastq_r2}")

    _copy_or_decompress_fastq(input_fastq, output_r1)

    if input_fastq_r2 is None:
        output_r2.parent.mkdir(parents=True, exist_ok=True)
        output_r2.write_text("", encoding="utf-8")
        mode = "single-end"
        reads_r2 = 0
    else:
        _copy_or_decompress_fastq(input_fastq_r2, output_r2)
        mode = "paired-end"
        reads_r2 = _count_fastq_reads(output_r2)

    reads_r1 = _count_fastq_reads(output_r1)
    return {
        "mode": mode,
        "input_fastq_r1": str(input_fastq.resolve()),
        "input_fastq_r2": None if input_fastq_r2 is None else str(input_fastq_r2.resolve()),
        "prepared_r1": str(output_r1.resolve()),
        "prepared_r2": str(output_r2.resolve()),
        "reads_r1": reads_r1,
        "reads_r2": reads_r2,
    }


def _load_predictions(predicted_csv: Path) -> list[tuple[int, int]]:
    if not predicted_csv.exists():
        raise FileNotFoundError(f"Predicted CSV not found: {predicted_csv}")

    events: list[tuple[int, int]] = []
    with predicted_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                start = int(float(row["predicted_start"]))
                end = int(float(row["predicted_end"]))
            except (KeyError, TypeError, ValueError):
                continue
            events.append((min(start, end), max(start, end)))
    return events


def _set_cmd_arg(cmd: list[str], flag: str, value: str) -> None:
    if flag in cmd:
        idx = cmd.index(flag)
        if idx + 1 >= len(cmd):
            raise ValueError(f"Flag {flag} has no value in command: {' '.join(cmd)}")
        cmd[idx + 1] = value
    else:
        cmd.extend([flag, value])


def _read_prediction_rows(predicted_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not predicted_csv.exists():
        raise FileNotFoundError(f"Predicted CSV not found: {predicted_csv}")

    with predicted_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_rows_csv(rows: list[dict[str, str]], fieldnames: list[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_caller_over_references(
    caller_cmd: list[str],
    reference_fastas: list[Path],
    aggregated_output_csv: Path,
    caller_runs_dir: Path,
) -> dict[str, object]:
    """Run caller once per reference FASTA and aggregate all predictions."""

    caller_runs_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, str]] = []
    aggregate_fieldnames: list[str] = []
    per_reference_counts: dict[str, int] = {}

    for reference_fasta in tqdm(reference_fastas):
        reference_stem = reference_fasta.stem
        print("----> run_caller_over_references:",reference_stem)
        reference_run_dir = caller_runs_dir / reference_stem
        reference_run_dir.mkdir(parents=True, exist_ok=True)
        reference_csv = reference_run_dir / "predicted_delvgs.csv"

        cmd = list(caller_cmd)
        _set_cmd_arg(cmd, "--reference-fasta", str(reference_fasta))
        _set_cmd_arg(cmd, "--output-csv", str(reference_csv))
        if "--output-dir" in cmd:
            _set_cmd_arg(cmd, "--output-dir", str(reference_run_dir))
        if "--output-tag" in cmd:
            _set_cmd_arg(cmd, "--output-tag", reference_stem)

        run_step(f"STEP 2 Run Caller [{reference_stem}]", cmd)

        source_fields, source_rows = _read_prediction_rows(reference_csv)
        per_reference_counts[reference_stem] = len(source_rows)

        for field in source_fields:
            if field not in aggregate_fieldnames:
                aggregate_fieldnames.append(field)
        for field in ("reference_segment", "reference_fasta"):
            if field not in aggregate_fieldnames:
                aggregate_fieldnames.append(field)

        for row in source_rows:
            row["reference_segment"] = reference_stem
            row["reference_fasta"] = str(reference_fasta.resolve())
            aggregate_rows.append(row)

    if not aggregate_fieldnames:
        aggregate_fieldnames = [
            "genome_id",
            "predicted_start",
            "predicted_end",
            "reference_segment",
            "reference_fasta",
        ]

    _write_rows_csv(aggregate_rows, aggregate_fieldnames, aggregated_output_csv)
    print(f"Saved aggregated predictions: {aggregated_output_csv}")

    return {
        "reference_count": len(reference_fastas),
        "references": [str(p.resolve()) for p in reference_fastas],
        "events_per_reference": per_reference_counts,
        "predicted_event_count_total": len(aggregate_rows),
    }


def save_results(
    predicted_csv: Path,
    results_root: Path,
    input_summary: dict[str, object],
    caller_cmd: list[str],
) -> Path:
    """Save standardized output and metadata."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_predicted = run_dir / "predicted_delvgs.csv"
    shutil.copy2(predicted_csv, saved_predicted)

    events = _load_predictions(predicted_csv)
    deletion_sizes = [end - start for start, end in events]
    summary = {
        "created_at": timestamp,
        "input": input_summary,
        "caller_cmd": caller_cmd,
        "predicted_event_count": len(events),
        "min_deletion_size": min(deletion_sizes) if deletion_sizes else None,
        "max_deletion_size": max(deletion_sizes) if deletion_sizes else None,
        "median_deletion_size": (
            sorted(deletion_sizes)[len(deletion_sizes) // 2] if deletion_sizes else None
        ),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {saved_predicted}")
    print(f"Saved: {summary_path}")
    return run_dir


def visualize_results(predicted_csv: Path, run_dir: Path) -> None:
    """Create and save simple diagnostic plots from predicted events."""

    events = _load_predictions(predicted_csv)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not events:
        placeholder = plot_dir / "no_predictions.txt"
        placeholder.write_text("No predicted events available for plotting.\n", encoding="utf-8")
        print(f"Saved: {placeholder}")
        return

    starts = [s for s, _ in events]
    ends = [e for _, e in events]
    sizes = [e - s for s, e in events]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=40, color="#3465a4", edgecolor="black", linewidth=0.3)
    plt.title("Predicted Deletion Size Distribution")
    plt.xlabel("Deletion size (bp)")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_path = plot_dir / "deletion_size_histogram.png"
    plt.savefig(hist_path, dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(starts, ends, s=10, alpha=0.5, color="#4e9a06")
    plt.title("Predicted Breakpoints (Start vs End)")
    plt.xlabel("Predicted start")
    plt.ylabel("Predicted end")
    plt.tight_layout()
    scatter_path = plot_dir / "breakpoint_scatter.png"
    plt.savefig(scatter_path, dpi=180)
    plt.close()

    print(f"Saved: {hist_path}")
    print(f"Saved: {scatter_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run identification workflow on real FASTQ input.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/input"),
        help="Directory containing real FASTQ input.",
    )
    parser.add_argument(
        "--input-fastq",
        type=Path,
        default=None,
        help="Optional explicit FASTQ R1 input path. If omitted, auto-detect one file in --input-dir.",
    )
    parser.add_argument(
        "--input-fastq-r2",
        type=Path,
        default=None,
        help="Optional FASTQ R2 input path for paired-end runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("output/identifacator"),
        help="Directory where final results/plots are archived.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing reference FASTA files.",
    )
    parser.add_argument(
        "--reference-accessions",
        type=Path,
        default=Path("data/reference_accessions.txt"),
        help="Optional accession/segment list used to choose and order reference FASTAs.",
    )
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--virema-script", type=Path, default=Path("src/ViReMa/ViReMa.py"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python_executable

    input_fastq = args.input_fastq if args.input_fastq is not None else _discover_single_fastq(args.input_dir)
    input_fastq_r2 = args.input_fastq_r2

    prepared_r1 = Path("output/reads_R1.fastq")
    prepared_r2 = Path("output/reads_R2.fastq")
    predicted_csv = Path("output/predicted_delvgs.csv")
    reference_fastas = _discover_reference_fastas(args.reference_dir, args.reference_accessions)

    input_summary: dict[str, object] = {}
    caller_summary: dict[str, object] = {}
    saved_runs: list[Path] = []

    run_action_step(
        "STEP 1 Read Input Data",
        lambda: input_summary.update(
            prepare_input_data(
                input_fastq=input_fastq,
                output_r1=prepared_r1,
                output_r2=prepared_r2,
                input_fastq_r2=input_fastq_r2,
            )
        ),
    )

    # NOTE: caller_cmd must stay exactly aligned with run_pipeline.py
    caller_cmd = [
        py,
        "src/run_virema.py",
        "--reference-fasta",
        "data/PB2.fasta",
        "--reads-r1",
        "output/reads_R1.fastq",
        "--reads-r2",
        "output/reads_R2.fastq",
        "--output-dir",
        "output/virema",
        "--output-csv",
        "output/predicted_delvgs.csv",
        "--virema-script",
        str(args.virema_script),
        "--python-executable",
        py,
    ]

    # caller_cmd = [
    # py,
    # "src/run_DItector.py",
    # "--reference-fasta",
    # "data/PB2.fasta",
    # "--reads-r1",
    # "output/reads_R1.fastq",
    # "--reads-r2",
    # "output/reads_R2.fastq",
    # "--output-dir",
    # "output/ditector",
    # "--output-csv",
    # "output/predicted_delvgs.csv",
    # "--python-executable",
    # py,
    # "--output-tag",
    # "pb2_sim",
    # "--ditector-script",
    # "src/DI-tector/DI-tector_06.py"
    # ##### "--virema-script",
    # ##### str(args.virema_script),  # optional legacy compatibility
    # ]

#     caller_cmd = [
#     py,
#     "src/run_DVGfinder.py",
#     "--reference-fasta",
#     "data/PB2.fasta",
#     "--reads-r1",
#     "output/reads_R1.fastq",
#     "--reads-r2",
#     "output/reads_R2.fastq",
#     "--output-dir",
#     "output/dvgfinder",
#     "--output-csv",
#     "output/predicted_delvgs.csv",
#     "--output-tag",
#     "pb2_sim",
#     "--threads",
#     "4",
#     "--margin",
#     "5",
#     "--ml-threshold",
#     "0.5",
#     "--polarity",
#     "1",
#     "--conda-executable",
#     "conda",
#     "--conda-env-name",
#     "dvgfinder_env",
#     "--auto-create-env",  # optional
    # ]

    run_action_step(
        "STEP 2 Run Caller",
        lambda: caller_summary.update(
            run_caller_over_references(
                caller_cmd=caller_cmd,
                reference_fastas=reference_fastas,
                aggregated_output_csv=predicted_csv,
                caller_runs_dir=args.results_dir / "caller_runs",
            )
        ),
    )

    input_summary["reference_fastas"] = [str(p.resolve()) for p in reference_fastas]
    input_summary["caller_summary"] = caller_summary

    run_action_step(
        "STEP 3 Save Results",
        lambda: saved_runs.append(
            save_results(
                predicted_csv=predicted_csv,
                results_root=args.results_dir,
                input_summary=input_summary,
                caller_cmd=caller_cmd,
            )
        ),
    )

    if not saved_runs:
        raise RuntimeError("Internal error: run_dir was not set after saving results.")
    run_dir = saved_runs[-1]

    run_action_step(
        "STEP 4 Visualize Results",
        lambda: visualize_results(predicted_csv=predicted_csv, run_dir=run_dir),
    )

    print("\nIDENTIFACATOR pipeline completed successfully.")


if __name__ == "__main__":
    main()
