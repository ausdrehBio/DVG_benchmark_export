"""End-to-end DelVG benchmark pipeline wrapper.

Runs, in order:
1) generate_delvgs.py
2) simulate_reads.py
3) run_caller.py
4) evaluate.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> None:
    """Execute one pipeline step and stop on failure."""

    print(f"\n[{name}] Running:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full synthetic DelVG benchmark pipeline.")

    parser.add_argument("--number-of-sequences", type=int, default=100)
    parser.add_argument("--min-deletion-size", type=int, default=50)
    parser.add_argument("--max-deletion-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--read-length", type=int, default=150)
    parser.add_argument("--coverage-depth", type=float, default=200.0)
    parser.add_argument("--error-rate", type=float, default=0.001)

    parser.add_argument("--tolerance-window", type=int, default=5)

    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--virema-script", type=Path, default=Path("src/ViReMa/ViReMa.py"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = args.python_executable

    generate_cmd = [
        py,
        "src/generate_delvgs.py",
        "--input-fasta",
        "data/PB2.fasta",
        "--output-fasta",
        "output/synthetic_delvgs.fasta",
        "--output-csv",
        "output/ground_truth.csv",
        "--number-of-sequences",
        str(args.number_of_sequences),
        "--min-deletion-size",
        str(args.min_deletion_size),
        "--max-deletion-size",
        str(args.max_deletion_size),
        "--seed",
        str(args.seed),
    ]

    simulate_cmd = [
        py,
        "src/simulate_reads.py",
        "--input-fasta",
        "output/synthetic_delvgs.fasta",
        "--output-r1",
        "output/reads_R1.fastq",
        "--output-r2",
        "output/reads_R2.fastq",
        "--read-length",
        str(args.read_length),
        "--coverage-depth",
        str(args.coverage_depth),
        "--error-rate",
        str(args.error_rate),
    ]

    # caller_cmd = [
    #     py,
    #     "src/run_virema.py",
    #     "--reference-fasta",
    #     "data/PB2.fasta",
    #     "--reads-r1",
    #     "output/reads_R1.fastq",
    #     "--reads-r2",
    #     "output/reads_R2.fastq",
    #     "--output-dir",
    #     "output/virema",
    #     "--output-csv",
    #     "output/predicted_delvgs.csv",
    #     "--virema-script",
    #     str(args.virema_script),
    #     "--python-executable",
    #     py,
    # ]

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

    caller_cmd = [
    py,
    "src/run_DVGfinder.py",
    "--reference-fasta",
    "data/PB2.fasta",
    "--reads-r1",
    "output/reads_R1.fastq",
    "--reads-r2",
    "output/reads_R2.fastq",
    "--output-dir",
    "output/dvgfinder",
    "--output-csv",
    "output/predicted_delvgs.csv",
    "--output-tag",
    "pb2_sim",
    "--threads",
    "4",
    "--margin",
    "5",
    "--ml-threshold",
    "0.5",
    "--polarity",
    "1",
    "--conda-executable",
    "conda",
    "--conda-env-name",
    "dvgfinder_env",
    "--auto-create-env",  # optional
]


    evaluate_cmd = [
        py,
        "src/evaluate.py",
        "--predicted-csv",
        "output/predicted_delvgs.csv",
        "--ground-truth-csv",
        "output/ground_truth.csv",
        "--tolerance-window",
        str(args.tolerance_window),
    ]

    run_step("STEP 1 Generate DelVGs", generate_cmd)
    run_step("STEP 2 Simulate Reads", simulate_cmd)
    run_step("STEP 3 Run Caller", caller_cmd)
    run_step("STEP 4 Evaluate", evaluate_cmd)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
