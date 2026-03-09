"""Generate synthetic DelVG sequences from a wild-type FASTA sequence."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SequenceRecord:
    """Container for a generated sequence and its deletion metadata."""

    sequence_id: str
    sequence: str
    deletion_start: int | None
    deletion_end: int | None
    deletion_size: int


def read_first_fasta(fasta_path: Path) -> tuple[str, str]:
    """Read the first FASTA record from *fasta_path*.

    Parameters
    ----------
    fasta_path:
        Path to the FASTA file.

    Returns
    -------
    tuple[str, str]
        FASTA identifier (first token after '>') and uppercase sequence.
    """

    header: str | None = None
    sequence_chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].strip()
            else:
                if header is None:
                    raise ValueError("Invalid FASTA: sequence encountered before header.")
                sequence_chunks.append(line)

    if header is None:
        raise ValueError(f"No FASTA header found in {fasta_path}")

    if not sequence_chunks:
        raise ValueError(f"No FASTA sequence found in {fasta_path}")

    sequence_id = header.split()[0]
    return sequence_id, "".join(sequence_chunks).upper()


def _validate_inputs(
    wt_sequence: str,
    number_of_sequences: int,
    min_deletion_size: int,
    max_deletion_size: int,
) -> None:
    if number_of_sequences < 0:
        raise ValueError("number_of_sequences must be >= 0")
    if len(wt_sequence) < 2:
        raise ValueError("WT sequence must be at least 2 bases long.")
    if min_deletion_size < 1:
        raise ValueError("min_deletion_size must be >= 1")
    if min_deletion_size > max_deletion_size:
        raise ValueError("min_deletion_size must be <= max_deletion_size")
    if max_deletion_size >= len(wt_sequence):
        raise ValueError("max_deletion_size must be smaller than WT sequence length.")


def _excise_deletion(wt_sequence: str, deletion_start: int, deletion_end: int) -> str:
    """Return WT sequence with inclusive interval [deletion_start, deletion_end] removed.

    Coordinates are 1-based and inclusive.
    """

    start_idx = deletion_start - 1
    end_idx = deletion_end
    return wt_sequence[:start_idx] + wt_sequence[end_idx:]


def generate_delvg_records(
    wt_id: str,
    wt_sequence: str,
    number_of_sequences: int,
    min_deletion_size: int,
    max_deletion_size: int,
    rng: random.Random | None = None,
) -> list[SequenceRecord]:
    """Generate a WT+DelVG record set with deletion metadata.

    Deletion coordinates are 1-based and inclusive.
    """

    _validate_inputs(wt_sequence, number_of_sequences, min_deletion_size, max_deletion_size)
    generator = rng if rng is not None else random.Random()
    wt_len = len(wt_sequence)

    records: list[SequenceRecord] = [
        SequenceRecord(
            sequence_id=f"{wt_id}|WT",
            sequence=wt_sequence,
            deletion_start=None,
            deletion_end=None,
            deletion_size=0,
        )
    ]

    for idx in range(1, number_of_sequences + 1):
        deletion_size = generator.randint(min_deletion_size, max_deletion_size)
        deletion_start = generator.randint(1, wt_len - deletion_size + 1)
        deletion_end = deletion_start + deletion_size - 1
        delvg_sequence = _excise_deletion(wt_sequence, deletion_start, deletion_end)
        records.append(
            SequenceRecord(
                sequence_id=f"{wt_id}|DelVG_{idx}",
                sequence=delvg_sequence,
                deletion_start=deletion_start,
                deletion_end=deletion_end,
                deletion_size=deletion_size,
            )
        )

    return records


def write_fasta(records: Iterable[SequenceRecord], output_path: Path, line_width: int = 80) -> None:
    """Write records to FASTA format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.sequence_id}\n")
            for i in range(0, len(record.sequence), line_width):
                handle.write(record.sequence[i : i + line_width] + "\n")


def write_ground_truth_csv(records: Iterable[SequenceRecord], output_path: Path) -> None:
    """Write per-sequence deletion metadata to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sequence_id", "deletion_start", "deletion_end", "deletion_size"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sequence_id": record.sequence_id,
                    "deletion_start": "NA" if record.deletion_start is None else record.deletion_start,
                    "deletion_end": "NA" if record.deletion_end is None else record.deletion_end,
                    "deletion_size": record.deletion_size,
                }
            )


def run_pipeline(
    input_fasta: Path,
    output_fasta: Path,
    output_csv: Path,
    number_of_sequences: int,
    min_deletion_size: int,
    max_deletion_size: int,
    seed: int | None = None,
) -> list[SequenceRecord]:
    """Run DelVG generation and write FASTA + ground truth CSV outputs."""

    wt_id, wt_sequence = read_first_fasta(input_fasta)
    rng = random.Random(seed) if seed is not None else random.Random()
    records = generate_delvg_records(
        wt_id=wt_id,
        wt_sequence=wt_sequence,
        number_of_sequences=number_of_sequences,
        min_deletion_size=min_deletion_size,
        max_deletion_size=max_deletion_size,
        rng=rng,
    )
    write_fasta(records, output_fasta)
    write_ground_truth_csv(records, output_csv)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic DelVG sequences from WT FASTA.")
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=Path("data/PB2.fasta"),
        help="Path to wild-type FASTA (default: data/PB2.fasta)",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=Path("output/synthetic_delvgs.fasta"),
        help="Path to output synthetic FASTA",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/ground_truth.csv"),
        help="Path to output ground-truth CSV",
    )
    parser.add_argument(
        "--number-of-sequences",
        type=int,
        default=100,
        help="Number of DelVG sequences to generate (WT is always included once).",
    )
    parser.add_argument("--min-deletion-size", type=int, default=50, help="Minimum deletion size.")
    parser.add_argument("--max-deletion-size", type=int, default=500, help="Maximum deletion size.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_fasta=args.input_fasta,
        output_fasta=args.output_fasta,
        output_csv=args.output_csv,
        number_of_sequences=args.number_of_sequences,
        min_deletion_size=args.min_deletion_size,
        max_deletion_size=args.max_deletion_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
