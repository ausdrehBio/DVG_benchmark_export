"""Split a multi-segment influenza reference FASTA into per-segment FASTA files.

Example:
    python3 src/split_reference.py \
        --input-fasta data/wildgoose/H5N1_G.fasta \
        --output-dir data/wildgoose/H5N1_G_segments
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


SEGMENT_NUMBER_TO_NAME = {
    "1": "PB2",
    "2": "PB1",
    "3": "PA",
    "4": "HA",
    "5": "NP",
    "6": "NA",
    "7": "M",
    "8": "NS",
}

SEGMENT_NAMES = tuple(SEGMENT_NUMBER_TO_NAME.values())
SEGMENT_TOKEN_ALIASES = {
    "PB2": "PB2",
    "PB1": "PB1",
    "PA": "PA",
    "HA": "HA",
    "NP": "NP",
    "NA": "NA",
    "M": "M",
    "M1": "M",
    "M2": "M",
    "MP": "M",
    "NS": "NS",
    "NS1": "NS",
    "NS2": "NS",
    "NEP": "NS",
}


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


def parse_fasta(path: Path) -> list[FastaRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {path}")

    records: list[FastaRecord] = []
    header: str | None = None
    seq_parts: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(seq_parts).upper()
                    if not sequence:
                        raise ValueError(f"Record '{header}' has an empty sequence in {path}")
                    records.append(FastaRecord(header=header, sequence=sequence))
                header = line[1:].strip()
                seq_parts = []
            else:
                if header is None:
                    raise ValueError(f"Invalid FASTA format in {path}: sequence before first header")
                seq_parts.append(line)

    if header is not None:
        sequence = "".join(seq_parts).upper()
        if not sequence:
            raise ValueError(f"Record '{header}' has an empty sequence in {path}")
        records.append(FastaRecord(header=header, sequence=sequence))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def infer_segment_name(header: str) -> str:
    # Priority 1: explicit token in parentheses, e.g. "(PB2)"
    paren_hits = re.findall(r"\(([A-Za-z0-9-]+)\)", header)
    for token in reversed(paren_hits):
        normalized = token.upper()
        if normalized in SEGMENT_TOKEN_ALIASES:
            return SEGMENT_TOKEN_ALIASES[normalized]

    # Priority 2: "segment 1", "segment 2", ... pattern
    segment_match = re.search(r"\bsegment\s+([1-8])\b", header, flags=re.IGNORECASE)
    if segment_match:
        return SEGMENT_NUMBER_TO_NAME[segment_match.group(1)]

    # Priority 3: infer from common gene/protein terms in the header.
    upper = header.upper()
    if re.search(r"\bPB2\b", upper):
        return "PB2"
    if re.search(r"\bPB1\b", upper):
        return "PB1"
    if re.search(r"\bPA\b|\bPA-X\b", upper):
        return "PA"
    if re.search(r"\bHEMAGGLUTININ\b|\bHA\b", upper):
        return "HA"
    if re.search(r"\bNUCLEOCAPSID\b|\bNUCLEOPROTEIN\b|\bNP\b", upper):
        return "NP"
    if re.search(r"\bNEURAMINIDASE\b|\bNA\b", upper):
        return "NA"
    if re.search(r"\bMATRIX\b|\bM1\b|\bM2\b|\bMP\b", upper):
        return "M"
    if re.search(r"\bNONSTRUCTURAL\b|\bNUCLEAR EXPORT\b|\bNS1\b|\bNS2\b|\bNEP\b|\bNS\b", upper):
        return "NS"

    raise ValueError(
        "Could not infer influenza segment name from header: "
        f"{header!r}. Expected one of: '(PB2)'-style token, 'segment <1-8>', "
        "or recognizable influenza gene names (e.g., M1/M2, NEP/NS1)."
    )


def wrap_sequence(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def split_reference(input_fasta: Path, output_dir: Path, expected_segments: int | None) -> list[Path]:
    records = parse_fasta(input_fasta)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    seen_segments: set[str] = set()

    for record in records:
        segment = infer_segment_name(record.header)
        if segment in seen_segments:
            raise ValueError(
                f"Duplicate segment '{segment}' detected in {input_fasta}. "
                "Expected one record per segment."
            )
        seen_segments.add(segment)

        out_path = output_dir / f"{segment}.fasta"
        fasta_text = f">{record.header}\n{wrap_sequence(record.sequence)}\n"
        out_path.write_text(fasta_text, encoding="utf-8")
        written.append(out_path)

    if expected_segments is not None and len(written) != expected_segments:
        raise ValueError(
            f"Expected {expected_segments} segments, but wrote {len(written)} files "
            f"from {input_fasta}."
        )

    return sorted(written, key=lambda p: p.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split multi-segment influenza reference FASTA into one FASTA per segment."
    )
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=Path("data/wildgoose/H5N1_G.fasta"),
        help="Path to input multi-record FASTA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wildgoose/H5N1_G_segments"),
        help="Directory to write per-segment FASTA files.",
    )
    parser.add_argument(
        "--expected-segments",
        type=int,
        default=8,
        help="Expected number of output segments (use 0 to disable check).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = None if args.expected_segments == 0 else args.expected_segments
    written = split_reference(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        expected_segments=expected,
    )
    print(f"Wrote {len(written)} segment FASTA files to: {args.output_dir.resolve()}")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
