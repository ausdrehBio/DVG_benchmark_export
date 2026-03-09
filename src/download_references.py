"""Download reference FASTA files from a list of accession IDs.

Default behavior:
- reads accession IDs from data/reference_accessions.txt
- downloads FASTA records from NCBI nuccore (E-utilities efetch)
- writes one FASTA file per accession into data/
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AccessionEntry:
    accession: str
    label: str | None


def _sanitize_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value.strip())


def parse_accessions(accession_file: Path) -> list[AccessionEntry]:
    """Read accession IDs and optional labels from a text file.

    Supported line format:
    - AF389115.1
    - AF389115.1 #PB2
    """

    if not accession_file.exists():
        raise FileNotFoundError(f"Accession list not found: {accession_file}")

    entries: list[AccessionEntry] = []
    seen_accessions: set[str] = set()
    seen_labels: set[str] = set()

    with accession_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            payload, comment = (raw.split("#", maxsplit=1) + [""])[:2]
            tokens = payload.strip().split()
            if not tokens:
                continue

            accession = tokens[0]
            label = comment.strip() or None

            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)

            if label:
                safe_label = _sanitize_stem(label)
                if safe_label in seen_labels:
                    raise ValueError(
                        f"Duplicate segment label in {accession_file}: {label!r}. "
                        "Each label must be unique."
                    )
                seen_labels.add(safe_label)

            entries.append(AccessionEntry(accession=accession, label=label))

    if not entries:
        raise ValueError(f"No accession IDs found in: {accession_file}")
    return entries


def accession_to_filename(accession: str) -> str:
    return f"{_sanitize_stem(accession)}.fasta"


def label_to_filename(label: str) -> str:
    return f"{_sanitize_stem(label)}.fasta"


def build_efetch_url(accession: str, email: str | None, api_key: str | None) -> str:
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "tool": "dvg_benchmark",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    query = urllib.parse.urlencode(params)
    return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}"


def fetch_fasta(accession: str, email: str | None, api_key: str | None, timeout: int, retries: int) -> str:
    url = build_efetch_url(accession, email=email, api_key=api_key)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = response.read().decode("utf-8", errors="replace").strip()
            if not payload.startswith(">"):
                raise RuntimeError(
                    f"NCBI response for accession '{accession}' is not FASTA. "
                    f"First 120 chars: {payload[:120]!r}"
                )
            return payload + "\n"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as err:
            last_error = err
            if attempt == retries:
                break
            time.sleep(0.8 * attempt)

    raise RuntimeError(f"Failed to download accession '{accession}' after {retries} attempts: {last_error}")


def download_references(
    accession_file: Path,
    output_dir: Path,
    email: str | None,
    api_key: str | None,
    force: bool,
    delay_seconds: float,
    timeout: int,
    retries: int,
) -> list[Path]:
    entries = parse_accessions(accession_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []

    for index, entry in enumerate(entries, start=1):
        accession = entry.accession
        output_name = label_to_filename(entry.label) if entry.label else accession_to_filename(accession)
        output_path = output_dir / output_name

        if output_path.exists() and not force:
            print(f"[{index}/{len(entries)}] Skipping existing: {output_path.name}")
            downloaded.append(output_path)
            continue

        label_note = f" ({entry.label})" if entry.label else ""
        print(f"[{index}/{len(entries)}] Downloading {accession}{label_note} -> {output_path.name}")
        fasta_text = fetch_fasta(
            accession=accession,
            email=email,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
        )
        output_path.write_text(fasta_text, encoding="utf-8")
        downloaded.append(output_path)

        if delay_seconds > 0 and index < len(entries):
            time.sleep(delay_seconds)

    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FASTA references listed as accession IDs in a text file."
    )
    parser.add_argument(
        "--accession-file",
        type=Path,
        default=Path("data/reference_accessions.txt"),
        help="Text file containing accession IDs (one per line), optionally with labels like: AF389115.1 #PB2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where downloaded FASTA files are stored.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Optional email sent to NCBI E-utilities.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional NCBI API key for higher request limits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing FASTA files if already present.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.34,
        help="Delay between requests (default follows NCBI public rate limit).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout per request in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for each accession.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_files = download_references(
        accession_file=args.accession_file,
        output_dir=args.output_dir,
        email=args.email,
        api_key=args.api_key,
        force=args.force,
        delay_seconds=args.delay_seconds,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(f"\nDone. FASTA files ready: {len(output_files)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
