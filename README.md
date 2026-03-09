# DVG Benchmark (Synthetic DelVG Pipeline)

This repository provides an end-to-end benchmarking pipeline for deletion viral genome (DelVG) detection on synthetic data.

Current implemented caller integration:
- ViReMa (`src/ViReMa.py`) via `src/run_caller.py`

## What The Pipeline Does

The benchmark runs four stages:

1. Generate synthetic DelVG sequences from a wild-type reference FASTA.
2. Simulate paired-end Illumina reads from those sequences with ART.
3. Run a caller (currently ViReMa) and normalize raw caller output into a common prediction CSV.
4. Evaluate predictions against ground truth with precision/recall/F1.

## Repository Layout

```text
.
├── data/
│   ├── PB2.fasta
│   ├── PB2.fasta*.ebwt                # Bowtie index files (generated artifacts)
│   └── vodka_data/                    # Additional input data not used by default pipeline
├── output/                            # Pipeline outputs (ignored by git)
└── src/
    ├── generate_delvgs.py             # Step 1
    ├── simulate_reads.py              # Step 2
    ├── run_caller.py                  # Step 3 (ViReMa wrapper + normalization)
    ├── evaluate.py                    # Step 4
    ├── run_pipeline.py                # Orchestrates steps 1-4
    ├── ViReMa.py
    ├── Compiler_Module.py
    ├── ConfigViReMa.py
    └── test_*.py
```

## Requirements

### Python

- Python 3.10+ (code uses modern typing syntax like `str | None`)
- `pytest` (for running tests)

### External Tools

- ART Illumina simulator executable available as `art_illumina` (or pass `--simulator-cmd`).
- Bowtie (used by ViReMa; includes `bowtie` and `bowtie-build`).
- ViReMa script is included in this repository (`src/ViReMa.py`).
- `samtools` is optional unless you enable ViReMa options that require BAM/coverage outputs.

### Recommended Environment Setup (Example)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip pytest
```

Install ART/Bowtie with your preferred package manager (conda, apt, brew, etc.) and ensure binaries are on `PATH`.

## Quick Start

Run the full pipeline from repository root:

```bash
python3 src/run_pipeline.py
```

With the defaults in `run_pipeline.py`, this executes:

- Step 1: generate 1000 DelVG sequences (+ WT) with seed 17
- Step 2: simulate 150 bp paired-end reads at 200x depth
- Step 3: run ViReMa and standardize predictions
- Step 4: print evaluation metrics to stdout

## Main Output Files

After a successful run, key outputs are:

- `output/synthetic_delvgs.fasta`
- `output/ground_truth.csv`
- `output/reads_R1.fastq`
- `output/reads_R2.fastq`
- `output/virema/` (raw ViReMa outputs)
- `output/predicted_delvgs.csv`

`evaluate.py` currently prints metrics to stdout and does not write a report file by default.

## Run Each Step Manually

### 1) Generate synthetic DelVG FASTA + ground truth

```bash
python3 src/generate_delvgs.py \
  --input-fasta data/PB2.fasta \
  --output-fasta output/synthetic_delvgs.fasta \
  --output-csv output/ground_truth.csv \
  --number-of-sequences 1000 \
  --min-deletion-size 50 \
  --max-deletion-size 500 \
  --seed 17
```

Notes:
- Coordinates are 1-based inclusive.
- A WT entry is always included once (`deletion_size=0`, `deletion_start=NA`, `deletion_end=NA`).

### 2) Simulate reads with ART

```bash
python3 src/simulate_reads.py \
  --input-fasta output/synthetic_delvgs.fasta \
  --output-r1 output/reads_R1.fastq \
  --output-r2 output/reads_R2.fastq \
  --read-length 150 \
  --coverage-depth 200 \
  --error-rate 0.001
```

Notes:
- `--error-rate` is mapped to ART quality shift (`-qs` / `-qs2`), because ART does not take direct error rate.
- FASTQ outputs are validated for basic structural correctness.

### 3) Run ViReMa wrapper + normalize predictions

```bash
python3 src/run_caller.py \
  --reference-fasta data/PB2.fasta \
  --reads-r1 output/reads_R1.fastq \
  --reads-r2 output/reads_R2.fastq \
  --output-dir output/virema \
  --output-csv output/predicted_delvgs.csv \
  --virema-script src/ViReMa.py \
  --python-executable python3
```

Notes:
- R1 and R2 are merged into a single FASTQ for ViReMa input.
- Normalization enforces:
  - schema: `genome_id,predicted_start,predicted_end`
  - sorted `(genome_id, start, end)`
  - deduplication
  - optional minimum deletion size filtering (`--min-deletion-size`)
  - optional genome filter (`--genome-filter`)

You can also skip running ViReMa and only parse existing raw output:

```bash
python3 src/run_caller.py \
  --raw-events path/to/raw_events_file \
  --output-csv output/predicted_delvgs.csv
```

### 4) Evaluate against ground truth

```bash
python3 src/evaluate.py \
  --predicted-csv output/predicted_delvgs.csv \
  --ground-truth-csv output/ground_truth.csv \
  --tolerance-window 5
```

Printed metrics:
- True Positives
- False Positives
- False Negatives
- Precision
- Recall
- F1-Score

## Evaluation Semantics

Evaluation uses one-to-one matching:

- Events must have identical `genome_id`.
- Both breakpoints must be within `+/-tolerance_window`.
- If multiple truth events match one prediction, the closest (minimum absolute breakpoint distance) is selected.
- WT rows (`deletion_size=0`) are excluded from truth events.

## Data Formats

### Ground truth CSV (`output/ground_truth.csv`)

Columns:
- `sequence_id` (e.g. `PB2|DelVG_37`)
- `deletion_start` (`NA` for WT)
- `deletion_end` (`NA` for WT)
- `deletion_size`

### Standardized prediction CSV (`output/predicted_delvgs.csv`)

Columns:
- `genome_id`
- `predicted_start`
- `predicted_end`

### Raw caller parsing

`run_caller.py` can parse several tabular raw formats (CSV/TSV/BEDPE-like), with or without headers, as long as reference and breakpoint columns can be inferred.

## Running Tests

Run all tests:

```bash
python3 -m pytest src/test_*.py
```

Or quick mode:

```bash
python3 -m pytest -q src/test_*.py
```

Covered checkpoints:
- `test_generate.py`: DelVG generation and coordinate integrity
- `test_simulate_reads.py`: simulation wrapper behavior and FASTQ validation
- `test_run_caller.py`: raw parsing and normalization logic
- `test_evaluate.py`: metric computation and tolerance behavior

## Reproducibility Notes

- Set `--seed` in generation for deterministic DelVG creation.
- Keep read simulation parameters fixed (`read-length`, `coverage-depth`, `error-rate`) when comparing runs.
- Keep evaluation `--tolerance-window` fixed across experiments.

## Troubleshooting

### `art_illumina` not found

Error usually looks like:
- `Simulator executable 'art_illumina' was not found on PATH`

Fix:
- Install ART and ensure executable is on `PATH`, or pass `--simulator-cmd /absolute/path/to/art_illumina`.

### ART runtime library error (macOS / conda)

If stderr mentions `dyld`, `Library not loaded`, or `libgsl`, install matching runtime libs in the same environment and reinstall ART in that environment.

### ViReMa completed but no known raw output file found

If `run_caller.py` cannot locate an expected ViReMa result file:
- check `--output-dir`
- check `--output-tag`
- try with default overwrite behavior (do not pass `--no-overwrite`) to avoid timestamped output folder variation

## Notes On Current Scope

- The project currently benchmarks a single integrated caller (ViReMa).
- The normalization/evaluation path is already structured around a common prediction schema, which is the base needed for adding more callers later.

## License

No license file is currently present in this repository. Add one before distributing or reusing outside your private/project context.
