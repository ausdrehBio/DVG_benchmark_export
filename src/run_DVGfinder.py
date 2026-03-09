"""Run DVGfinder and normalize output to predicted_delvgs.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import stat
import shutil
import subprocess
import sys
import time
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
_BOWTIE_INDEX_SUFFIXES = (".1.ebwt", ".2.ebwt", ".3.ebwt", ".4.ebwt", ".rev.1.ebwt", ".rev.2.ebwt")
_DVGFINDER_REQUIRED_PYTHON = "3.9"
_DVGFINDER_REQUIRED_PANDAS_PREFIX = "1.2."
_DVGFINDER_REQUIRED_SKLEARN_PREFIX = "1.0."
_PORTABLE_CONDA_SPECS = (
    "python=3.9",
    "pip",
    "bwa",
    "bowtie",
    "samtools",
    "biopython",
    "numpy",
    "pandas=1.2.4",
    "scikit-learn>=1.0,<1.1",
    "pyfastx",
    "matplotlib",
    "seaborn",
    "plotly",
)


def _run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd, env=env)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr if stderr else stdout
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{details}")


def _merge_fastq_files(reads_r1: Path, reads_r2: Path, merged_output: Path) -> None:
    """Concatenate paired-end FASTQ files into a single FASTQ input."""

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    with merged_output.open("w", encoding="utf-8") as out_handle:
        for read_path in (reads_r1, reads_r2):
            with read_path.open("r", encoding="utf-8") as in_handle:
                out_handle.write(in_handle.read())


def _read_first_fasta_id(reference_fasta: Path) -> str:
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    with reference_fasta.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(">"):
                return stripped[1:].split()[0]

    raise ValueError(f"No FASTA header found in reference file: {reference_fasta}")


def _conda_env_exists(conda_executable: str, env_name: str) -> bool:
    result = subprocess.run(
        [conda_executable, "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr if stderr else stdout
        raise RuntimeError(
            f"Failed to inspect conda environments via '{conda_executable} env list --json'.\n{details}"
        )

    payload = json.loads(result.stdout)
    env_paths = payload.get("envs", [])
    return any(Path(path).name == env_name for path in env_paths)


def _inspect_conda_env_runtime(conda_executable: str, env_name: str) -> dict[str, str | None]:
    probe_code = (
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "out = {'python': sys.version.split()[0], 'pandas': None, 'sklearn': None}\n"
        "if importlib.util.find_spec('pandas') is not None:\n"
        "    import pandas as pd\n"
        "    out['pandas'] = pd.__version__\n"
        "if importlib.util.find_spec('sklearn') is not None:\n"
        "    import sklearn\n"
        "    out['sklearn'] = sklearn.__version__\n"
        "print(json.dumps(out))\n"
    )

    result = subprocess.run(
        [conda_executable, "run", "-n", env_name, "python", "-c", probe_code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr if stderr else stdout
        raise RuntimeError(
            f"Failed to inspect runtime versions inside conda env '{env_name}'.\n{details}"
        )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Failed to inspect conda env '{env_name}': no probe output produced.")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse runtime probe output for conda env '{env_name}': {lines[-1]}"
        ) from exc

    python_version = str(payload.get("python", "")).strip() or None
    pandas_version = payload.get("pandas")
    if pandas_version is not None:
        pandas_version = str(pandas_version).strip() or None

    sklearn_version = payload.get("sklearn")
    if sklearn_version is not None:
        sklearn_version = str(sklearn_version).strip() or None

    return {"python": python_version, "pandas": pandas_version, "sklearn": sklearn_version}


def _conda_env_is_compatible(conda_executable: str, env_name: str) -> tuple[bool, str]:
    runtime = _inspect_conda_env_runtime(conda_executable, env_name)
    python_version = runtime.get("python")
    pandas_version = runtime.get("pandas")
    sklearn_version = runtime.get("sklearn")

    problems: list[str] = []
    if python_version is None:
        problems.append("python runtime version is unavailable")
    else:
        py_major_minor = ".".join(python_version.split(".")[:2])
        if py_major_minor != _DVGFINDER_REQUIRED_PYTHON:
            problems.append(
                f"python={python_version} (expected major.minor {_DVGFINDER_REQUIRED_PYTHON})"
            )

    if pandas_version is None:
        problems.append("pandas is not importable")
    elif not pandas_version.startswith(_DVGFINDER_REQUIRED_PANDAS_PREFIX):
        problems.append(
            f"pandas={pandas_version} (expected prefix {_DVGFINDER_REQUIRED_PANDAS_PREFIX})"
        )

    if sklearn_version is None:
        problems.append("scikit-learn is not importable")
    elif not sklearn_version.startswith(_DVGFINDER_REQUIRED_SKLEARN_PREFIX):
        problems.append(
            f"scikit-learn={sklearn_version} (expected prefix {_DVGFINDER_REQUIRED_SKLEARN_PREFIX})"
        )

    return (len(problems) == 0, "; ".join(problems))


def _create_conda_env_from_yaml(conda_executable: str, env_name: str, env_yaml: Path) -> None:
    _run_command([conda_executable, "env", "create", "-n", env_name, "-f", str(env_yaml)])


def _create_portable_conda_env(conda_executable: str, env_name: str) -> None:
    create_cmd = [
        conda_executable,
        "create",
        "-y",
        "-n",
        env_name,
        "-c",
        "conda-forge",
        "-c",
        "bioconda",
        "-c",
        "defaults",
        *_PORTABLE_CONDA_SPECS,
    ]
    _run_command(create_cmd)


def _repair_conda_env_in_place(conda_executable: str, env_name: str) -> None:
    repair_specs = [
        "python=3.9",
        "pandas=1.2.4",
        "scikit-learn>=1.0,<1.1",
        "pyfastx",
    ]
    repair_cmd = [
        conda_executable,
        "install",
        "-y",
        "-n",
        env_name,
        "-c",
        "conda-forge",
        "-c",
        "bioconda",
        "-c",
        "defaults",
        *repair_specs,
    ]
    _run_command(repair_cmd)


def _create_conda_env(conda_executable: str, env_name: str, env_yaml: Path) -> None:
    yaml_error: RuntimeError | None = None
    classic_error: RuntimeError | None = None
    portable_error: RuntimeError | None = None

    try:
        _create_conda_env_from_yaml(conda_executable, env_name, env_yaml)
        return
    except RuntimeError as exc:
        yaml_error = exc

    # Some systems default to libmamba solver and fail on old lock-like specs.
    try:
        _run_command(
            [
                conda_executable,
                "env",
                "create",
                "--solver",
                "classic",
                "-n",
                env_name,
                "-f",
                str(env_yaml),
            ]
        )
        return
    except RuntimeError as exc:
        classic_error = exc

    # Cross-platform fallback for old linux-pinned YAMLs (notably on macOS).
    try:
        _create_portable_conda_env(conda_executable, env_name)
        return
    except RuntimeError as exc:
        portable_error = exc

    os_name = platform.system() or "UnknownOS"
    raise RuntimeError(
        "Failed to create DVGfinder conda environment.\n"
        f"OS: {os_name}\n"
        f"YAML attempt error:\n{yaml_error}\n\n"
        f"YAML classic-solver attempt error:\n{classic_error}\n\n"
        f"Portable fallback attempt error:\n{portable_error}"
    )


def _recreate_conda_env(conda_executable: str, env_name: str, env_yaml: Path) -> None:
    _run_command([conda_executable, "env", "remove", "-n", env_name, "-y"])
    _create_conda_env(conda_executable, env_name, env_yaml)


def _ensure_conda_env(
    conda_executable: str,
    env_name: str,
    env_yaml: Path,
    auto_create_env: bool,
) -> None:
    if "/" not in conda_executable and shutil.which(conda_executable) is None:
        raise FileNotFoundError(f"Conda executable '{conda_executable}' was not found on PATH.")

    allow_incompatible_runtime = False
    exists = _conda_env_exists(conda_executable, env_name)
    if exists:
        compatible, details = _conda_env_is_compatible(conda_executable, env_name)
        if compatible:
            return

        if not auto_create_env:
            raise RuntimeError(
                f"Conda environment '{env_name}' exists but is incompatible with DVGfinder requirements: {details}. "
                "Re-run with --auto-create-env to auto-repair, or recreate an environment with "
                "python 3.9 and pandas 1.2.x."
            )

        try:
            _repair_conda_env_in_place(conda_executable, env_name)
        except RuntimeError as exc:
            if _conda_env_exists(conda_executable, env_name):
                print(
                    "Warning: Could not repair incompatible conda env due setup/network failure; "
                    f"continuing with existing env '{env_name}'.",
                    file=sys.stderr,
                )
                print(f"Repair error: {exc}", file=sys.stderr)
                allow_incompatible_runtime = True
            else:
                raise
    else:
        if not auto_create_env:
            raise FileNotFoundError(
                f"Conda environment '{env_name}' was not found. "
                f"Create it first with: {conda_executable} env create -n {env_name} -f {env_yaml}"
            )

        _create_conda_env(conda_executable, env_name, env_yaml)

    if not _conda_env_exists(conda_executable, env_name):
        raise FileNotFoundError(
            f"Conda environment '{env_name}' is still unavailable after attempted creation."
        )

    compatible, details = _conda_env_is_compatible(conda_executable, env_name)
    if not compatible:
        if allow_incompatible_runtime:
            print(
                f"Warning: Conda environment '{env_name}' remains incompatible: {details}.",
                file=sys.stderr,
            )
            return
        raise RuntimeError(
            f"Conda environment '{env_name}' is incompatible with DVGfinder requirements after setup: {details}"
        )


def _ensure_dvgfinder_runtime_dirs(dvgfinder_dir: Path) -> None:
    for rel_dir in (
        "Outputs",
        "Outputs/alignments",
        "Outputs/virema",
        "Outputs/ditector",
        "FinalReports",
        "OldOutputs",
        "OldOutputs/virema",
        "OldOutputs/ditector",
    ):
        (dvgfinder_dir / rel_dir).mkdir(parents=True, exist_ok=True)


def _ensure_dvgfinder_scripts_executable(dvgfinder_dir: Path) -> None:
    """Ensure DVGfinder helper shell scripts have execute permissions."""

    required_scripts = [
        dvgfinder_dir / "Models" / "extract_H_reads.sh",
        dvgfinder_dir / "Models" / "extract_recombination_events_virema.sh",
        dvgfinder_dir / "Models" / "extract_recombination_events_ditector.sh",
    ]

    for script in required_scripts:
        if not script.exists():
            raise FileNotFoundError(f"Required DVGfinder helper script not found: {script}")

        mode = script.stat().st_mode
        execute_mask = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        if (mode & execute_mask) == 0:
            script.chmod(mode | execute_mask)


def _archive_stale_sample_artifacts(dvgfinder_dir: Path, sample_name: str) -> None:
    """Archive stale per-sample outputs that can break reruns with same sample tag."""

    outputs_dir = dvgfinder_dir / "Outputs"
    final_reports_dir = dvgfinder_dir / "FinalReports"

    candidates: list[Path] = []
    candidates.extend(outputs_dir.glob(f"virema/{sample_name}*"))
    candidates.extend(outputs_dir.glob(f"ditector/{sample_name}*"))
    candidates.append(outputs_dir / "alignments" / sample_name)
    candidates.extend(outputs_dir.glob(f"{sample_name}_from_raw_*"))
    candidates.append(outputs_dir / f"{sample_name}_unificated_table.csv")
    candidates.append(outputs_dir / f"{sample_name}_df_ML.csv")
    candidates.append(final_reports_dir / sample_name)

    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)

    if not existing:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_root = dvgfinder_dir / "OldOutputs" / "wrapper_preclean" / f"{sample_name}_{timestamp}"

    for path in existing:
        target = archive_root / path.relative_to(dvgfinder_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def _conda_env_has_module(conda_executable: str, env_name: str, module_name: str) -> bool:
    probe_code = (
        "import importlib.util\n"
        "import sys\n"
        f"sys.stdout.write('1' if importlib.util.find_spec('{module_name}') is not None else '0')\n"
    )
    result = subprocess.run(
        [conda_executable, "run", "-n", env_name, "python", "-c", probe_code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return False
    return lines[-1] == "1"


def _write_datapane_shim(shim_dir: Path) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_module = shim_dir / "datapane.py"
    shim_module.write_text(
        "\n".join(
            [
                "\"\"\"Minimal datapane compatibility shim for DVGfinder pipeline runs.\"\"\"",
                "",
                "from __future__ import annotations",
                "",
                "from pathlib import Path",
                "",
                "",
                "class _Block:",
                "    def __init__(self, *args, **kwargs):",
                "        self.args = args",
                "        self.kwargs = kwargs",
                "",
                "",
                "class HTML(_Block):",
                "    pass",
                "",
                "",
                "class Text(_Block):",
                "    pass",
                "",
                "",
                "class DataTable(_Block):",
                "    pass",
                "",
                "",
                "class Plot(_Block):",
                "    pass",
                "",
                "",
                "class Select(_Block):",
                "    pass",
                "",
                "",
                "class Page(_Block):",
                "    pass",
                "",
                "",
                "class Report:",
                "    def __init__(self, *pages, **kwargs):",
                "        self.pages = pages",
                "        self.kwargs = kwargs",
                "",
                "    def save(self, path=None, open=False, **kwargs):",
                "        if not path:",
                "            return",
                "        Path(path).write_text(",
                "            \"<html><body><h1>DVGfinder Report Placeholder</h1>\"",
                "            \"<p>Generated without datapane dependency.</p></body></html>\",",
                "            encoding=\"utf-8\",",
                "        )",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_pyfastx_shim(shim_dir: Path) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_module = shim_dir / "pyfastx.py"
    shim_module.write_text(
        "\n".join(
            [
                "\"\"\"Minimal pyfastx compatibility shim for DVGfinder pipeline runs.\"\"\"",
                "",
                "from __future__ import annotations",
                "",
                "import gzip",
                "",
                "",
                "class Fastq:",
                "    def __init__(self, path):",
                "        p = str(path)",
                "        opener = gzip.open if p.endswith('.gz') else open",
                "        self._count = 0",
                "        total_len = 0",
                "        total_gc = 0",
                "        max_len = 0",
                "        min_len = None",
                "        max_qual = None",
                "        min_qual = None",
                "",
                "        with opener(p, 'rt', encoding='utf-8', errors='replace') as handle:",
                "            while True:",
                "                header = handle.readline()",
                "                if not header:",
                "                    break",
                "                seq = handle.readline().strip()",
                "                handle.readline()",
                "                qual = handle.readline().strip()",
                "",
                "                if not seq and not qual:",
                "                    continue",
                "",
                "                self._count += 1",
                "                slen = len(seq)",
                "                total_len += slen",
                "                total_gc += sum(1 for ch in seq.upper() if ch in ('G', 'C'))",
                "                if slen > max_len:",
                "                    max_len = slen",
                "                if min_len is None or slen < min_len:",
                "                    min_len = slen",
                "",
                "                if qual:",
                "                    qmax = max((ord(ch) - 33) for ch in qual)",
                "                    qmin = min((ord(ch) - 33) for ch in qual)",
                "                    if max_qual is None or qmax > max_qual:",
                "                        max_qual = qmax",
                "                    if min_qual is None or qmin < min_qual:",
                "                        min_qual = qmin",
                "",
                "        self.gc_content = (float(total_gc) / float(total_len) * 100.0) if total_len else 0.0",
                "        self.avglen = (float(total_len) / float(self._count)) if self._count else 0.0",
                "        self.maxlen = max_len if self._count else 0",
                "        self.minlen = min_len if min_len is not None else 0",
                "        self.maxqual = max_qual if max_qual is not None else 0",
                "        self.minqual = min_qual if min_qual is not None else 0",
                "",
                "    def __len__(self):",
                "        return self._count",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_dvgfinder_subprocess_env(
    output_dir: Path,
    conda_executable: str,
    conda_env_name: str,
) -> dict[str, str]:
    env = os.environ.copy()

    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_config_dir.resolve())

    shim_needed = False
    shim_dir = output_dir / "_dvgfinder_shims"

    if not _conda_env_has_module(conda_executable, conda_env_name, "datapane"):
        shim_dir = output_dir / "_dvgfinder_shims"
        _write_datapane_shim(shim_dir)
        shim_needed = True

    if not _conda_env_has_module(conda_executable, conda_env_name, "pyfastx"):
        _write_pyfastx_shim(shim_dir)
        shim_needed = True

    if shim_needed:
        existing = env.get("PYTHONPATH", "")
        shim_path = str(shim_dir.resolve())
        env["PYTHONPATH"] = shim_path if not existing else f"{shim_path}{os.pathsep}{existing}"

    return env


def _expected_bwa_index_files(reference_fasta: Path) -> list[Path]:
    return [Path(f"{reference_fasta}{suffix}") for suffix in _BWA_INDEX_SUFFIXES]


def _expected_bowtie_index_files(reference_prefix: Path) -> list[Path]:
    return [Path(f"{reference_prefix}{suffix}") for suffix in _BOWTIE_INDEX_SUFFIXES]


def _stage_reference_for_dvgfinder(
    reference_fasta: Path,
    dvgfinder_dir: Path,
    conda_executable: str,
    conda_env_name: str,
) -> Path:
    """Copy reference to DVGfinder references dir and ensure bwa+bowtie indexes."""

    ref_dir = dvgfinder_dir / "ExternalNeeds" / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    staged_fasta = ref_dir / reference_fasta.name
    if reference_fasta.resolve() != staged_fasta.resolve():
        shutil.copy2(reference_fasta, staged_fasta)

    staged_prefix = staged_fasta.with_suffix("")

    missing_bwa = [path for path in _expected_bwa_index_files(staged_fasta) if not path.exists()]
    if missing_bwa:
        _run_command(
            [
                conda_executable,
                "run",
                "-n",
                conda_env_name,
                "bwa",
                "index",
                str(staged_fasta.resolve()),
            ],
            cwd=dvgfinder_dir,
        )

    missing_bowtie = [path for path in _expected_bowtie_index_files(staged_prefix) if not path.exists()]
    if missing_bowtie:
        _run_command(
            [
                conda_executable,
                "run",
                "-n",
                conda_env_name,
                "bowtie-build",
                str(staged_fasta.resolve()),
                str(staged_prefix.resolve()),
            ],
            cwd=dvgfinder_dir,
        )

    missing_bwa_after = [path for path in _expected_bwa_index_files(staged_fasta) if not path.exists()]
    if missing_bwa_after:
        raise FileNotFoundError(
            "BWA index files are missing for staged DVGfinder reference. "
            f"Missing: {[str(path) for path in missing_bwa_after]}"
        )

    missing_bowtie_after = [path for path in _expected_bowtie_index_files(staged_prefix) if not path.exists()]
    if missing_bowtie_after:
        raise FileNotFoundError(
            "Bowtie index files are missing for staged DVGfinder reference. "
            f"Missing: {[str(path) for path in missing_bowtie_after]}"
        )

    return staged_fasta


def _candidate_dvgfinder_outputs(dvgfinder_dir: Path, sample_name: str) -> list[Path]:
    report_dir = dvgfinder_dir / "FinalReports" / sample_name
    return [
        report_dir / f"{sample_name}_ALL.csv",
        report_dir / f"{sample_name}_virema.csv",
        report_dir / f"{sample_name}_ditector.csv",
    ]


def _candidate_dvgfinder_partial_outputs(dvgfinder_dir: Path, sample_name: str) -> list[Path]:
    outputs_dir = dvgfinder_dir / "Outputs"
    candidates = [
        outputs_dir / f"{sample_name}_unificated_table.csv",
        outputs_dir / f"{sample_name}_from_raw_ditector.tsv",
        outputs_dir / f"{sample_name}_from_raw_virema.tsv",
    ]

    archive_root = dvgfinder_dir / "OldOutputs" / "wrapper_preclean"
    if archive_root.exists():
        archived_runs = sorted(
            archive_root.glob(f"{sample_name}_*"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for run_dir in archived_runs:
            archived_outputs = run_dir / "Outputs"
            candidates.extend(
                [
                    archived_outputs / f"{sample_name}_unificated_table.csv",
                    archived_outputs / f"{sample_name}_from_raw_ditector.tsv",
                    archived_outputs / f"{sample_name}_from_raw_virema.tsv",
                ]
            )

    return candidates


def _select_latest_nonempty(candidates: list[Path]) -> Path | None:
    existing = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    if not existing:
        return None
    existing.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return existing[0]


def run_dvgfinder(
    reference_fasta: Path,
    reads_r1: Path,
    reads_r2: Path,
    output_dir: Path,
    output_tag: str = "pb2_sim",
    dvgfinder_dir: Path = Path("src/DVGfinder"),
    conda_executable: str = "conda",
    conda_env_name: str = "dvgfinder_env",
    auto_create_env: bool = False,
    margin: int = 5,
    ml_threshold: float = 0.5,
    threads: int = 4,
    polarity: int = 1,
    keep_merged_fastq: bool = False,
    additional_args: list[str] | None = None,
) -> Path:
    """Run DVGfinder and return path to the finalized DVGfinder table."""

    for path in (reference_fasta, reads_r1, reads_r2):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    if polarity not in (0, 1):
        raise ValueError("polarity must be 0 (negative) or 1 (positive).")
    if threads < 1:
        raise ValueError("threads must be >= 1.")
    if margin < 0:
        raise ValueError("margin must be >= 0.")

    if not dvgfinder_dir.exists():
        raise FileNotFoundError(f"DVGfinder directory not found: {dvgfinder_dir}")

    env_yaml = dvgfinder_dir / "dvgfinder_env.yaml"
    if not env_yaml.exists():
        raise FileNotFoundError(f"DVGfinder environment file not found: {env_yaml}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = output_tag
    merged_fastq = output_dir / f"{sample_name}.fastq"

    _ensure_conda_env(
        conda_executable=conda_executable,
        env_name=conda_env_name,
        env_yaml=env_yaml,
        auto_create_env=auto_create_env,
    )
    _archive_stale_sample_artifacts(dvgfinder_dir, sample_name)
    _ensure_dvgfinder_runtime_dirs(dvgfinder_dir)
    _ensure_dvgfinder_scripts_executable(dvgfinder_dir)
    staged_reference = _stage_reference_for_dvgfinder(
        reference_fasta=reference_fasta,
        dvgfinder_dir=dvgfinder_dir,
        conda_executable=conda_executable,
        conda_env_name=conda_env_name,
    )

    _merge_fastq_files(reads_r1, reads_r2, merged_fastq)

    final_report_candidates = _candidate_dvgfinder_outputs(dvgfinder_dir, sample_name)
    partial_output_candidates = _candidate_dvgfinder_partial_outputs(dvgfinder_dir, sample_name)
    before_mtime = {
        path: path.stat().st_mtime_ns if path.exists() else None
        for path in final_report_candidates + partial_output_candidates
    }

    cmd = [
        conda_executable,
        "run",
        "-n",
        conda_env_name,
        "python",
        "DVGfinder_v3.1.py",
        "-fq",
        str(merged_fastq.resolve()),
        "-r",
        str(staged_reference.resolve()),
        "-m",
        str(margin),
        "-t",
        str(ml_threshold),
        "-n",
        str(threads),
        "-s",
        str(polarity),
    ]

    if additional_args:
        cmd.extend(additional_args)

    dvgfinder_run_env = _build_dvgfinder_subprocess_env(
        output_dir=output_dir,
        conda_executable=conda_executable,
        conda_env_name=conda_env_name,
    )

    try:
        _run_command(cmd, cwd=dvgfinder_dir, env=dvgfinder_run_env)
    except RuntimeError as exc:
        # DVGfinder can fail late in ML/reporting while raw event files already exist.
        fallback_raw = _select_latest_nonempty(partial_output_candidates)
        if fallback_raw is not None:
            print(
                "Warning: DVGfinder exited with error; using partial raw events output: "
                f"{fallback_raw}",
                file=sys.stderr,
            )
            print(f"Underlying error: {exc}", file=sys.stderr)
            return fallback_raw
        raise
    finally:
        if not keep_merged_fastq and merged_fastq.exists():
            merged_fastq.unlink()

    updated_candidates = [
        path
        for path in final_report_candidates
        if path.exists() and (before_mtime[path] is None or path.stat().st_mtime_ns != before_mtime[path])
    ]
    for candidate in updated_candidates:
        return candidate

    for candidate in final_report_candidates:
        if candidate.exists():
            return candidate

    fallback_raw = _select_latest_nonempty(partial_output_candidates)
    if fallback_raw is not None:
        return fallback_raw

    raise FileNotFoundError(
        "DVGfinder completed but no supported output table was found. "
        f"Checked: {[str(path) for path in final_report_candidates + partial_output_candidates]}"
    )


def _normalize_header(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")


def _to_int(value: str) -> int:
    return int(float(value))


def _is_deletion_type(raw_type: str) -> bool:
    normalized = _normalize_header(raw_type)
    return "deletion" in normalized


def _extract_genome_id(value: str, fallback: str) -> str:
    text = value.strip()
    if not text:
        return fallback
    if "|" in text:
        return text.split("|", 1)[0].strip() or fallback
    return text


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
        "dvg_type",
        "dvg_s_type",
        "bp",
        "bp_pos",
        "ri",
        "ri_pos",
        "predicted_start",
        "predicted_end",
        "genome_id",
        "reference",
    }
    return bool(normalized.intersection(known_tokens))


def _parse_with_header(rows: list[list[str]], default_genome_id: str) -> list[PredictedDelVG]:
    headers = [_normalize_header(col) for col in rows[0]]
    data_rows = rows[1:]
    header_to_index = {name: idx for idx, name in enumerate(headers)}

    type_col = next((c for c in ("dvg_type", "dvg_s_type", "type") if c in header_to_index), None)
    genome_col = next((c for c in ("genome_id", "reference", "ref", "chrom", "rname_f") if c in header_to_index), None)
    left_col = next((c for c in ("bp", "bp_pos", "predicted_start", "start", "deletion_start") if c in header_to_index), None)
    right_col = next((c for c in ("ri", "ri_pos", "predicted_end", "end", "deletion_end") if c in header_to_index), None)

    if left_col is None or right_col is None:
        raise ValueError(
            "Unsupported DVGfinder table format. Need columns describing BP and RI coordinates."
        )

    predictions: list[PredictedDelVG] = []
    for row in data_rows:
        if len(row) < len(headers):
            continue

        if type_col is not None and not _is_deletion_type(row[header_to_index[type_col]]):
            continue

        try:
            predicted_start = _to_int(row[header_to_index[left_col]])
            predicted_end = _to_int(row[header_to_index[right_col]])
        except ValueError:
            continue

        if genome_col is None:
            genome_id = default_genome_id
        else:
            genome_id = _extract_genome_id(row[header_to_index[genome_col]], default_genome_id)

        predictions.append(PredictedDelVG(genome_id, predicted_start, predicted_end))

    return predictions


def _parse_without_header(rows: list[list[str]], default_genome_id: str) -> list[PredictedDelVG]:
    predictions: list[PredictedDelVG] = []

    for row in rows:
        if not row:
            continue

        # Outputs/{sample}_from_raw_ditector.tsv: DVG_type, BP, RI, read_counts_ditector
        if len(row) >= 4 and _is_deletion_type(row[0]):
            try:
                predicted_start = _to_int(row[1])
                predicted_end = _to_int(row[2])
            except ValueError:
                continue
            predictions.append(PredictedDelVG(default_genome_id, predicted_start, predicted_end))
            continue

        # Outputs/{sample}_from_raw_virema.tsv: BP, RI, read_counts_virema, sense
        if len(row) >= 4:
            try:
                predicted_start = _to_int(row[0])
                predicted_end = _to_int(row[1])
            except ValueError:
                continue
            predictions.append(PredictedDelVG(default_genome_id, predicted_start, predicted_end))
            continue

        # Generic 3-column fallback: genome/start/end or start/end/count
        if len(row) == 3:
            try:
                start = _to_int(row[1])
                end = _to_int(row[2])
                genome = _extract_genome_id(row[0], default_genome_id)
                predictions.append(PredictedDelVG(genome, start, end))
                continue
            except ValueError:
                pass

            try:
                start = _to_int(row[0])
                end = _to_int(row[1])
            except ValueError:
                continue
            predictions.append(PredictedDelVG(default_genome_id, start, end))

    return predictions


def parse_raw_events(raw_events_path: Path, default_genome_id: str) -> list[PredictedDelVG]:
    """Parse DVGfinder output into normalized deletion prediction objects."""

    if not raw_events_path.exists():
        raise FileNotFoundError(f"Raw events file not found: {raw_events_path}")

    rows = _read_noncomment_rows(raw_events_path)
    if not rows:
        return []

    if _looks_like_header(rows[0]):
        return _parse_with_header(rows, default_genome_id=default_genome_id)
    return _parse_without_header(rows, default_genome_id=default_genome_id)


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
    default_genome_id: str,
    genome_filter: str | None = None,
    min_deletion_size: int = 1,
) -> list[PredictedDelVG]:
    """Convert DVGfinder table output to standardized predicted_delvgs.csv."""

    if min_deletion_size < 1:
        raise ValueError("min_deletion_size must be >= 1.")

    events = parse_raw_events(raw_events_path, default_genome_id=default_genome_id)

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
    parser = argparse.ArgumentParser(description="Run DVGfinder and standardize predicted DelVG breakpoints.")
    script_dir = Path(__file__).resolve().parent

    parser.add_argument("--reference-fasta", type=Path, default=Path("data/PB2.fasta"))
    parser.add_argument("--reads-r1", type=Path, default=Path("output/reads_R1.fastq"))
    parser.add_argument("--reads-r2", type=Path, default=Path("output/reads_R2.fastq"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/dvgfinder"))
    parser.add_argument("--output-csv", type=Path, default=Path("output/predicted_delvgs.csv"))
    parser.add_argument(
        "--raw-events",
        type=Path,
        default=None,
        help="Existing DVGfinder output table to parse. If provided, DVGfinder is not executed.",
    )

    parser.add_argument("--output-tag", type=str, default="pb2_sim")
    parser.add_argument("--dvgfinder-dir", type=Path, default=script_dir / "DVGfinder")
    parser.add_argument("--conda-executable", type=str, default="conda")
    parser.add_argument("--conda-env-name", type=str, default="dvgfinder_env")
    parser.add_argument(
        "--auto-create-env",
        action="store_true",
        help="Automatically create missing conda env from src/DVGfinder/dvgfinder_env.yaml.",
    )

    parser.add_argument("--margin", type=int, default=5)
    parser.add_argument("--ml-threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--polarity", type=int, default=1)

    parser.add_argument("--genome-id", type=str, default=None)
    parser.add_argument("--genome-filter", type=str, default=None)
    parser.add_argument("--min-deletion-size", type=int, default=1)
    parser.add_argument(
        "--keep-merged-fastq",
        action="store_true",
        help="Keep intermediate merged FASTQ used for DVGfinder input.",
    )
    parser.add_argument(
        "--additional-dvgfinder-arg",
        action="append",
        default=[],
        help="Additional argument to append to DVGfinder command (can be repeated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_genome_id = args.genome_id
    if default_genome_id is None:
        default_genome_id = _read_first_fasta_id(args.reference_fasta)

    raw_events_path = args.raw_events
    if raw_events_path is None:
        try:
            raw_events_path = run_dvgfinder(
                reference_fasta=args.reference_fasta,
                reads_r1=args.reads_r1,
                reads_r2=args.reads_r2,
                output_dir=args.output_dir,
                output_tag=args.output_tag,
                dvgfinder_dir=args.dvgfinder_dir,
                conda_executable=args.conda_executable,
                conda_env_name=args.conda_env_name,
                auto_create_env=args.auto_create_env,
                margin=args.margin,
                ml_threshold=args.ml_threshold,
                threads=args.threads,
                polarity=args.polarity,
                keep_merged_fastq=args.keep_merged_fastq,
                additional_args=args.additional_dvgfinder_arg,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            fallback_candidates = _candidate_dvgfinder_partial_outputs(args.dvgfinder_dir, args.output_tag)
            fallback_raw = _select_latest_nonempty(fallback_candidates)
            if fallback_raw is None:
                raise
            print(
                "Warning: DVGfinder execution was unavailable; reusing existing raw events output: "
                f"{fallback_raw}",
                file=sys.stderr,
            )
            print(f"Underlying error: {exc}", file=sys.stderr)
            raw_events_path = fallback_raw

    standardize_predictions(
        raw_events_path=raw_events_path,
        output_csv=args.output_csv,
        default_genome_id=default_genome_id,
        genome_filter=args.genome_filter,
        min_deletion_size=args.min_deletion_size,
    )


if __name__ == "__main__":
    main()
