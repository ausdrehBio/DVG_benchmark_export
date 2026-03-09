"""Tests for DVGfinder wrapper + normalization logic."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_DVGfinder  # noqa: E402
from run_DVGfinder import parse_raw_events, run_dvgfinder, standardize_predictions  # noqa: E402


def _touch_bowtie_index_files(reference_prefix: Path) -> None:
    for suffix in (".1.ebwt", ".2.ebwt", ".3.ebwt", ".4.ebwt", ".rev.1.ebwt", ".rev.2.ebwt"):
        Path(f"{reference_prefix}{suffix}").write_text("idx\n", encoding="utf-8")


def _touch_bwa_index_files(reference_fasta: Path) -> None:
    for suffix in (".amb", ".ann", ".bwt", ".pac", ".sa"):
        Path(f"{reference_fasta}{suffix}").write_text("idx\n", encoding="utf-8")


def _create_dvgfinder_helper_scripts(dvgfinder_dir: Path) -> None:
    models_dir = dvgfinder_dir / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for script_name in (
        "extract_H_reads.sh",
        "extract_recombination_events_virema.sh",
        "extract_recombination_events_ditector.sh",
    ):
        (models_dir / script_name).write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")


def test_standardize_predictions_from_dvgfinder_all_table(tmp_path: Path) -> None:
    raw_events = tmp_path / "pb2_sim_ALL.csv"
    output_csv = tmp_path / "predicted_delvgs.csv"

    raw_events.write_text(
        "\n".join(
            [
                "cID_DI,BP,RI,sense,DVG_type,length_dvg,read_counts_virema,pBP_virema,pRI_virema,rpht_virema,read_counts_ditector,pBP_ditector,pRI_ditector,rpht_ditector",
                "++_500_900,500,900,++,Deletion_forward,400,10,0.1,0.1,20,9,0.1,0.1,18",
                "++_700_705,700,705,++,Insertion_forward,6,3,0.02,0.02,4,0,0,0,0",
                "--_1700_1300,1700,1300,--,Deletion_reverse,400,8,0.05,0.07,16,7,0.04,0.06,14",
                "--_1700_1300,1700,1300,--,Deletion_reverse,400,6,0.04,0.06,12,6,0.04,0.06,12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events, default_genome_id="PB2")
    assert len(parsed) == 3  # insertion filtered out

    standardized = standardize_predictions(
        raw_events_path=raw_events,
        output_csv=output_csv,
        default_genome_id="PB2",
        genome_filter="PB2",
        min_deletion_size=1,
    )

    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8")))
    assert [(event.genome_id, event.predicted_start, event.predicted_end) for event in standardized] == [
        ("PB2", 500, 900),
        ("PB2", 1300, 1700),
    ]
    assert rows == [
        {"genome_id": "PB2", "predicted_start": "500", "predicted_end": "900"},
        {"genome_id": "PB2", "predicted_start": "1300", "predicted_end": "1700"},
    ]


def test_parse_raw_events_from_ditector_raw_tsv_without_header(tmp_path: Path) -> None:
    raw_events = tmp_path / "pb2_sim_from_raw_ditector.tsv"
    raw_events.write_text(
        "\n".join(
            [
                "Deletion_forward\t500\t900\t11",
                "Insertion_forward\t700\t705\t3",
                "Deletion_reverse\t1700\t1300\t9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_raw_events(raw_events, default_genome_id="PB2")
    assert [(item.genome_id, item.predicted_start, item.predicted_end) for item in parsed] == [
        ("PB2", 500, 900),
        ("PB2", 1700, 1300),
    ]


def test_run_dvgfinder_builds_expected_command_and_discovers_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "dvgfinder_wrapper"
    dvgfinder_dir = tmp_path / "DVGfinder"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")

    (dvgfinder_dir / "ExternalNeeds" / "references").mkdir(parents=True, exist_ok=True)
    _create_dvgfinder_helper_scripts(dvgfinder_dir)
    (dvgfinder_dir / "dvgfinder_env.yaml").write_text("name: dvgfinder_env\n", encoding="utf-8")

    captured_cmds: list[tuple[list[str], Path | None]] = []

    def fake_run_command(
        cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> None:
        captured_cmds.append((cmd, cwd))

        if len(cmd) >= 7 and cmd[4] == "bwa" and cmd[5] == "index":
            _touch_bwa_index_files(Path(cmd[6]))
            return

        if len(cmd) >= 7 and cmd[4] == "bowtie-build":
            _touch_bowtie_index_files(Path(cmd[6]))
            return

        if "DVGfinder_v3.1.py" in cmd:
            sample_report_dir = dvgfinder_dir / "FinalReports" / "pb2_sim"
            sample_report_dir.mkdir(parents=True, exist_ok=True)
            (sample_report_dir / "pb2_sim_ALL.csv").write_text(
                "cID_DI,BP,RI,sense,DVG_type\n++_10_30,10,30,++,Deletion_forward\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(run_DVGfinder, "_run_command", fake_run_command)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_is_compatible", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(run_DVGfinder, "_conda_env_has_module", lambda *_args, **_kwargs: True)

    raw_output = run_dvgfinder(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        dvgfinder_dir=dvgfinder_dir,
        conda_executable="conda",
        conda_env_name="dvgfinder_env",
        margin=6,
        ml_threshold=0.7,
        threads=8,
        polarity=1,
    )

    assert raw_output == dvgfinder_dir / "FinalReports" / "pb2_sim" / "pb2_sim_ALL.csv"
    assert not (output_dir / "pb2_sim.fastq").exists()

    dvgfinder_cmd = next((cmd for cmd, _cwd in captured_cmds if "DVGfinder_v3.1.py" in cmd), None)
    assert dvgfinder_cmd is not None
    assert dvgfinder_cmd[:6] == ["conda", "run", "-n", "dvgfinder_env", "python", "DVGfinder_v3.1.py"]
    script_start = dvgfinder_cmd.index("DVGfinder_v3.1.py") + 1
    script_args = dvgfinder_cmd[script_start:]
    assert "-fq" in script_args
    assert str((output_dir / "pb2_sim.fastq").resolve()) in script_args
    assert "-r" in script_args
    assert str((dvgfinder_dir / "ExternalNeeds" / "references" / "PB2.fasta").resolve()) in script_args
    assert ["-m", "6"] == script_args[script_args.index("-m") : script_args.index("-m") + 2]
    assert ["-t", "0.7"] == script_args[script_args.index("-t") : script_args.index("-t") + 2]
    assert ["-n", "8"] == script_args[script_args.index("-n") : script_args.index("-n") + 2]
    assert ["-s", "1"] == script_args[script_args.index("-s") : script_args.index("-s") + 2]


def test_run_dvgfinder_archives_stale_sample_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "dvgfinder_wrapper"
    dvgfinder_dir = tmp_path / "DVGfinder"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")

    (dvgfinder_dir / "ExternalNeeds" / "references").mkdir(parents=True, exist_ok=True)
    _create_dvgfinder_helper_scripts(dvgfinder_dir)
    (dvgfinder_dir / "dvgfinder_env.yaml").write_text("name: dvgfinder_env\n", encoding="utf-8")

    stale_virema_dir = dvgfinder_dir / "Outputs" / "virema" / "pb2_sim"
    stale_virema_dir.mkdir(parents=True, exist_ok=True)
    (stale_virema_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
    stale_raw_virema = dvgfinder_dir / "Outputs" / "pb2_sim_from_raw_virema.tsv"
    stale_raw_virema.write_text("1\t2\t3\t++\n", encoding="utf-8")

    def fake_run_command(
        cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> None:
        if len(cmd) >= 7 and cmd[4] == "bwa" and cmd[5] == "index":
            _touch_bwa_index_files(Path(cmd[6]))
            return

        if len(cmd) >= 7 and cmd[4] == "bowtie-build":
            _touch_bowtie_index_files(Path(cmd[6]))
            return

        if "DVGfinder_v3.1.py" in cmd:
            sample_report_dir = dvgfinder_dir / "FinalReports" / "pb2_sim"
            sample_report_dir.mkdir(parents=True, exist_ok=True)
            (sample_report_dir / "pb2_sim_ALL.csv").write_text(
                "cID_DI,BP,RI,sense,DVG_type\n++_10_30,10,30,++,Deletion_forward\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(run_DVGfinder, "_run_command", fake_run_command)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_is_compatible", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(run_DVGfinder, "_conda_env_has_module", lambda *_args, **_kwargs: True)

    run_dvgfinder(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        dvgfinder_dir=dvgfinder_dir,
        conda_executable="conda",
        conda_env_name="dvgfinder_env",
    )

    assert not stale_virema_dir.exists()
    assert not stale_raw_virema.exists()

    archive_roots = list((dvgfinder_dir / "OldOutputs" / "wrapper_preclean").glob("pb2_sim_*"))
    assert len(archive_roots) == 1
    assert (archive_roots[0] / "Outputs" / "virema" / "pb2_sim" / "stale.txt").exists()
    assert (archive_roots[0] / "Outputs" / "pb2_sim_from_raw_virema.tsv").exists()


def test_run_dvgfinder_uses_datapane_shim_when_module_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "dvgfinder_wrapper"
    dvgfinder_dir = tmp_path / "DVGfinder"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")

    (dvgfinder_dir / "ExternalNeeds" / "references").mkdir(parents=True, exist_ok=True)
    _create_dvgfinder_helper_scripts(dvgfinder_dir)
    (dvgfinder_dir / "dvgfinder_env.yaml").write_text("name: dvgfinder_env\n", encoding="utf-8")

    captured_run_env: dict[str, str] | None = None

    def fake_run_command(
        cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> None:
        nonlocal captured_run_env

        if len(cmd) >= 7 and cmd[4] == "bwa" and cmd[5] == "index":
            _touch_bwa_index_files(Path(cmd[6]))
            return

        if len(cmd) >= 7 and cmd[4] == "bowtie-build":
            _touch_bowtie_index_files(Path(cmd[6]))
            return

        if "DVGfinder_v3.1.py" in cmd:
            captured_run_env = env
            sample_report_dir = dvgfinder_dir / "FinalReports" / "pb2_sim"
            sample_report_dir.mkdir(parents=True, exist_ok=True)
            (sample_report_dir / "pb2_sim_ALL.csv").write_text(
                "cID_DI,BP,RI,sense,DVG_type\n++_10_30,10,30,++,Deletion_forward\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(run_DVGfinder, "_run_command", fake_run_command)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_is_compatible", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(run_DVGfinder, "_conda_env_has_module", lambda *_args, **_kwargs: False)

    run_dvgfinder(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        dvgfinder_dir=dvgfinder_dir,
        conda_executable="conda",
        conda_env_name="dvgfinder_env",
    )

    shim_datapane = output_dir / "_dvgfinder_shims" / "datapane.py"
    shim_pyfastx = output_dir / "_dvgfinder_shims" / "pyfastx.py"
    assert shim_datapane.exists()
    assert shim_pyfastx.exists()
    assert captured_run_env is not None
    assert "PYTHONPATH" in captured_run_env
    assert str((output_dir / "_dvgfinder_shims").resolve()) in captured_run_env["PYTHONPATH"]
    assert captured_run_env["MPLCONFIGDIR"] == str((output_dir / ".mplconfig").resolve())


def test_run_dvgfinder_returns_partial_output_on_late_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "dvgfinder_wrapper"
    dvgfinder_dir = tmp_path / "DVGfinder"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")

    (dvgfinder_dir / "ExternalNeeds" / "references").mkdir(parents=True, exist_ok=True)
    _create_dvgfinder_helper_scripts(dvgfinder_dir)
    (dvgfinder_dir / "dvgfinder_env.yaml").write_text("name: dvgfinder_env\n", encoding="utf-8")

    partial_output = dvgfinder_dir / "Outputs" / "pb2_sim_from_raw_ditector.tsv"

    def fake_run_command(
        cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> None:
        if len(cmd) >= 7 and cmd[4] == "bwa" and cmd[5] == "index":
            _touch_bwa_index_files(Path(cmd[6]))
            return

        if len(cmd) >= 7 and cmd[4] == "bowtie-build":
            _touch_bowtie_index_files(Path(cmd[6]))
            return

        if "DVGfinder_v3.1.py" in cmd:
            partial_output.parent.mkdir(parents=True, exist_ok=True)
            partial_output.write_text("Deletion_forward\t10\t30\t2\n", encoding="utf-8")
            raise RuntimeError("DVGfinder failed after writing raw outputs")

    monkeypatch.setattr(run_DVGfinder, "_run_command", fake_run_command)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(run_DVGfinder, "_conda_env_is_compatible", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(run_DVGfinder, "_conda_env_has_module", lambda *_args, **_kwargs: True)

    raw_output = run_dvgfinder(
        reference_fasta=reference_fasta,
        reads_r1=reads_r1,
        reads_r2=reads_r2,
        output_dir=output_dir,
        output_tag="pb2_sim",
        dvgfinder_dir=dvgfinder_dir,
        conda_executable="conda",
        conda_env_name="dvgfinder_env",
    )

    assert raw_output == partial_output


def test_run_dvgfinder_missing_conda_env_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference_fasta = tmp_path / "PB2.fasta"
    reads_r1 = tmp_path / "reads_R1.fastq"
    reads_r2 = tmp_path / "reads_R2.fastq"
    output_dir = tmp_path / "dvgfinder_wrapper"
    dvgfinder_dir = tmp_path / "DVGfinder"

    reference_fasta.write_text(">PB2\nACGTACGTACGT\n", encoding="utf-8")
    reads_r1.write_text("@r1\nACGT\n+\nFFFF\n", encoding="utf-8")
    reads_r2.write_text("@r2\nTGCA\n+\nFFFF\n", encoding="utf-8")
    (dvgfinder_dir / "ExternalNeeds" / "references").mkdir(parents=True, exist_ok=True)
    (dvgfinder_dir / "dvgfinder_env.yaml").write_text("name: dvgfinder_env\n", encoding="utf-8")

    monkeypatch.setattr(run_DVGfinder, "_conda_env_exists", lambda *_args, **_kwargs: False)

    with pytest.raises(FileNotFoundError, match="Conda environment 'dvgfinder_env' was not found"):
        run_dvgfinder(
            reference_fasta=reference_fasta,
            reads_r1=reads_r1,
            reads_r2=reads_r2,
            output_dir=output_dir,
            output_tag="pb2_sim",
            dvgfinder_dir=dvgfinder_dir,
            conda_executable="conda",
            conda_env_name="dvgfinder_env",
            auto_create_env=False,
        )
