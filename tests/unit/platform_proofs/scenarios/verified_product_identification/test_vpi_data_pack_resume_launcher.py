"""Contract tests for VPI Data Pack one-click resume launcher (no GPU build)."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config import (
    VPI_DATA_PACK_ARTIFACT_ROOT,
    VPI_DATA_PACK_EXECUTION_PROFILE,
    VPI_DATA_PACK_RESUME_BUILD_SOURCE_SHA,
    VPI_DATA_PACK_SHARD_SIZE,
    build_vpi_data_pack_resume_cli_argv,
    resolve_vpi_data_pack_resume_launch_plan,
)
from platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume import (
    VpiDataPackResumePreflightError,
    _read_process_pids,
    assert_cuda_preflight,
    assert_no_active_canonical_writer,
    assert_path_preflight,
    run_vpi_data_pack_resume,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VPI_ROOT = _REPO_ROOT / "platform_proofs" / "scenarios" / "verified_product_identification"
_BAT_PATH = _VPI_ROOT / "scripts" / "operator" / "resume_vpi_data_pack.bat"
_COMPAT_BAT_PATH = _REPO_ROOT / "scripts" / "vpi" / "resume_vpi_data_pack.bat"
_OLD_BAT_PATH = _REPO_ROOT / "scripts" / "vpi" / "resume_vpi_5c4f.bat"


def test_canonical_launcher_exists_old_launcher_removed() -> None:
    assert _BAT_PATH.is_file()
    assert _COMPAT_BAT_PATH.is_file()
    assert not _OLD_BAT_PATH.exists()


def test_bat_contract_resume_only_production_parameters() -> None:
    text = _BAT_PATH.read_text(encoding="utf-8")
    assert "scripts.operator.run_vpi_data_pack_resume" in text
    assert "5C4F" not in text.upper()
    assert "5c4f" not in text
    assert "--start-fresh" not in text
    assert "%1" not in text
    plan = resolve_vpi_data_pack_resume_launch_plan()
    argv = build_vpi_data_pack_resume_cli_argv(plan)
    joined = " ".join(argv)
    assert "--resume" in joined
    assert "--start-fresh" not in joined
    assert f"--shard-size {VPI_DATA_PACK_SHARD_SIZE}" in joined
    assert f"--execution-profile {VPI_DATA_PACK_EXECUTION_PROFILE}" in joined
    assert str(VPI_DATA_PACK_ARTIFACT_ROOT) in joined


def test_build_source_sha_pinned_not_placeholder() -> None:
    assert VPI_DATA_PACK_RESUME_BUILD_SOURCE_SHA != "PENDING_R1_COMMIT"
    assert len(VPI_DATA_PACK_RESUME_BUILD_SOURCE_SHA) == 40


def test_path_preflight_missing_build_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "pack"
    output_root.mkdir()
    (output_root / "state").mkdir()
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config.VPI_DATA_PACK_RESUME_BUILD_SOURCE_SHA",
        "test-sha",
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config.VPI_DATA_PACK_ARTIFACT_ROOT",
        output_root,
    )
    build_source = tmp_path / "build-source" / "test-sha"
    build_source.mkdir(parents=True)
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config.VPI_DATA_PACK_BUILD_SOURCE_ROOT",
        tmp_path / "build-source",
    )
    dataset = tmp_path / "dataset.parquet"
    manifest = tmp_path / "manifest.json"
    dataset.write_bytes(b"x")
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config.VPI_DATA_PACK_DATASET_PATH",
        dataset,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config.VPI_DATA_PACK_DATASET_MANIFEST_PATH",
        manifest,
    )
    python_stub = tmp_path / "python.exe"
    python_stub.write_bytes(b"")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.vpi_data_pack_resume_config.VPI_DATA_PACK_CUDA_PYTHON",
        python_stub,
    )
    plan = resolve_vpi_data_pack_resume_launch_plan()
    with pytest.raises(VpiDataPackResumePreflightError, match="build-state"):
        assert_path_preflight(plan)


def test_cuda_python_missing_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing-python.exe"
    with pytest.raises(VpiDataPackResumePreflightError, match="CUDA python missing"):
        assert_cuda_preflight(missing)


_PROCESS_SAMPLE = '{\n  "python_pid": 123,\n  "powershell_pid": 456\n}\n'


def test_read_process_pids_utf8_without_bom(tmp_path: Path) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text(_PROCESS_SAMPLE, encoding="utf-8")
    assert _read_process_pids(process_json) == (123, 456)


def test_read_process_pids_utf8_with_bom(tmp_path: Path) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text(_PROCESS_SAMPLE, encoding="utf-8-sig")
    assert _read_process_pids(process_json) == (123, 456)


def test_read_process_pids_invalid_json_raises_preflight(tmp_path: Path) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(VpiDataPackResumePreflightError, match="invalid process evidence"):
        _read_process_pids(process_json)


def test_read_process_pids_missing_file_returns_none_pair(tmp_path: Path) -> None:
    assert _read_process_pids(tmp_path / "process.json") == (None, None)


def test_read_process_pids_null_or_missing_pids(tmp_path: Path) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text("{}", encoding="utf-8")
    assert _read_process_pids(process_json) == (None, None)
    process_json.write_text('{"python_pid": null, "powershell_pid": null}', encoding="utf-8")
    assert _read_process_pids(process_json) == (None, None)
    process_json.write_text('{"python_pid": "123"}', encoding="utf-8")
    assert _read_process_pids(process_json) == (None, None)


def test_active_writer_guard_fails_when_pid_alive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text('{"python_pid": 99999}', encoding="utf-8")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume._windows_pid_alive",
        lambda pid: pid == 99999,
    )
    with pytest.raises(VpiDataPackResumePreflightError, match="ALREADY RUNNING"):
        assert_no_active_canonical_writer(process_json)


def test_active_writer_guard_with_bom_process_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text('{"python_pid": 4242}', encoding="utf-8-sig")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume._windows_pid_alive",
        lambda pid: pid == 4242,
    )
    with pytest.raises(VpiDataPackResumePreflightError, match="ALREADY RUNNING"):
        assert_no_active_canonical_writer(process_json)


def test_run_resume_dry_run_does_not_start_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume.assert_no_active_canonical_writer",
        lambda: None,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume.assert_cuda_preflight",
        lambda _path: "test-gpu",
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume.assert_path_preflight",
        lambda _plan: None,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume._read_build_progress_counts",
        lambda _root: (1200, 3771, 1201, 1200),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.scripts.operator.run_vpi_data_pack_resume.resolve_vpi_data_pack_resume_launch_plan",
        resolve_vpi_data_pack_resume_launch_plan,
    )
    exit_code = run_vpi_data_pack_resume(dry_run=True)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "VPI DATA PACK — RESUME" in out
    assert "resume shard: 1201" in out
    assert "boundary validation: shard 1200 only" in out
    assert "5C4F" not in out.upper()
