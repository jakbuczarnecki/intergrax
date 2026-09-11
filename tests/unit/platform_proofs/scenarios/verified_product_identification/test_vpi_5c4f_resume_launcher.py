"""Contract tests for VPI 5C4F one-click resume launcher (no GPU build)."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config import (
    VPI_5C4F_ARTIFACT_ROOT,
    VPI_5C4F_EXECUTION_PROFILE,
    VPI_5C4F_SHARD_SIZE,
    build_resume_cli_argv,
    resolve_vpi_5c4f_resume_launch_plan,
)
from platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher import (
    Vpi5C4FResumePreflightError,
    assert_cuda_preflight,
    assert_no_active_canonical_writer,
    assert_path_preflight,
    run_vpi_5c4f_resume,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BAT_PATH = _REPO_ROOT / "scripts" / "vpi" / "resume_vpi_5c4f.bat"


def test_bat_contract_resume_only_production_parameters() -> None:
    text = _BAT_PATH.read_text(encoding="utf-8")
    assert "vpi_5c4f_resume_launcher" in text
    assert "--start-fresh" not in text
    plan = resolve_vpi_5c4f_resume_launch_plan()
    argv = build_resume_cli_argv(plan)
    joined = " ".join(argv)
    assert "--resume" in joined
    assert "--start-fresh" not in joined
    assert f"--shard-size {VPI_5C4F_SHARD_SIZE}" in joined
    assert f"--execution-profile {VPI_5C4F_EXECUTION_PROFILE}" in joined
    assert str(VPI_5C4F_ARTIFACT_ROOT) in joined


def test_path_preflight_missing_build_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = resolve_vpi_5c4f_resume_launch_plan()
    output_root = tmp_path / "pack"
    output_root.mkdir()
    (output_root / "state").mkdir()
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config.VPI_5C4F_RESUME_BUILD_SOURCE_SHA",
        "test-sha",
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config.VPI_5C4F_ARTIFACT_ROOT",
        output_root,
    )
    build_source = tmp_path / "build-source" / "test-sha"
    build_source.mkdir(parents=True)
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config.VPI_5C4F_BUILD_SOURCE_ROOT",
        tmp_path / "build-source",
    )
    dataset = tmp_path / "dataset.parquet"
    manifest = tmp_path / "manifest.json"
    dataset.write_bytes(b"x")
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config.VPI_5C4F_DATASET_PATH",
        dataset,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config.VPI_5C4F_DATASET_MANIFEST_PATH",
        manifest,
    )
    python_stub = tmp_path / "python.exe"
    python_stub.write_bytes(b"")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_config.VPI_5C4F_CUDA_PYTHON",
        python_stub,
    )
    plan = resolve_vpi_5c4f_resume_launch_plan()
    with pytest.raises(Vpi5C4FResumePreflightError, match="build-state"):
        assert_path_preflight(plan)


def test_cuda_python_missing_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing-python.exe"
    with pytest.raises(Vpi5C4FResumePreflightError, match="CUDA python missing"):
        assert_cuda_preflight(missing)


def test_active_writer_guard_fails_when_pid_alive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process_json = tmp_path / "process.json"
    process_json.write_text('{"python_pid": 99999}', encoding="utf-8")
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher._windows_pid_alive",
        lambda pid: pid == 99999,
    )
    with pytest.raises(Vpi5C4FResumePreflightError, match="ALREADY RUNNING"):
        assert_no_active_canonical_writer(process_json)


def test_run_resume_dry_run_does_not_start_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher.assert_no_active_canonical_writer",
        lambda: None,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher.assert_cuda_preflight",
        lambda _path: "test-gpu",
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher.assert_path_preflight",
        lambda _plan: None,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher._read_build_progress_counts",
        lambda _root: (1200, 3771, 1201, 1200),
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.operator.vpi_5c4f_resume_launcher.resolve_vpi_5c4f_resume_launch_plan",
        resolve_vpi_5c4f_resume_launch_plan,
    )
    exit_code = run_vpi_5c4f_resume(dry_run=True)
    assert exit_code == 0
