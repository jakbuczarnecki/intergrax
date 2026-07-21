# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ORCH = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-windows-native-certification.py"
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def orch() -> ModuleType:
    return _load(_ORCH, "lkw_windows_native_cert_orch")


def _pass_hosting() -> dict[str, Any]:
    return {
        "proof_kind": "platform_application_hosting",
        "certified_scope": "application_hosting_phase",
        "full_core_platform_proof": False,
        "proof_id": "local_workspace:platform_application_hosting:run-h",
        "run_id": "run-h",
        "correlation_id": "corr-h",
        "result": "PASS",
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
    }


def _pass_interaction() -> dict[str, Any]:
    return {
        "proof_kind": "platform_windows_interaction",
        "proof_id": "local_workspace:platform_windows_interaction:run-i",
        "run_id": "run-i",
        "correlation_id": "corr-i",
        "adapter_id": "lkw.windows_powershell",
        "source": "windows_powershell",
        "client_runtime": "python",
        "wrapper_runtime": "windows_powershell",
        "powershell_runtime": "Windows PowerShell",
        "result": "PASS",
        "receipt_recorded": True,
        "receipt_verified": True,
        "receipt_query_verified": True,
    }


def _hosting_kv(**overrides: str) -> dict[str, str]:
    values = {
        "core_proof_result": "PASS",
        "proof_kind": "platform_application_hosting",
        "proof_id": "hosting-id",
        "run_id": "hosting-run",
        "correlation_id": "hosting-corr",
        "proof_receipt_recorded": "true",
        "proof_receipt_verified": "true",
        "proof_receipt_query_verified": "true",
        "result": "PASS",
    }
    values.update(overrides)
    return values


def _interaction_kv(**overrides: str) -> dict[str, str]:
    values = {
        "proof_result": "PASS",
        "proof_kind": "platform_windows_interaction",
        "adapter_id": "lkw.windows_powershell",
        "source": "windows_powershell",
        "client_runtime": "python",
        "wrapper_runtime": "windows_powershell",
        "powershell_runtime": "Windows PowerShell",
        "os_family": "windows",
        "proof_receipt_id": "ix-id",
        "proof_receipt_run_id": "ix-run",
        "correlation_id": "ix-corr",
        "proof_receipt_recorded": "true",
        "proof_receipt_verified": "true",
        "proof_receipt_query_verified": "true",
    }
    values.update(overrides)
    return values


def test_accepts_native_windows_metadata(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch.platform, "system", lambda: "Windows")
    monkeypatch.setattr(orch.os, "name", "nt")
    monkeypatch.setattr(orch.platform, "version", lambda: "10.0.26200")
    monkeypatch.setattr(orch.platform, "release", lambda: "10")
    monkeypatch.setattr(orch.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(orch.platform, "python_version", lambda: "3.12.0")
    meta = orch.require_native_windows()
    assert meta["execution_os_family"] == "windows"


def test_rejects_non_windows_runtime(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch.platform, "system", lambda: "Linux")
    with pytest.raises(orch.CertificationOrchestratorError, match="non_windows_runtime"):
        orch.require_native_windows()


def test_rejects_missing_powershell(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch.shutil, "which", lambda _name: None)
    with pytest.raises(
        orch.CertificationOrchestratorError, match="powershell_unavailable"
    ):
        orch.require_powershell()


def test_rejects_missing_docker(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orch.shutil, "which", lambda _name: None)
    with pytest.raises(orch.CertificationOrchestratorError, match="docker_unavailable"):
        orch.require_docker()


def test_application_hosting_block_required(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="missing_application_hosting_proof"
    ):
        orch.validate_certification_blocks(
            application_hosting={},
            interaction=_pass_interaction(),
        )


def test_interaction_block_required(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="missing_interaction_proof"
    ):
        orch.validate_certification_blocks(
            application_hosting=_pass_hosting(),
            interaction={},
        )


def test_rejects_core_proof_as_substitute(orch: ModuleType) -> None:
    evidence = orch.build_evidence(
        runtime={
            "execution_os_version": "10",
            "execution_kernel_release": "10",
            "execution_architecture": "AMD64",
            "python_version": "3.12.0",
        },
        docker={
            "docker_engine_os": "linux",
            "docker_engine_architecture": "x86_64",
            "docker_engine_version": "27.0.0",
        },
        powershell_runtime="Windows PowerShell",
        application_hosting=_pass_hosting(),
        interaction=_pass_interaction(),
        source_commit="6b71a841c894728766fd6f574c9cd53ad12ec5f9",
        source_tree_dirty=True,
        source_tree_diff_sha256="abc",
    )
    assert "core_proof" not in evidence
    assert "application_hosting_proof" in evidence


def test_application_hosting_proof_kind_frozen(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="unexpected_.*proof_kind"
    ):
        orch.validate_application_hosting_kv(
            _hosting_kv(proof_kind="platform_core_proof")
        )


def test_application_hosting_scope_frozen(orch: ModuleType) -> None:
    block = orch.validate_application_hosting_kv(_hosting_kv())
    assert block["certified_scope"] == "application_hosting_phase"


def test_full_core_claim_must_be_false(orch: ModuleType) -> None:
    bad = _pass_hosting()
    bad["full_core_platform_proof"] = True
    with pytest.raises(
        orch.CertificationOrchestratorError,
        match="application_hosting_must_not_claim_full_core",
    ):
        orch.validate_certification_blocks(
            application_hosting=bad,
            interaction=_pass_interaction(),
        )


def test_windows_interaction_proof_kind_frozen(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="unexpected_proof_kind"
    ):
        orch.validate_interaction_kv(
            _interaction_kv(proof_kind="platform_linux_interaction")
        )


def test_adapter_id_frozen(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="unexpected_adapter_id"
    ):
        orch.validate_interaction_kv(_interaction_kv(adapter_id="lkw.linux_shell"))


def test_source_frozen(orch: ModuleType) -> None:
    with pytest.raises(orch.CertificationOrchestratorError, match="unexpected_source"):
        orch.validate_interaction_kv(_interaction_kv(source="linux_shell"))


def test_client_runtime_frozen(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="unexpected_client_runtime"
    ):
        orch.validate_interaction_kv(_interaction_kv(client_runtime="powershell"))


def test_wrapper_runtime_frozen(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="unexpected_wrapper_runtime"
    ):
        orch.validate_interaction_kv(_interaction_kv(wrapper_runtime="posix_sh"))


def test_powershell_runtime_frozen(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="unexpected_powershell_runtime"
    ):
        orch.validate_interaction_kv(_interaction_kv(powershell_runtime="pwsh"))


def test_all_receipt_flags_must_be_true(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="false_or_missing"
    ):
        orch.validate_interaction_kv(
            _interaction_kv(proof_receipt_verified="false")
        )


def test_blank_proof_id_fails(orch: ModuleType) -> None:
    with pytest.raises(orch.CertificationOrchestratorError, match="blank_proof_id"):
        orch.validate_application_hosting_kv(_hosting_kv(proof_id=""))


def test_blank_run_id_fails(orch: ModuleType) -> None:
    with pytest.raises(orch.CertificationOrchestratorError, match="blank_run_id"):
        orch.validate_application_hosting_kv(_hosting_kv(run_id=""))


def test_blank_correlation_id_fails(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="blank_correlation_id"
    ):
        orch.validate_application_hosting_kv(_hosting_kv(correlation_id=""))


def test_failed_proof_output_fails(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="interaction_proof_failed"
    ):
        orch.validate_interaction_kv(_interaction_kv(proof_result="FAIL"))


def test_skipped_proof_output_fails(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="interaction_proof_skipped"
    ):
        orch.validate_interaction_kv(_interaction_kv(proof_result="SKIP"))


def test_malformed_output_fails(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="malformed_interaction_output"
    ):
        orch.validate_interaction_kv({})


def test_secrets_are_removed(orch: ModuleType) -> None:
    cleaned = orch.scrub_secrets(
        {
            "password": "secret",
            "INTERGRAX_MONGODB_URI": "mongodb://u:p@host/db",
            "note": "ok",
            "nested": {"token": "abc", "keep": "yes"},
        }
    )
    assert "password" not in cleaned
    assert "INTERGRAX_MONGODB_URI" not in cleaned
    assert cleaned["note"] == "ok"
    assert "token" not in cleaned["nested"]
    assert cleaned["nested"]["keep"] == "yes"


def test_evidence_schema_deterministic(orch: ModuleType) -> None:
    evidence = orch.build_evidence(
        runtime={
            "execution_os_version": "10.0.26200",
            "execution_kernel_release": "10",
            "execution_architecture": "AMD64",
            "python_version": "3.12.8",
        },
        docker={
            "docker_engine_os": "linux",
            "docker_engine_architecture": "x86_64",
            "docker_engine_version": "27.5.1",
        },
        powershell_runtime="Windows PowerShell",
        application_hosting=_pass_hosting(),
        interaction=_pass_interaction(),
        source_commit="6b71a841c894728766fd6f574c9cd53ad12ec5f9",
        source_tree_dirty=True,
        source_tree_diff_sha256="deadbeef",
    )
    text = json.dumps(evidence, indent=2, sort_keys=True)
    assert evidence["certification_profile"] == "windows_native_runtime"
    assert evidence["certification_result"] == "PASS"
    assert evidence["execution_environment"] == "native_host"
    assert evidence["execution_os_family"] == "windows"
    assert evidence["powershell_runtime"] == "Windows PowerShell"
    assert evidence["native_windows_host_certified"] is True
    assert evidence["full_core_platform_proof_certified_by_this_run"] is False
    assert evidence["certification_commit_parent"] == (
        "6b71a841c894728766fd6f574c9cd53ad12ec5f9"
    )
    assert evidence["final_documentation_commit"] == "pending_pre_commit"
    assert evidence["source_tree_dirty"] is True
    assert evidence["source_tree_diff_sha256"] == "deadbeef"
    assert "application_hosting_proof" in evidence
    assert "interaction_proof" in evidence
    assert '"core_proof"' not in text
    assert "mongodb://" not in text.lower()
    assert evidence["application_hosting_proof"]["full_core_platform_proof"] is False


def test_source_tree_fingerprint_represented(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(args: list[str], **_k: Any) -> Any:
        class _R:
            returncode = 0
            stderr = ""

            def __init__(self) -> None:
                if args[:3] == ["git", "diff", "--binary"]:
                    self.stdout = "diff --git a/x b/x\n"
                elif args[:2] == ["git", "status"]:
                    self.stdout = (
                        "?? applications/local_workspace_application/scripts/"
                        "run-lkw-windows-native-certification.py\n"
                    )
                else:
                    self.stdout = ""

        return _R()

    monkeypatch.setattr(orch, "_run", _fake_run)
    dirty, digest = orch.git_diff_sha256()
    assert dirty is True
    assert isinstance(digest, str) and len(digest) == 64


def test_cleanup_runs_in_finally(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        orch,
        "require_native_windows",
        lambda: {
            "execution_os_version": "10",
            "execution_kernel_release": "10",
            "execution_architecture": "AMD64",
            "python_version": "3.12.0",
        },
    )
    monkeypatch.setattr(orch, "require_powershell", lambda: "Windows PowerShell")
    monkeypatch.setattr(orch, "require_docker", lambda: None)
    monkeypatch.setattr(
        orch,
        "inspect_docker_engine",
        lambda: {
            "docker_engine_os": "linux",
            "docker_engine_architecture": "x86_64",
            "docker_engine_version": "27.0.0",
        },
    )
    monkeypatch.setattr(
        orch,
        "git_rev_parse_head",
        lambda: "6b71a841c894728766fd6f574c9cd53ad12ec5f9",
    )
    monkeypatch.setattr(orch, "git_diff_sha256", lambda: (True, "abc"))

    def _fail_hosting() -> dict[str, Any]:
        calls.append("hosting")
        raise orch.CertificationOrchestratorError("application_hosting_proof_failed:x")

    def _cleanup() -> bool:
        calls.append("cleanup")
        return True

    monkeypatch.setattr(orch, "run_application_hosting_proof", _fail_hosting)
    monkeypatch.setattr(orch, "cleanup_managed_compose", _cleanup)
    code = orch.main(["--pre-commit-certification"])
    assert code == 1
    assert calls == ["hosting", "cleanup"]


def test_failure_cannot_produce_pass_evidence(
    orch: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "LKW_WINDOWS_NATIVE_CERTIFICATION.json"
    monkeypatch.setattr(orch, "_EVIDENCE_PATH", target)
    monkeypatch.setattr(
        orch,
        "require_native_windows",
        lambda: {
            "execution_os_version": "10",
            "execution_kernel_release": "10",
            "execution_architecture": "AMD64",
            "python_version": "3.12.0",
        },
    )
    monkeypatch.setattr(orch, "require_powershell", lambda: "Windows PowerShell")
    monkeypatch.setattr(orch, "require_docker", lambda: None)
    monkeypatch.setattr(
        orch,
        "inspect_docker_engine",
        lambda: {
            "docker_engine_os": "linux",
            "docker_engine_architecture": "x86_64",
            "docker_engine_version": "27.0.0",
        },
    )
    monkeypatch.setattr(
        orch,
        "git_rev_parse_head",
        lambda: "6b71a841c894728766fd6f574c9cd53ad12ec5f9",
    )
    monkeypatch.setattr(orch, "git_diff_sha256", lambda: (True, "abc"))
    monkeypatch.setattr(
        orch,
        "run_application_hosting_proof",
        lambda: (_ for _ in ()).throw(
            orch.CertificationOrchestratorError("application_hosting_proof_failed")
        ),
    )
    monkeypatch.setattr(orch, "cleanup_managed_compose", lambda: True)
    code = orch.main(["--pre-commit-certification"])
    assert code == 1
    assert not target.exists()
