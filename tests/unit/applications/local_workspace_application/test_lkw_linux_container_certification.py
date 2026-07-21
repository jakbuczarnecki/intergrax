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
    / "run-lkw-linux-container-certification.py"
)
_INSIDE = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-linux-container-certification-inside.py"
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
    return _load(_ORCH, "lkw_linux_container_cert_orch")


@pytest.fixture(scope="module")
def inside() -> ModuleType:
    return _load(_INSIDE, "lkw_linux_container_cert_inside")


def _pass_summary() -> dict[str, Any]:
    return {
        "schema_version": "lkw.linux_docker_certification_inside.v1",
        "certification_result": "PASS",
        "certification_profile": "linux_docker_runtime",
        "execution_environment": "container",
        "execution_os_family": "linux",
        "os_version": "linux-test",
        "kernel_release": "6.1.0",
        "architecture": "x86_64",
        "containerized": True,
        "container_runtime": "docker",
        "client_runtime": "python",
        "wrapper_runtime": "posix_sh",
        "source_commit": "abc",
        "full_core_platform_proof_certified": False,
        "application_hosting_proof": {
            "proof_kind": "platform_application_hosting",
            "certified_scope": "application_hosting_phase",
            "full_core_platform_proof": False,
            "proof_id": "hosting-id",
            "run_id": "hosting-run",
            "correlation_id": "hosting-corr",
            "result": "PASS",
            "receipt_recorded": True,
            "receipt_verified": True,
            "receipt_query_verified": True,
        },
        "interaction_proof": {
            "proof_kind": "platform_linux_interaction",
            "proof_id": "ix-id",
            "run_id": "ix-run",
            "correlation_id": "ix-corr",
            "adapter_id": "lkw.linux_shell",
            "source": "linux_shell",
            "client_runtime": "python",
            "wrapper_runtime": "posix_sh",
            "result": "PASS",
            "receipt_recorded": True,
            "receipt_verified": True,
            "receipt_query_verified": True,
        },
    }


def test_rejects_missing_docker(orch: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orch.shutil, "which", lambda _name: None)
    with pytest.raises(orch.CertificationOrchestratorError, match="docker_unavailable"):
        orch.require_docker()


def test_rejects_windows_container_mode(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(args: list[str], **_k: Any) -> Any:
        class _R:
            returncode = 0
            stdout = "windows\n"
            stderr = ""

        if args[:3] == ["docker", "info", "--format"]:
            return _R()
        return _R()

    monkeypatch.setattr(orch, "_run", _fake_run)
    with pytest.raises(
        orch.CertificationOrchestratorError, match="windows_container_mode"
    ):
        orch.inspect_docker_engine()


def test_accepts_linux_docker_engine(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(args: list[str], **_k: Any) -> Any:
        class _R:
            returncode = 0
            stderr = ""

            def __init__(self) -> None:
                fmt = args[-1] if args else ""
                if "OSType" in fmt:
                    self.stdout = "linux\n"
                elif "Architecture" in fmt:
                    self.stdout = "x86_64\n"
                elif "Server.Version" in fmt:
                    self.stdout = "27.5.1\n"
                else:
                    self.stdout = "\n"

        return _R()

    monkeypatch.setattr(orch, "_run", _fake_run)
    meta = orch.inspect_docker_engine()
    assert meta["docker_engine_os"] == "linux"
    assert meta["docker_engine_architecture"] == "x86_64"
    assert meta["docker_engine_version"] == "27.5.1"


def test_empty_repo_digests_yields_unavailable(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(args: list[str], **_k: Any) -> Any:
        class _R:
            returncode = 0
            stderr = ""

            def __init__(self) -> None:
                if "{{.Id}}" in args:
                    self.stdout = "sha256:abc123\n"
                elif "RepoDigests" in args[-1]:
                    self.stdout = "[]\n"
                else:
                    self.stdout = "\n"

        return _R()

    monkeypatch.setattr(orch, "_run", _fake_run)
    meta = orch.resolve_image_metadata("intergrax-lkw-linux-certification:local")
    assert meta["certification_image_reference"] == (
        "intergrax-lkw-linux-certification:local"
    )
    assert meta["certification_image_id"] == "sha256:abc123"
    assert meta["certification_image_repo_digest"] == "unavailable"
    assert meta["raw_repo_digests"] == []
    assert meta["certification_image_repo_digest"] != meta["certification_image_id"]


def test_real_repo_digest_is_parsed(
    orch: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(args: list[str], **_k: Any) -> Any:
        class _R:
            returncode = 0
            stderr = ""

            def __init__(self) -> None:
                if "{{.Id}}" in args:
                    self.stdout = "sha256:localimageid\n"
                elif "RepoDigests" in args[-1]:
                    self.stdout = (
                        '["registry.example/intergrax@sha256:repodigest99"]\n'
                    )
                else:
                    self.stdout = "\n"

        return _R()

    monkeypatch.setattr(orch, "_run", _fake_run)
    meta = orch.resolve_image_metadata("intergrax-lkw-linux-certification:local")
    assert meta["certification_image_id"] == "sha256:localimageid"
    assert meta["certification_image_repo_digest"] == "sha256:repodigest99"
    assert meta["raw_repo_digests"] == [
        "registry.example/intergrax@sha256:repodigest99"
    ]
    assert meta["certification_image_repo_digest"] != meta["certification_image_id"]


def test_does_not_invent_repository_digest(orch: ModuleType) -> None:
    evidence = orch.scrub_secrets(
        {
            "certification_image_repo_digest": "unavailable",
            "password": "secret",
            "INTERGRAX_MONGODB_URI": "mongodb://u:p@host/db",
        }
    )
    assert evidence["certification_image_repo_digest"] == "unavailable"
    assert "password" not in evidence
    assert "INTERGRAX_MONGODB_URI" not in evidence


def test_distinguishes_host_and_execution_os(orch: ModuleType) -> None:
    evidence = orch.build_evidence(
        engine={
            "docker_engine_os": "linux",
            "docker_engine_architecture": "x86_64",
            "docker_engine_version": "27.5.1",
        },
        base_image={
            "container_base_image": "python:3.12-slim-bookworm",
            "container_base_image_digest": "sha256:deadbeef",
        },
        image_meta={
            "certification_image_reference": "intergrax-lkw-linux-certification:local",
            "certification_image_id": "sha256:img",
            "certification_image_repo_digest": "unavailable",
        },
        inside=_pass_summary(),
        source_commit="40a73fbb455def6d5106180d74a7e65388457465",
        source_tree_dirty=True,
        source_tree_diff_sha256="abc",
    )
    assert evidence["orchestrator_host_os"] in {"windows", "linux", "darwin"}
    assert evidence["docker_engine_os"] == "linux"
    assert evidence["execution_os_family"] == "linux"
    assert evidence["native_linux_host_certified"] is False
    assert evidence["full_core_platform_proof_certified"] is False
    assert "application_hosting_proof" in evidence
    assert "core_proof" not in evidence


def test_rejects_malformed_inside_output(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="malformed_in_container_output"
    ):
        orch.extract_json_summary("not json at all")


def test_rejects_failed_application_hosting_and_interaction(orch: ModuleType) -> None:
    bad_hosting = _pass_summary()
    bad_hosting["application_hosting_proof"]["result"] = "FAIL"
    with pytest.raises(
        orch.CertificationOrchestratorError, match="application_hosting_proof_failed"
    ):
        orch.validate_inside_summary(bad_hosting)

    bad_ix = _pass_summary()
    bad_ix["interaction_proof"]["receipt_verified"] = False
    with pytest.raises(
        orch.CertificationOrchestratorError, match="interaction_false_receipt_flag"
    ):
        orch.validate_inside_summary(bad_ix)


def test_rejects_legacy_core_proof_only_summary(orch: ModuleType) -> None:
    legacy = _pass_summary()
    legacy["core_proof"] = legacy.pop("application_hosting_proof")
    with pytest.raises(
        orch.CertificationOrchestratorError, match="missing_application_hosting_proof"
    ):
        orch.validate_inside_summary(legacy)


def test_rejects_missing_interaction_proof(orch: ModuleType) -> None:
    bad = _pass_summary()
    del bad["interaction_proof"]
    with pytest.raises(
        orch.CertificationOrchestratorError, match="missing_interaction_proof"
    ):
        orch.validate_inside_summary(bad)


def test_rejects_full_core_platform_proof_certified_true(orch: ModuleType) -> None:
    bad = _pass_summary()
    bad["full_core_platform_proof_certified"] = True
    with pytest.raises(
        orch.CertificationOrchestratorError,
        match="full_core_platform_proof_must_not_be_certified",
    ):
        orch.validate_inside_summary(bad)


def test_accepts_corrected_pass_summary(orch: ModuleType) -> None:
    orch.validate_inside_summary(_pass_summary())


def test_evidence_schema_deterministic(orch: ModuleType) -> None:
    evidence = orch.build_evidence(
        engine={
            "docker_engine_os": "linux",
            "docker_engine_architecture": "x86_64",
            "docker_engine_version": "27.5.1",
        },
        base_image={
            "container_base_image": "python:3.12-slim-bookworm",
            "container_base_image_digest": "sha256:deadbeef",
        },
        image_meta={
            "certification_image_reference": "intergrax-lkw-linux-certification:local",
            "certification_image_id": "sha256:img",
            "certification_image_repo_digest": "unavailable",
        },
        inside=_pass_summary(),
        source_commit="40a73fbb455def6d5106180d74a7e65388457465",
        source_tree_dirty=True,
        source_tree_diff_sha256="abc",
    )
    text = json.dumps(evidence, indent=2, sort_keys=True)
    assert "schema_version" in text
    assert "certification_profile" in text
    assert "linux_docker_runtime" in text
    assert "application_hosting_proof" in text
    assert "full_core_platform_proof_certified" in text
    assert "certification_image_reference" in text
    assert '"core_proof"' not in text
    assert "mongodb://" not in text.lower()
    assert evidence["application_hosting_proof"]["proof_kind"] == (
        "platform_application_hosting"
    )
    assert evidence["application_hosting_proof"]["certified_scope"] == (
        "application_hosting_phase"
    )
    assert evidence["application_hosting_proof"]["full_core_platform_proof"] is False


def test_inside_stops_after_application_hosting_failure(
    inside: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inside,
        "detect_linux_runtime",
        lambda: {
            "platform_system": "Linux",
            "os_version": "x",
            "kernel_release": "y",
            "architecture": "z",
        },
    )
    monkeypatch.setenv("INTERGRAX_MONGODB_URI", "mongodb://example/db")

    def _fail_hosting(**_k: Any) -> dict[str, Any]:
        raise inside.CertificationInsideError("application_hosting_proof_failed:boom")

    called: list[str] = []

    def _interaction(**_k: Any) -> dict[str, Any]:
        called.append("interaction")
        return {}

    monkeypatch.setattr(inside, "run_application_hosting_proof", _fail_hosting)
    monkeypatch.setattr(inside, "run_interaction_proof", _interaction)
    code = inside.main([])
    assert code == 1
    assert called == []


def test_inside_rejects_non_linux(inside: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inside.platform, "system", lambda: "Windows")
    with pytest.raises(inside.CertificationInsideError, match="non_linux_runtime"):
        inside.detect_linux_runtime()


def test_inside_build_summary_uses_application_hosting_fields(inside: ModuleType) -> None:
    summary = inside.build_summary(
        runtime={
            "os_version": "v",
            "kernel_release": "k",
            "architecture": "a",
        },
        application_hosting={
            "proof_kind": "platform_application_hosting",
            "certified_scope": "application_hosting_phase",
            "full_core_platform_proof": False,
            "proof_id": "c",
            "run_id": "r",
            "correlation_id": "x",
            "result": "PASS",
            "receipt_recorded": True,
            "receipt_verified": True,
            "receipt_query_verified": True,
        },
        interaction={
            "proof_kind": "platform_linux_interaction",
            "proof_id": "i",
            "run_id": "ir",
            "correlation_id": "ix",
            "adapter_id": "lkw.linux_shell",
            "source": "linux_shell",
            "client_runtime": "python",
            "wrapper_runtime": "posix_sh",
            "result": "PASS",
            "receipt_recorded": True,
            "receipt_verified": True,
            "receipt_query_verified": True,
        },
    )
    assert summary["certification_result"] == "PASS"
    assert summary["execution_os_family"] == "linux"
    assert summary["wrapper_runtime"] == "posix_sh"
    assert "application_hosting_proof" in summary
    assert "core_proof" not in summary
    assert summary["full_core_platform_proof_certified"] is False
    assert summary["application_hosting_proof"]["proof_kind"] == (
        "platform_application_hosting"
    )
    assert summary["application_hosting_proof"]["certified_scope"] == (
        "application_hosting_phase"
    )
    assert summary["application_hosting_proof"]["full_core_platform_proof"] is False
    assert summary["interaction_proof"]["adapter_id"] == "lkw.linux_shell"
    assert summary["interaction_proof"]["source"] == "linux_shell"
    assert summary["interaction_proof"]["client_runtime"] == "python"
    assert summary["interaction_proof"]["wrapper_runtime"] == "posix_sh"
