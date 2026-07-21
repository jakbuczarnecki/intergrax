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
        "core_proof": {
            "proof_kind": "platform_application_hosting",
            "proof_id": "core-id",
            "run_id": "core-run",
            "correlation_id": "core-corr",
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


def test_parse_image_id_and_digest_rules(
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
    assert meta["certification_image_id"] == "sha256:abc123"
    assert meta["certification_image_repo_digest"] == "unavailable"


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
            "certification_image_id": "sha256:img",
            "certification_image_repo_digest": "unavailable",
        },
        inside=_pass_summary(),
        source_commit="022d3f9dadbf250051f69bf513d408fc05d0a333",
        source_tree_dirty=True,
        source_tree_diff_sha256="abc",
    )
    assert evidence["orchestrator_host_os"] in {"windows", "linux", "darwin"}
    assert evidence["docker_engine_os"] == "linux"
    assert evidence["execution_os_family"] == "linux"
    assert evidence["native_linux_host_certified"] is False


def test_rejects_malformed_inside_output(orch: ModuleType) -> None:
    with pytest.raises(
        orch.CertificationOrchestratorError, match="malformed_in_container_output"
    ):
        orch.extract_json_summary("not json at all")


def test_rejects_failed_core_and_interaction(orch: ModuleType) -> None:
    bad_core = _pass_summary()
    bad_core["core_proof"]["result"] = "FAIL"
    with pytest.raises(orch.CertificationOrchestratorError, match="core_proof_failed"):
        orch.validate_inside_summary(bad_core)

    bad_ix = _pass_summary()
    bad_ix["interaction_proof"]["receipt_verified"] = False
    with pytest.raises(
        orch.CertificationOrchestratorError, match="interaction_false_receipt_flag"
    ):
        orch.validate_inside_summary(bad_ix)


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
            "certification_image_id": "sha256:img",
            "certification_image_repo_digest": "unavailable",
        },
        inside=_pass_summary(),
        source_commit="022d3f9dadbf250051f69bf513d408fc05d0a333",
        source_tree_dirty=True,
        source_tree_diff_sha256="abc",
    )
    text = json.dumps(evidence, indent=2, sort_keys=True)
    assert "schema_version" in text
    assert "certification_profile" in text
    assert "linux_docker_runtime" in text
    assert "mongodb://" not in text.lower()


def test_inside_stops_after_core_failure(
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

    def _fail_core(**_k: Any) -> dict[str, Any]:
        raise inside.CertificationInsideError("core_proof_failed:boom")

    called: list[str] = []

    def _interaction(**_k: Any) -> dict[str, Any]:
        called.append("interaction")
        return {}

    monkeypatch.setattr(inside, "run_core_proof", _fail_core)
    monkeypatch.setattr(inside, "run_interaction_proof", _interaction)
    code = inside.main([])
    assert code == 1
    assert called == []


def test_inside_rejects_non_linux(inside: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inside.platform, "system", lambda: "Windows")
    with pytest.raises(inside.CertificationInsideError, match="non_linux_runtime"):
        inside.detect_linux_runtime()


def test_inside_build_summary_requires_pass_fields(inside: ModuleType) -> None:
    summary = inside.build_summary(
        runtime={
            "os_version": "v",
            "kernel_release": "k",
            "architecture": "a",
        },
        core={
            "proof_kind": "platform_application_hosting",
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
