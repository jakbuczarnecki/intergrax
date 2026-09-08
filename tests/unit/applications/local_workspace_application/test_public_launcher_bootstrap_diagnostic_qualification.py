# © Artur Czarnecki. All rights reserved.

"""DG-001C — public launcher / bootstrap diagnostic visibility qualification."""

from __future__ import annotations

import importlib.util
import inspect
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

from intergrax.applications._shared.hosted_application_diagnostic_wiring import (
    HostedApplicationDiagnosticEventPublisher,
    build_hosted_application_diagnostic_event_publisher,
)
from intergrax.hosting.eventing import ObservabilityHostedApplicationEventPublisher
from intergrax.hosting.runner import _default_runner_factories
from local_workspace_application.hosting import foreground as foreground_module
from local_workspace_application.hosting.__main__ import main as hosting_main

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO_ROOT / "applications/local_workspace_application/scripts"
_CORE_PROOF = _SCRIPTS / "run-lkw-core-platform-proof.py"
_CORE_WINDOWS_BAT = _SCRIPTS / "run-lkw-core-platform-proof-windows.bat"
_CORE_LINUX_SH = _SCRIPTS / "run-lkw-core-platform-proof-linux.sh"
_QUICKSTART_BAT = _SCRIPTS / "run-lkw-product-quickstart-windows.bat"
_FOREGROUND = (
    _REPO_ROOT
    / "applications/local_workspace_application/hosting/foreground.py"
)
_HOSTING_MAIN = (
    _REPO_ROOT
    / "applications/local_workspace_application/hosting/__main__.py"
)


def _load_core_proof_module() -> ModuleType:
    module_name = "run_lkw_core_platform_proof_dg001c"
    spec = importlib.util.spec_from_file_location(module_name, _CORE_PROOF)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core_proof() -> ModuleType:
    return _load_core_proof_module()


@pytest.mark.parametrize(
    "path",
    [
        _CORE_WINDOWS_BAT,
        _CORE_LINUX_SH,
        _QUICKSTART_BAT,
    ],
)
def test_public_wrappers_delegate_to_uv_project_python(path: Path) -> None:
    text = " ".join(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    assert "uv run --project applications/local_workspace_application python" in text


def test_core_proof_runner_does_not_wire_host_diag_3_publisher() -> None:
    source = _CORE_PROOF.read_text(encoding="utf-8")
    assert "build_hosted_application_diagnostic_event_publisher" not in source
    assert "HostedApplicationDiagnosticEventPublisher" not in source
    assert "HostedDiagnosticTenantBinding" not in source
    assert "DiagnosticOrchestrator" not in source


def test_product_quickstart_runner_does_not_wire_host_diag_3_publisher() -> None:
    source = (_SCRIPTS / "run-lkw-product-quickstart.py").read_text(encoding="utf-8")
    assert "build_hosted_application_diagnostic_event_publisher" not in source
    assert "run_local_workspace_hosted_application" not in source
    assert "run_hosted_application" not in source


def test_lkw_foreground_composition_root_uses_canonical_host_runner() -> None:
    source = _FOREGROUND.read_text(encoding="utf-8")
    assert "run_hosted_application" in source
    assert "build_hosted_application_diagnostic_event_publisher" in source
    signature = inspect.signature(foreground_module.run_local_workspace_hosted_application)
    assert "diagnostic_orchestrator" in signature.parameters
    assert "diagnostic_tenant_binding" in signature.parameters


def test_lkw_foreground_default_path_has_no_event_publisher_factory_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(foreground_module, "run_hosted_application", _fake_run)
    monkeypatch.setattr(
        foreground_module,
        "build_local_workspace_hosted_profile",
        lambda **_kwargs: object(),
    )

    foreground_module.run_local_workspace_hosted_application(
        process_composition=object(),  # type: ignore[arg-type]
    )

    assert captured.get("event_publisher_factory") is None


def test_platform_default_publisher_is_observability_only() -> None:
    publisher = _default_runner_factories().create_event_publisher()
    assert isinstance(publisher, ObservabilityHostedApplicationEventPublisher)
    assert not isinstance(publisher, HostedApplicationDiagnosticEventPublisher)


def test_hosting_cli_rejects_bootstrap_without_composition() -> None:
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        code = hosting_main()
    assert code == 1
    assert "requires an activated process composition" in stderr.getvalue()


def test_python_bootstrap_failure_surfaces_kv_not_central_diagnostics(
    core_proof: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_proof, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(
        core_proof,
        "validate_environment",
        lambda *_a, **_k: (_ for _ in ()).throw(core_proof.CoreProofError("uv_missing")),
    )
    cfg = core_proof.ProofConfig(
        os_family=core_proof.OsFamily.WINDOWS,
        wrapper_id=core_proof.WrapperId.WINDOWS_BAT,
        phase="all",
        run_id_prefix="dg001c-",
        base_url="http://127.0.0.1:8020",
        kafka_ui="http://127.0.0.1:8085",
        mongo_express="http://127.0.0.1:8086",
        elasticsearch_url="http://127.0.0.1:9200",
        kibana_url="http://127.0.0.1:5601",
        sentry_url="http://127.0.0.1:9000",
        phase_timeout_seconds=30,
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core_proof.run_core_proof(cfg, phase_runners={})
    text = buffer.getvalue()
    assert code == 1
    assert "failure_reason=uv_missing" in text


def test_canonical_host_diag_3_factory_remains_product_owned() -> None:
    signature = inspect.signature(build_hosted_application_diagnostic_event_publisher)
    assert "tenant_binding" in signature.parameters
    assert "orchestrator" in signature.parameters
