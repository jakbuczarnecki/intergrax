# © Artur Czarnecki. All rights reserved.

"""W5-H1 — OTLP optional capability dependency & qualification profile contract."""

from __future__ import annotations

import builtins
import tomllib
from pathlib import Path

import pytest

from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import GovernanceBundle
from intergrax.contracts.observability_export import (
    ConfigurationError,
    ExporterKind,
    OtlpExportConfiguration,
    OtlpProtocol,
)
from intergrax.runtime.observability.exporters.otlp.otlp_dependency import (
    OTLP_OBSERVABILITY_PROFILE_EXTRA,
    require_otlp_observability_dependency_profile,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]
_OTLP_PACKAGE_PREFIXES = (
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-exporter-otlp-proto-grpc",
)

_CORE_RUNTIME_SCAN_ROOTS = (
    REPO_ROOT / "intergrax" / "runtime" / "execution",
    REPO_ROOT / "intergrax" / "runtime" / "events",
    REPO_ROOT / "intergrax" / "runtime" / "recovery",
    REPO_ROOT / "agents",
)


def _otlp_specs_from_pyproject() -> dict[str, set[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]
    test_group = set(data["dependency-groups"]["test"])
    extra = set(optional[OTLP_OBSERVABILITY_PROFILE_EXTRA])
    dev_ci = set(optional["dev-ci"])
    return {
        "test": test_group,
        "observability-otlp": extra,
        "dev-ci": dev_ci,
    }


def _filter_otlp_specs(specs: set[str]) -> set[str]:
    return {
        spec
        for spec in specs
        if any(spec.startswith(f"{prefix}>=") or spec.startswith(f"{prefix}==") for prefix in _OTLP_PACKAGE_PREFIXES)
    }


def test_otlp_adapter_dependency_declaration_matches_qualification_profiles() -> None:
    profiles = _otlp_specs_from_pyproject()
    extra_specs = _filter_otlp_specs(profiles["observability-otlp"])
    test_specs = _filter_otlp_specs(profiles["test"])
    dev_ci_specs = _filter_otlp_specs(profiles["dev-ci"])
    assert extra_specs
    assert extra_specs == test_specs
    assert extra_specs.issubset(dev_ci_specs)


def test_missing_otlp_dependency_fails_with_typed_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _blocking_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "opentelemetry.sdk._logs" or name.startswith("opentelemetry."):
            raise ImportError("simulated missing OTLP profile")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    with pytest.raises(ConfigurationError, match="OTLP observability dependency profile"):
        require_otlp_observability_dependency_profile(
            exporter_kind=ExporterKind.DISTRIBUTED_OTLP,
        )


def test_missing_otlp_dependency_at_wiring_fails_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _blocking_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "opentelemetry.sdk._logs" or name.startswith("opentelemetry."):
            raise ImportError("simulated missing OTLP profile")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w5.h1.missing")
    observability = GovernanceBundle.enterprise_cluster_observability(
        otlp_export_endpoint="http://127.0.0.1:4318/v1/logs",
        observability_export_service_name="intergrax-runtime",
    )
    env = env.model_copy(
        update={
            "governance": env.governance.model_copy(update={"observability": observability}),
        },
    )
    with pytest.raises(ConfigurationError, match="DISTRIBUTED_OTLP requires"):
        resolve_application_runtime_event_delivery_wiring(env)


def test_core_runtime_does_not_import_opentelemetry_sdk() -> None:
    offenders: list[str] = []
    for root in _CORE_RUNTIME_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "opentelemetry" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_otlp_transport_requires_profile_before_sdk_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _blocking_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "opentelemetry.sdk._logs":
            raise ImportError("simulated missing OTLP profile")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    config = OtlpExportConfiguration(
        endpoint="http://127.0.0.1:4318/v1/logs",
        protocol=OtlpProtocol.HTTP_PROTOBUF,
        timeout_seconds=1.0,
    )
    with pytest.raises(ConfigurationError, match="OTLP observability dependency profile"):
        OtlpTransport(config)
