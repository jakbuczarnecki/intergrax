# © Artur Czarnecki. All rights reserved.

"""T-IGX-074-A (EC-V3-W02) OTLP composition fails closed without production qualification fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from intergrax.applications._shared.package_wiring import (
    PACKAGE_GATE_OTLP_QUALIFICATION_ENDPOINT,
    package_gate_metadata_environment,
)
from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.observability_export import ConfigurationError, ExporterKind
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OTLP_ENDPOINT = "http://127.0.0.1:0/v1/traces"


@dataclass
class _OtlpSettings:
    observability_otlp_endpoint: str
    observability_otlp_timeout_seconds: float = 12.0


def _patch_otlp_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.runtime_event_delivery_wiring.require_otlp_observability_dependency_profile",
        lambda **_: None,
    )


def _otlp_env(
    base: ApplicationEnvironmentProfile,
    *,
    endpoint: str,
) -> ApplicationEnvironmentProfile:
    return base.model_copy(
        update={
            "observability_profile": base.observability_profile.model_copy(
                update={
                    "bounded_event_delivery_enabled": True,
                    "observability_exporter_kind": ExporterKind.OTLP,
                    "otlp_export_endpoint": endpoint,
                },
            ),
        },
    )


def test_product_otlp_without_endpoint_refuses_qualification_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_otlp_dependency(monkeypatch)
    env = _otlp_env(ApplicationEnvironmentProfile.product_defaults(), endpoint="")
    with patch(
        "intergrax.runtime.observability.exporters.otlp.otlp_transport.OtlpTransport",
        MagicMock(),
    ) as transport_cls:
        with pytest.raises(ConfigurationError, match="otlp_export_endpoint"):
            resolve_application_runtime_event_delivery_wiring(env)
        transport_cls.assert_not_called()


def test_strict_lab_otlp_without_endpoint_refuses_qualification_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ecv3.otlp.strict")
    meta = base.meta.model_copy(
        update={
            "application_profile": ApplicationProfile.LAB,
            "execution_mode": ExecutionMode.STRICT,
        },
    )
    env = _otlp_env(base.model_copy(update={"meta": meta}), endpoint="")
    with pytest.raises(ConfigurationError, match="otlp_export_endpoint"):
        resolve_application_runtime_event_delivery_wiring(env)


def test_package_gate_metadata_supplies_explicit_qualification_endpoint() -> None:
    source = ApplicationEnvironmentProfile.product_defaults()
    source = source.model_copy(
        update={
            "observability_profile": source.observability_profile.model_copy(
                update={"otlp_export_endpoint": ""},
            ),
        },
    )
    metadata = package_gate_metadata_environment(source)
    assert metadata.observability_profile.otlp_export_endpoint == PACKAGE_GATE_OTLP_QUALIFICATION_ENDPOINT
    assert metadata.execution_mode == source.execution_mode
    assert metadata.application_profile == source.application_profile


def test_settings_endpoint_satisfies_otlp_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    env = _otlp_env(ApplicationEnvironmentProfile.product_defaults(), endpoint="")
    settings = _OtlpSettings(observability_otlp_endpoint=_OTLP_ENDPOINT)
    wiring = resolve_application_runtime_event_delivery_wiring(env, settings=settings)
    assert wiring.otlp_transport is not None
    assert isinstance(wiring.otlp_transport, OtlpTransport)


def test_lab_balanced_otlp_without_endpoint_keeps_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ecv3.otlp.balanced")
    meta = base.meta.model_copy(
        update={
            "application_profile": ApplicationProfile.LAB,
            "execution_mode": ExecutionMode.BALANCED,
        },
    )
    env = _otlp_env(base.model_copy(update={"meta": meta}), endpoint="")
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.otlp_transport is None
