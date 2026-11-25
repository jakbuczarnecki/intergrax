# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) OBS-1..OBS-4: OTLP composition fails closed in PRODUCT/STRICT postures."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
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
    env = base.model_copy(
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
    return env


def test_product_otlp_without_endpoint_raises_configuration_error(
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


def test_strict_lab_otlp_without_endpoint_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc2.otlp.strict")
    meta = base.meta.model_copy(
        update={
            "application_profile": ApplicationProfile.LAB,
            "execution_mode": ExecutionMode.STRICT,
        },
    )
    env = _otlp_env(base.model_copy(update={"meta": meta}), endpoint="")
    with pytest.raises(ConfigurationError, match="otlp_export_endpoint"):
        resolve_application_runtime_event_delivery_wiring(env)


def test_settings_endpoint_satisfies_otlp_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    env = _otlp_env(ApplicationEnvironmentProfile.product_defaults(), endpoint="")
    settings = _OtlpSettings(observability_otlp_endpoint=_OTLP_ENDPOINT)
    wiring = resolve_application_runtime_event_delivery_wiring(env, settings=settings)
    assert wiring.otlp_transport is not None
    assert isinstance(wiring.otlp_transport, OtlpTransport)


def test_lab_balanced_otlp_without_endpoint_keeps_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc2.otlp.balanced")
    meta = base.meta.model_copy(
        update={
            "application_profile": ApplicationProfile.LAB,
            "execution_mode": ExecutionMode.BALANCED,
        },
    )
    env = _otlp_env(base.model_copy(update={"meta": meta}), endpoint="")
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.otlp_transport is None
