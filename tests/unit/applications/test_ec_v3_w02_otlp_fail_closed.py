# © Artur Czarnecki. All rights reserved.

"""EC-V3-W02 OTLP guarded composition regression (A4-032a, T-IGX-074-A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from intergrax.applications._shared.runtime_event_delivery_wiring import (
    ApplicationRuntimeEventDeliveryWiring,
    _create_export_transport,
    resolve_application_runtime_event_delivery_wiring,
    resolve_observability_export_profile,
)
from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.observability_export import (
    ConfigurationError,
    EventExportSinkFactoryPort,
    ExporterKind,
    ObservabilityExportProfile,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_OTLP_ENDPOINT_PROFILE = "http://127.0.0.1:0/v1/logs"
_OTLP_ENDPOINT_SETTINGS = "http://127.0.0.1:0/v1/traces"


@dataclass
class _OtlpSettings:
    observability_otlp_endpoint: str
    observability_otlp_timeout_seconds: float = 12.0


def _with_posture(
    base: ApplicationEnvironmentProfile,
    *,
    profile: ApplicationProfile,
    mode: ExecutionMode,
) -> ApplicationEnvironmentProfile:
    meta = base.meta.model_copy(update={"application_profile": profile, "execution_mode": mode})
    return base.model_copy(update={"meta": meta})


def _otlp_enabled_env(
    base: ApplicationEnvironmentProfile,
    *,
    endpoint: str,
    exporter_kind: ExporterKind = ExporterKind.OTLP,
) -> ApplicationEnvironmentProfile:
    env = base.model_copy(deep=True)
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": exporter_kind,
            "otlp_export_endpoint": endpoint,
        },
    )
    return env


def _patch_otlp_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.runtime_event_delivery_wiring.require_otlp_observability_dependency_profile",
        lambda **_: None,
    )


def test_otlp_01_product_empty_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint="",
    )
    with patch(
        "intergrax.runtime.observability.exporters.otlp.otlp_transport.OtlpTransport",
        MagicMock(),
    ) as transport_cls:
        with pytest.raises(ConfigurationError, match="otlp_export_endpoint"):
            resolve_application_runtime_event_delivery_wiring(env)
        transport_cls.assert_not_called()


def test_otlp_02_lab_strict_empty_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.otlp.02")
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.LAB, mode=ExecutionMode.STRICT),
        endpoint="",
    )
    assert env.application_profile is ApplicationProfile.LAB
    assert env.execution_mode is ExecutionMode.STRICT
    assert wiring_fails_closed(env) is True
    with pytest.raises(ConfigurationError, match="otlp_export_endpoint"):
        resolve_application_runtime_event_delivery_wiring(env)


def test_otlp_03_whitespace_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint="   ",
    )
    with pytest.raises(ConfigurationError):
        resolve_application_runtime_event_delivery_wiring(env)


def test_otlp_04_valid_profile_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint=_OTLP_ENDPOINT_PROFILE,
    )
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.otlp_transport is not None
    assert isinstance(wiring.otlp_transport, OtlpTransport)


def test_otlp_05_settings_endpoint_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint="",
    )
    settings = _OtlpSettings(observability_otlp_endpoint=_OTLP_ENDPOINT_SETTINGS)
    wiring = resolve_application_runtime_event_delivery_wiring(env, settings=settings)
    assert wiring.otlp_transport is not None
    if TYPE_CHECKING:
        assert wiring.otlp_transport is not None
    transport = wiring.otlp_transport
    assert getattr(transport, "_config").endpoint == _OTLP_ENDPOINT_SETTINGS


def test_otlp_06_settings_precedence_over_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint=_OTLP_ENDPOINT_PROFILE,
    )
    settings = _OtlpSettings(observability_otlp_endpoint=_OTLP_ENDPOINT_SETTINGS)
    wiring = resolve_application_runtime_event_delivery_wiring(env, settings=settings)
    assert wiring.otlp_transport is not None
    assert getattr(wiring.otlp_transport, "_config").endpoint == _OTLP_ENDPOINT_SETTINGS


def test_otlp_07_non_guarded_lab_preserves_noop_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.otlp.07")
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.OTLP,
            "otlp_export_endpoint": "",
        },
    )
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.otlp_transport is None


def test_otlp_08_explicitly_disabled_delivery() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    env.observability_profile = env.observability_profile.model_copy(
        update={"bounded_event_delivery_enabled": False},
    )
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring == ApplicationRuntimeEventDeliveryWiring.disabled()


def test_otlp_09_distributed_missing_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint=_OTLP_ENDPOINT_PROFILE,
        exporter_kind=ExporterKind.DISTRIBUTED_OTLP,
    )
    env.observability_profile = env.observability_profile.model_copy(
        update={"observability_export_service_name": ""},
    )
    with pytest.raises(ConfigurationError, match="DISTRIBUTED_OTLP"):
        resolve_application_runtime_event_delivery_wiring(env)


def test_otlp_10_injected_factory_not_called_on_missing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint="",
    )

    class _SpyFactory(EventExportSinkFactoryPort):
        def __init__(self) -> None:
            self.called = False

        def create(self, _profile: ObservabilityExportProfile):  # noqa: ANN001
            self.called = True
            raise AssertionError("factory must not run on invalid OTLP configuration")

    factory = _SpyFactory()
    with pytest.raises(ConfigurationError):
        resolve_application_runtime_event_delivery_wiring(env, export_sink_factory=factory)
    assert factory.called is False


def test_otlp_product_balanced_empty_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.BALANCED),
        endpoint="",
    )
    assert wiring_fails_closed(env) is True
    with pytest.raises(ConfigurationError, match="otlp_export_endpoint"):
        resolve_application_runtime_event_delivery_wiring(env)


def test_otlp_product_strict_whitespace_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ec.w02.otlp.strict.ws")
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.LAB, mode=ExecutionMode.STRICT),
        endpoint="   ",
    )
    assert wiring_fails_closed(env) is True
    with pytest.raises(ConfigurationError):
        resolve_application_runtime_event_delivery_wiring(env)


def test_otlp_guard_raises_before_transport_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_otlp_dependency(monkeypatch)
    base = ApplicationEnvironmentProfile.product_defaults()
    env = _otlp_enabled_env(
        _with_posture(base, profile=ApplicationProfile.PRODUCT, mode=ExecutionMode.STRICT),
        endpoint="",
    )
    export_profile = resolve_observability_export_profile(env)
    with patch(
        "intergrax.runtime.observability.exporters.otlp.otlp_transport.OtlpTransport",
        MagicMock(),
    ) as transport_cls:
        with pytest.raises(ConfigurationError):
            _create_export_transport(export_profile, env, settings=None)
        transport_cls.assert_not_called()
