# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from intergrax.applications._shared.settings_loader import (
    ApplicationSettingsEnvHost,
    EnvReader,
    load_application_settings_from_env,
)
from intergrax.applications.contracts.settings import IntergraxApplicationSettingsBase

pytestmark = pytest.mark.unit


@dataclass(frozen=True, kw_only=True)
class CustomSettings(ApplicationSettingsEnvHost, IntergraxApplicationSettingsBase):
    env_prefix: ClassVar[str] = "CUSTOM_"
    crm_api_url: str = ""

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        return {"crm_api_url": env.str("CRM_API_URL", default="")}


def test_custom_settings_inherits_base_and_loads_platform_and_app_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_ROUTE_PREFIX", "/v1/custom")
    monkeypatch.setenv("CUSTOM_BACKEND_PORT", "9100")
    monkeypatch.setenv("CUSTOM_CRM_API_URL", "https://crm.example.test")

    settings = CustomSettings.from_env()

    assert isinstance(settings, IntergraxApplicationSettingsBase)
    assert settings.route_prefix == "/v1/custom"
    assert settings.backend_port == 9100
    assert settings.crm_api_url == "https://crm.example.test"


def test_custom_settings_does_not_require_from_env_override() -> None:
    assert "from_env" not in CustomSettings.__dict__
    assert (
        CustomSettings.from_env.__func__
        is ApplicationSettingsEnvHost.from_env.__func__
    )


def test_contract_settings_module_has_no_from_env() -> None:
    assert not hasattr(IntergraxApplicationSettingsBase, "from_env")


def test_loader_is_explicit_env_owner() -> None:
    @dataclass(frozen=True, kw_only=True)
    class _HostSettings(ApplicationSettingsEnvHost, IntergraxApplicationSettingsBase):
        pass

    settings = load_application_settings_from_env(_HostSettings)
    assert settings.backend_host == "127.0.0.1"
