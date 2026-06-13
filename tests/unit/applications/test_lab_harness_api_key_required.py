# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from lab_application.host.factory import create_lab_application
from lab_application.host.settings import LabApplicationSettings
from intergrax.fastapi_core.config import ApiEnvironment

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_staging_profile_requires_harness_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    settings = LabApplicationSettings(environment=ApiEnvironment.STAGE, strict_harness=False)
    assert settings.requires_harness_api_key is True
    with pytest.raises(ValueError, match="INTERGRAX_HARNESS_API_KEY"):
        create_lab_application(settings=settings)


def test_dev_profile_allows_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_HARNESS_API_KEY", raising=False)
    settings = LabApplicationSettings(environment=ApiEnvironment.DEV, strict_harness=False)
    assert settings.requires_harness_api_key is False
    app = create_lab_application(settings=settings)
    assert app.title == "Intergrax Lab Application"


def test_from_env_stage_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_ENV", "stage")
    settings = LabApplicationSettings.from_env()
    assert settings.environment == ApiEnvironment.STAGE
    assert settings.requires_harness_api_key is True
