# © Artur Czarnecki. All rights reserved.

"""Host environment wiring must not eagerly materialize LLM adapters (NPSC-3B-R3V-R5)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest_default

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_environment_wiring_without_llm_selection_does_not_materialize_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INTERGRAX_LLM_PROVIDER", raising=False)
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="host.no-llm")
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    manifest = build_lab_manifest_default()
    with patch(
        "intergrax.applications._shared.llm_resolver.resolve_optional_environment_llm_adapter",
    ) as resolve_optional_mock:
        wire_application_environment(manifest, env, settings=settings, conformance_check=False)
    resolve_optional_mock.assert_not_called()
