# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import LLMRoutingProfile


@pytest.mark.unit
@pytest.mark.gate
def test_llm_routing_profile_json_schema_skips_runtime_rules() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile")
    routing = LLMRoutingProfile(default_profile=profile)

    schema = LLMRoutingProfile.model_json_schema()

    assert "rules" not in schema.get("properties", {})


@pytest.mark.unit
@pytest.mark.gate
def test_application_environment_profile_json_schema_includes_llm_routing() -> None:
    schema = ApplicationEnvironmentProfile.model_json_schema()

    assert schema
    assert "properties" in schema
