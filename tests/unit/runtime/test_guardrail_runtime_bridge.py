# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.guardrail_runtime_bridge import (
    apply_guardrail_profiles_to_runtime_config,
    resolve_guardrail_backend,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    GuardrailProfile,
)
from intergrax.integrations.providers.llm_guardrail.register_all import register_llm_guardrail_integrations
from intergrax.integrations.registry.presets import harness_guardrail_stack
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_guardrail_backend_chains_semantic_slug() -> None:
    register_llm_guardrail_integrations(override=True)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr.chain")
    env = env.model_copy(
        update={
            "integration_profile": harness_guardrail_stack(
                primary="llm_guard",
                semantic="presidio",
            ),
            "guardrail_profile": GuardrailProfile(enabled=True),
        },
    )
    backend = resolve_guardrail_backend(env)
    assert backend is not None
    assert backend.slug == "llm_guard+presidio"


def test_apply_guardrail_profiles_attaches_runtime_middleware() -> None:
    register_llm_guardrail_integrations(override=True)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="gr.runtime")
    env = env.model_copy(
        update={
            "integration_profile": harness_guardrail_stack(primary="llm_guard"),
            "guardrail_profile": GuardrailProfile(enabled=True),
        },
    )
    config = RuntimeConfig(llm_adapter=object())  # type: ignore[arg-type]
    apply_guardrail_profiles_to_runtime_config(config, env)
    assert "guardrail_middleware" in config.metadata
