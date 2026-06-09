# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.integrations._shared.conformance import assert_llm_guardrail_backend
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.llm_guardrail._factory import create_guardrail_backend
from intergrax.integrations.providers.llm_guardrail.register_all import register_llm_guardrail_integrations
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_guardrail_backend_blocks_injection() -> None:
    backend = create_guardrail_backend("llm_guard")
    result = backend.scan_input("BLOCK_INPUT test")
    assert result.allowed is False
    assert "test_block" in result.categories


def test_guardrail_backend_conformance() -> None:
    backend = create_guardrail_backend("presidio")
    assert_llm_guardrail_backend(backend)
    assert backend.health_check() is True


def test_llm_guardrail_category_registered() -> None:
    assert IntegrationCategory.LLM_GUARDRAIL.value == "llm_guardrail"


def test_register_llm_guardrail_integrations() -> None:
    register_llm_guardrail_integrations(override=True)
    entry = get_entry("llm_guard")
    assert entry is not None
    assert IntegrationCategory.LLM_GUARDRAIL in entry.categories


def test_integration_profile_resolves_llm_guardrail() -> None:
    register_llm_guardrail_integrations(override=True)
    profile = IntegrationProfile(llm_guardrail="llm_guard")
    backend = profile.resolve(IntegrationCategory.LLM_GUARDRAIL)
    assert_llm_guardrail_backend(backend)
