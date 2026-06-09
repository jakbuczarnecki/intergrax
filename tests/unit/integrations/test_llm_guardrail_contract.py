# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.llm_guardrail._stub_backend import create_stub_guardrail
from intergrax.integrations.providers.llm_guardrail.register_all import register_llm_guardrail_integrations
from intergrax.integrations.registry.catalog import get_entry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_stub_guardrail_blocks_input() -> None:
    backend = create_stub_guardrail("llm_guard")
    result = backend.scan_input("BLOCK_INPUT test")
    assert result.allowed is False


def test_llm_guardrail_category_registered() -> None:
    assert IntegrationCategory.LLM_GUARDRAIL.value == "llm_guardrail"


def test_register_llm_guardrail_integrations() -> None:
    register_llm_guardrail_integrations(override=True)
    entry = get_entry("llm_guard")
    assert entry is not None
    assert IntegrationCategory.LLM_GUARDRAIL in entry.categories
