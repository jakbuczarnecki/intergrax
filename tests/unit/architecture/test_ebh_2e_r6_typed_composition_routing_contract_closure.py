# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6 — typed LLM composition & routing contract closure."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider, llm_provider_slug
from intergrax.llm_adapters.registry.catalog_capabilities import (
    CatalogCapabilityAdapter,
    enrich_adapter_with_catalog_capabilities,
)
from intergrax.llm_adapters.registry.failover_adapter import FailoverLLMAdapter
from intergrax.llm_adapters.registry.failover_policy import PlatformDefaultFailoverPolicy
from intergrax.llm_adapters.registry.model_catalog import ModelRecord
from intergrax.llm_adapters.registry.model_router import ModelRouter
from tests.unit.architecture.ebh_2e_external_structural_llm_adapter import (
    ExternalStructuralAdapter,
    assert_external_structural_llm_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROUTER_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/registry/model_router.py"
_FAILOVER_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/registry/failover_adapter.py"
_CATALOG_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/registry/catalog_capabilities.py"


def test_ebh_2e_r6_llm_provider_slug_enum_and_external() -> None:
    assert llm_provider_slug(LLMProvider.OPENAI) == "openai"
    assert llm_provider_slug("external-provider") == "external-provider"
    assert llm_provider_slug(" External-Provider ") == "external-provider"


@pytest.mark.parametrize(
    ("provider", "expected_slug"),
    [
        (LLMProvider.OPENAI, "openai"),
        ("openai", "openai"),
        ("external-provider", "external-provider"),
        (" External-Provider ", "external-provider"),
    ],
)
def test_ebh_2e_r6_model_router_profile_id_and_resolve(
    provider: LLMProvider | str, expected_slug: str
) -> None:
    profile = LLMProfile(provider=provider, model="m1")
    router = ModelRouter.from_profiles(profile)
    profile_id = f"{expected_slug}:m1"
    assert router.ordered_profile_ids() == (profile_id,)
    decision = router.resolve()
    assert decision.profile_id == profile_id
    assert decision.provider == expected_slug
    assert decision.model == "m1"


def test_ebh_2e_r6_model_router_no_provider_value_assumption_in_source() -> None:
    tree = ast.parse(_ROUTER_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "value":
            if isinstance(node.value, ast.Attribute) and node.value.attr == "provider":
                pytest.fail("ModelRouter must not use profile.provider.value")


def test_ebh_2e_r6_failover_with_structural_adapters() -> None:
    primary = ExternalStructuralAdapter()
    secondary = ExternalStructuralAdapter()
    secondary.model = "ext-backup"
    adapter = FailoverLLMAdapter(
        [primary, secondary],
        profile_ids=("external-structural:ext-model", "external-structural:ext-backup"),
        failover_policy=PlatformDefaultFailoverPolicy(),
    )
    response = adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert response.content == "external"


def test_ebh_2e_r6_catalog_wrapper_with_structural_adapter() -> None:
    inner = ExternalStructuralAdapter()
    record = ModelRecord(
        model_id="ext-model",
        context_window_tokens=8192,
        supports_vision=True,
    )
    wrapped = CatalogCapabilityAdapter(inner, record)
    assert wrapped.supports_vision() is True
    assert wrapped.generate_messages([ChatMessage(role="user", content="x")]).content == "external"


def test_ebh_2e_r6_enrich_catalog_with_structural_adapter() -> None:
    inner = ExternalStructuralAdapter()
    inner.model = "gemini-2.0-flash"
    enriched = enrich_adapter_with_catalog_capabilities(
        inner,
        provider="external-structural",
        model="gemini-2.0-flash",
    )
    assert isinstance(enriched, CatalogCapabilityAdapter)


def test_ebh_2e_r6_failover_source_no_hasattr_provider_slug() -> None:
    text = _FAILOVER_SOURCE.read_text(encoding="utf-8")
    assert "hasattr" not in text


def test_ebh_2e_r6_catalog_source_no_attribute_access_probing() -> None:
    text = _CATALOG_SOURCE.read_text(encoding="utf-8")
    assert "attribute_access" not in text


def test_ebh_2e_r6_external_structural_satisfies_llm_adapter_protocol() -> None:
    assert_external_structural_llm_adapter(ExternalStructuralAdapter())
