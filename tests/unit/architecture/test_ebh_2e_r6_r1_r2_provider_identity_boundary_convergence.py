# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R2 — provider identity boundary convergence."""

from __future__ import annotations

import ast
from pathlib import Path
import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider, llm_provider_slug
from intergrax.llm_adapters.registry.catalog_miss_diag import (
    CatalogResolutionTier,
    maybe_emit_catalog_miss,
    reset_catalog_miss_diagnostics,
)
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens
from intergrax.llm_adapters.registry.model_catalog import ModelCatalog
from intergrax.llm_adapters.registry.model_router import ModelRouter
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.registry.secrets import default_secret_path_for_provider
from intergrax.llm_adapters.routing.evaluator import LLMRoutingEvaluator, profile_identity
from intergrax.llm_adapters.routing import LLMRoutingProfile, RoutingContext
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LLM_ADAPTERS = _REPO_ROOT / "intergrax/llm_adapters"
_EVALUATOR_SOURCE = _LLM_ADAPTERS / "routing/evaluator.py"
_IDENTITY_FILES = (
    _LLM_ADAPTERS / "routing/evaluator.py",
    _LLM_ADAPTERS / "registry/context_window.py",
    _LLM_ADAPTERS / "registry/secrets.py",
    _LLM_ADAPTERS / "registry/catalog_miss_diag.py",
    _LLM_ADAPTERS / "base/base_llm_adapter.py",
)

_DUPLICATE_IDENTITY_RETURN = 'return str(provider or "").strip().lower()'


def _source_has_banned_identity_normalization(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if _DUPLICATE_IDENTITY_RETURN in text:
        return [f"{path.relative_to(_REPO_ROOT)}: {_DUPLICATE_IDENTITY_RETURN}"]
    return []


def test_ebh_2e_r6_r1_r2_evaluator_no_profile_provider_value_in_source() -> None:
    tree = ast.parse(_EVALUATOR_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "value":
            if isinstance(node.value, ast.Attribute) and node.value.attr == "provider":
                pytest.fail("LLMRoutingEvaluator must not use profile.provider.value")


def test_ebh_2e_r6_r1_r2_no_duplicate_strip_lower_identity_helpers() -> None:
    violations: list[str] = []
    for path in _IDENTITY_FILES:
        violations.extend(_source_has_banned_identity_normalization(path))
    assert violations == []


def test_ebh_2e_r6_r1_r2_external_provider_routing_chain_identity() -> None:
    profile = LLMProfile(provider="external-provider", model="model-x")
    assert profile_identity(profile) == "external-provider:model-x"
    router = ModelRouter.from_profiles(profile)
    assert router.ordered_profile_ids() == ("external-provider:model-x",)
    routing = LLMRoutingProfile(
        default_profile=profile,
        allowed_profiles=(profile,),
        rules=(),
    )
    evaluation = LLMRoutingEvaluator().evaluate(routing, RoutingContext())
    assert evaluation.selected_profile.provider == "external-provider"


def test_ebh_2e_r6_r1_r2_context_window_external_provider_default_lookup() -> None:
    catalog = ModelCatalog.from_mapping(
        {
            "provider_defaults": {"external-provider": 77_777},
            "fallback_default": 32_000,
        }
    )
    tokens = resolve_context_window_tokens(
        " External-Provider ",
        "unknown-model",
        catalog=catalog,
        run_id="run-ext",
    )
    assert tokens == 77_777


def test_ebh_2e_r6_r1_r2_context_window_rejects_empty_provider() -> None:
    with pytest.raises(ValueError, match="provider must not be empty"):
        resolve_context_window_tokens("", "m", catalog=ModelCatalog.from_mapping({}))


def test_ebh_2e_r6_r1_r2_secrets_external_provider_canonical_path() -> None:
    assert (
        default_secret_path_for_provider(" External-Provider ")
        == "llm/external-provider/api_key"
    )


def test_ebh_2e_r6_r1_r2_catalog_miss_diag_canonical_slug() -> None:
    reset_catalog_miss_diagnostics()
    diag = maybe_emit_catalog_miss(
        " External-Provider ",
        "m1",
        1000,
        resolution_tier=CatalogResolutionTier.PROVIDER_DEFAULT,
        run_id="run-diag",
    )
    assert diag is not None
    assert diag.provider_slug == "external-provider"


class _SlugProbeAdapter(BaseLLMAdapter):
    provider = " External-Provider "
    model = "m"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        return build_adapter_response(content="")


def test_ebh_2e_r6_r1_r2_base_adapter_provider_slug_delegates() -> None:
    adapter = _SlugProbeAdapter()
    assert adapter._provider_slug() == llm_provider_slug(adapter.provider)


def test_ebh_2e_r6_r1_r2_model_router_builtin_regression() -> None:
    profile = LLMProfile(provider=LLMProvider.OPENAI, model="m1")
    decision = ModelRouter.from_profiles(profile).resolve()
    assert decision.provider == "openai"
