# © Artur Czarnecki. All rights reserved.

"""Gated live E2E for Token Optimization LLM router (TOKEN-9)."""

from __future__ import annotations

import os

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.token_optimization.llm_router import TokenOptimizationLLMRouter
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterTransport,
)
from intergrax.runtime.token_optimization.contracts import TokenOptimizationRequest
from tests.fixtures.token_optimization.llm_router_corpus import LLM_ROUTER_CORPUS

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_TOKEN_OPTIMIZATION_ROUTER_E2E"


def _enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _provider() -> str:
    provider = os.environ.get("INTERGRAX_LLM_PROVIDER", "").strip()
    if not provider:
        pytest.fail(f"INTERGRAX_LLM_PROVIDER is required when {_E2E_FLAG}=1")
    return provider


def _model() -> str:
    model = os.environ.get("INTERGRAX_LLM_MODEL", "").strip()
    if not model:
        pytest.fail(f"INTERGRAX_LLM_MODEL is required when {_E2E_FLAG}=1")
    return model


@pytest.fixture(scope="module")
def live_adapter():
    if not _enabled():
        pytest.skip(f"{_E2E_FLAG} is not set")
    provider = LLMProvider(_provider())
    adapter = LLMAdapterRegistry.create(provider, model=_model())
    return adapter


def test_live_router_runs_real_engine(live_adapter) -> None:
    case = next(item for item in LLM_ROUTER_CORPUS if item.case_id == "router.rag_exact_duplicates")
    router = TokenOptimizationLLMRouter(adapter=live_adapter)
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=case.content,
            source_type=case.source_type,
            policy=case.policy,
            metadata=dict(case.metadata),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="live-generic-router",
    )
    result = router.route_and_execute(request)
    transport = result.transport
    assert transport in {
        TokenOptimizationRouterTransport.NATIVE_TOOLS,
        TokenOptimizationRouterTransport.STRUCTURED_OUTPUT,
    }
    if result.configuration_id in case.forbidden_configuration_ids and result.executed:
        pytest.fail("forbidden configuration executed")
    if result.status is TokenOptimizationRouterStatus.ROUTED:
        assert result.executed is True
        assert result.pipeline_result is not None
