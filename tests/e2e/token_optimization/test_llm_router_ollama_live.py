# © Artur Czarnecki. All rights reserved.

"""Gated live Ollama multi-model E2E for Token Optimization LLM router (TOKEN-9)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.providers.ollama_capabilities import OllamaModelCapabilityResolver
from intergrax.runtime.token_optimization.contracts import TokenOptimizationRequest
from intergrax.runtime.token_optimization.llm_router import (
    TokenOptimizationLLMRouter,
    token_optimization_router_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterTransport,
)
from tests.fixtures.token_optimization.llm_router_corpus import LLM_ROUTER_CORPUS

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_TOKEN_OPTIMIZATION_OLLAMA_E2E"
_MODELS_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_OLLAMA_MODELS"
_REPEATS_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_ROUTER_E2E_REPEATS"
_MIN_SUITABILITY_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_ROUTER_E2E_MIN_SUITABILITY"
_REPORT_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_ROUTER_E2E_REPORT"


def _enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _models() -> list[str]:
    raw = os.environ.get(_MODELS_ENV, "").strip()
    if not raw:
        pytest.fail(f"{_MODELS_ENV} is required when {_E2E_FLAG}=1")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _repeats() -> int:
    raw = os.environ.get(_REPEATS_ENV, "3").strip()
    return max(1, int(raw))


def _min_suitability() -> float:
    raw = os.environ.get(_MIN_SUITABILITY_ENV, "0.80").strip()
    return float(raw)


@dataclass
class _ModelCaseMetrics:
    valid_decision: bool = False
    valid_tool_call: bool = False
    invalid_tool_call: bool = False
    no_tool_call: bool = False
    multiple_tool_call: bool = False
    suitable_configuration: bool = False
    forbidden_configuration: bool = False
    review_correct: bool = False
    execution_success: bool = False
    pipeline_correct: bool = False
    protected_safe: bool = False
    fallback_correct: bool = False
    duration_ms: float = 0.0


@dataclass
class _ModelSummary:
    model: str
    transport: str
    declared_capabilities: tuple[str, ...]
    native_tools_supported: bool
    structured_output_supported: bool
    case_count: int = 0
    execution_count: int = 0
    valid_decision_count: int = 0
    valid_tool_call_count: int = 0
    invalid_tool_call_count: int = 0
    no_tool_call_count: int = 0
    multiple_tool_call_count: int = 0
    suitable_configuration_count: int = 0
    forbidden_configuration_count: int = 0
    review_correctness_count: int = 0
    execution_success_count: int = 0
    pipeline_correctness_count: int = 0
    protected_content_safety_count: int = 0
    fallback_correctness_count: int = 0
    total_duration_ms: float = 0.0
    results: list[_ModelCaseMetrics] = field(default_factory=list)

    @property
    def average_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return self.total_duration_ms / len(self.results)

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "transport": self.transport,
            "case_count": self.case_count,
            "execution_count": self.execution_count,
            "valid_decision_count": self.valid_decision_count,
            "valid_tool_call_count": self.valid_tool_call_count,
            "invalid_tool_call_count": self.invalid_tool_call_count,
            "no_tool_call_count": self.no_tool_call_count,
            "multiple_tool_call_count": self.multiple_tool_call_count,
            "suitable_configuration_count": self.suitable_configuration_count,
            "forbidden_configuration_count": self.forbidden_configuration_count,
            "review_correctness_count": self.review_correctness_count,
            "execution_success_count": self.execution_success_count,
            "pipeline_correctness_count": self.pipeline_correctness_count,
            "protected_content_safety_count": self.protected_content_safety_count,
            "fallback_correctness_count": self.fallback_correctness_count,
            "average_duration_ms": self.average_duration_ms,
            "declared_capabilities": list(self.declared_capabilities),
            "native_tools_supported": self.native_tools_supported,
            "structured_output_supported": self.structured_output_supported,
        }


def _evaluate_case(
    *,
    router: TokenOptimizationLLMRouter,
    case,
    request_id: str,
    native_tools_supported: bool,
) -> _ModelCaseMetrics:
    metrics = _ModelCaseMetrics()
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=case.content,
            source_type=case.source_type,
            policy=case.policy,
            protected_regions=case.protected_regions,
            metadata=dict(case.metadata),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id=request_id,
    )
    started = time.perf_counter()
    result = router.route_and_execute(request)
    metrics.duration_ms = (time.perf_counter() - started) * 1000.0

    if result.configuration_id is not None:
        metrics.valid_decision = True
        if result.configuration_id in case.acceptable_configuration_ids:
            metrics.suitable_configuration = True
        if result.configuration_id in case.forbidden_configuration_ids:
            metrics.forbidden_configuration = True

    if result.transport is TokenOptimizationRouterTransport.NATIVE_TOOLS:
        if result.tool_call_id:
            metrics.valid_tool_call = True
        elif result.reason and result.reason.value == "no_tool_call":
            metrics.no_tool_call = True
        elif result.reason and result.reason.value == "multiple_tool_calls":
            metrics.multiple_tool_call = True
        elif result.reason and result.reason.value in {
            "invalid_tool_arguments",
            "unexpected_tool",
        }:
            metrics.invalid_tool_call = True
    else:
        metrics.fallback_correct = not native_tools_supported

    if case.expected_review:
        metrics.review_correct = result.status is TokenOptimizationRouterStatus.REVIEW_REQUIRED
    else:
        metrics.review_correct = result.status is not TokenOptimizationRouterStatus.REVIEW_REQUIRED

    if case.expected_execution:
        metrics.execution_success = result.executed
        metrics.pipeline_correct = (
            result.executed
            and result.pipeline_result is not None
            and len(result.pipeline_result.failed_layer_ids) == 0
        )
    else:
        metrics.execution_success = not result.executed
        metrics.pipeline_correct = not result.executed

    if case.case_id == "router.protected_noisy_output":
        metrics.protected_safe = not (
            result.executed
            and result.configuration_id in case.forbidden_configuration_ids
        )
    else:
        metrics.protected_safe = True

    _ = token_optimization_router_result_to_safe_dict(result)
    return metrics


def _run_model_matrix(model: str) -> _ModelSummary:
    adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA, model=model)
    caps = OllamaModelCapabilityResolver().resolve(model)
    native_tools = adapter.supports_tools()
    structured = adapter.supports_structured_output()
    transport = (
        TokenOptimizationRouterTransport.NATIVE_TOOLS.value
        if native_tools
        else TokenOptimizationRouterTransport.STRUCTURED_OUTPUT.value
    )
    summary = _ModelSummary(
        model=model,
        transport=transport,
        declared_capabilities=tuple(sorted(caps.capabilities)),
        native_tools_supported=native_tools,
        structured_output_supported=structured,
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    repeats = _repeats()
    for repeat in range(repeats):
        for case in LLM_ROUTER_CORPUS:
            metrics = _evaluate_case(
                router=router,
                case=case,
                request_id=f"ollama-{model}-{case.case_id}-{repeat}",
                native_tools_supported=native_tools,
            )
            summary.results.append(metrics)
            summary.case_count += 1
            summary.total_duration_ms += metrics.duration_ms
            if metrics.valid_decision:
                summary.valid_decision_count += 1
            if metrics.valid_tool_call:
                summary.valid_tool_call_count += 1
            if metrics.invalid_tool_call:
                summary.invalid_tool_call_count += 1
            if metrics.no_tool_call:
                summary.no_tool_call_count += 1
            if metrics.multiple_tool_call:
                summary.multiple_tool_call_count += 1
            if metrics.suitable_configuration:
                summary.suitable_configuration_count += 1
            if metrics.forbidden_configuration and metrics.execution_success:
                summary.forbidden_configuration_count += 1
            if metrics.review_correct:
                summary.review_correctness_count += 1
            if metrics.execution_success and case.expected_execution:
                summary.execution_count += 1
                summary.execution_success_count += 1
            if metrics.pipeline_correct:
                summary.pipeline_correctness_count += 1
            if metrics.protected_safe:
                summary.protected_content_safety_count += 1
            if metrics.fallback_correct:
                summary.fallback_correctness_count += 1
    return summary


@pytest.mark.parametrize("model", _models() if _enabled() else ["__skip__"])
def test_live_ollama_router_model_matrix(model: str) -> None:
    if not _enabled():
        pytest.skip(f"{_E2E_FLAG} is not set")
    if model == "__skip__":
        pytest.skip("matrix not configured")

    summary = _run_model_matrix(model)
    safe = summary.to_safe_dict()
    print(json.dumps(safe, ensure_ascii=False, sort_keys=True))

    report_path = os.environ.get(_REPORT_ENV, "").strip()
    if report_path:
        with open(report_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False) + "\n")

    if summary.forbidden_configuration_count > 0:
        pytest.fail("forbidden configuration execution rate must be 0")
    if summary.case_count and (
        summary.protected_content_safety_count < summary.case_count
    ):
        pytest.fail("protected lossy execution detected")

    if summary.native_tools_supported:
        tool_attempts = summary.case_count - summary.fallback_correctness_count
        if tool_attempts > 0:
            valid_rate = summary.valid_tool_call_count / tool_attempts
            if valid_rate < 1.0:
                pytest.fail(f"valid native tool-call rate below 100%: {valid_rate:.2f}")

    suitability = (
        summary.suitable_configuration_count / summary.case_count
        if summary.case_count
        else 0.0
    )
    if suitability < _min_suitability():
        pytest.fail(
            f"routing suitability {suitability:.2f} below threshold {_min_suitability():.2f}"
        )
