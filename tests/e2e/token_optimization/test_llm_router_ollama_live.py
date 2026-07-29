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
from intergrax.llm_adapters.registry.catalog_capabilities import unwrap_catalog_capability_adapter
from intergrax.runtime.token_optimization.contracts import TokenOptimizationRequest
from intergrax.runtime.token_optimization.llm_router import (
    TokenOptimizationLLMRouter,
    token_optimization_router_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReason,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterTransport,
)
from tests.fixtures.token_optimization.llm_router_corpus import LLM_ROUTER_CORPUS
from intergrax.utils import attribute_access

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

_INVALID_TOOL_CALL_REASONS = frozenset(
    {
        TokenOptimizationRouterReason.NO_TOOL_CALL,
        TokenOptimizationRouterReason.MULTIPLE_TOOL_CALLS,
        TokenOptimizationRouterReason.UNEXPECTED_TOOL,
        TokenOptimizationRouterReason.INVALID_TOOL_ARGUMENTS,
        TokenOptimizationRouterReason.LLM_ERROR,
    }
)
_INVALID_TOOL_CALL_STATUSES = frozenset(
    {
        TokenOptimizationRouterStatus.INVALID_DECISION,
        TokenOptimizationRouterStatus.LLM_ERROR,
        TokenOptimizationRouterStatus.UNSUPPORTED_ADAPTER,
    }
)
_LOSSY_CONFIGURATION_IDS = frozenset(
    {
        "extractive_only",
        "exact_then_extractive",
        "extractive_then_exact",
    }
)


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
    valid_tool_call: bool = False
    invalid_tool_call: bool = False
    no_tool_call: bool = False
    multiple_tool_call: bool = False
    suitable_configuration: bool = False
    forbidden_configuration_executed: bool = False
    review_correct: bool = False
    execution_correct: bool = False
    pipeline_correct: bool = False
    protected_safe: bool = False
    policy_bypass: bool = False
    duration_ms: float = 0.0


@dataclass
class _ModelSummary:
    model: str
    transport: str
    declared_capabilities: tuple[str, ...]
    capabilities_resolved: bool
    native_tools_supported: bool
    structured_output_supported: bool
    case_count: int = 0
    tool_attempt_count: int = 0
    valid_tool_call_count: int = 0
    invalid_tool_call_count: int = 0
    no_tool_call_count: int = 0
    multiple_tool_call_count: int = 0
    routing_quality_case_count: int = 0
    suitable_configuration_count: int = 0
    execution_correctness_count: int = 0
    pipeline_correctness_count: int = 0
    review_correctness_count: int = 0
    policy_safety_case_count: int = 0
    policy_bypass_count: int = 0
    protected_content_safety_count: int = 0
    forbidden_configuration_execution_count: int = 0
    total_duration_ms: float = 0.0
    results: list[_ModelCaseMetrics] = field(default_factory=list)

    @property
    def average_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return self.total_duration_ms / len(self.results)

    def to_safe_dict(self) -> dict[str, object]:
        suitability = (
            self.suitable_configuration_count / self.routing_quality_case_count
            if self.routing_quality_case_count
            else 0.0
        )
        valid_tool_call_rate = (
            self.valid_tool_call_count / self.tool_attempt_count
            if self.tool_attempt_count
            else 0.0
        )
        return {
            "model": self.model,
            "transport": self.transport,
            "case_count": self.case_count,
            "tool_attempt_count": self.tool_attempt_count,
            "valid_tool_call_count": self.valid_tool_call_count,
            "invalid_tool_call_count": self.invalid_tool_call_count,
            "no_tool_call_count": self.no_tool_call_count,
            "multiple_tool_call_count": self.multiple_tool_call_count,
            "routing_quality_case_count": self.routing_quality_case_count,
            "suitable_configuration_count": self.suitable_configuration_count,
            "routing_suitability": suitability,
            "valid_native_tool_call_rate": valid_tool_call_rate,
            "execution_correctness_count": self.execution_correctness_count,
            "pipeline_correctness_count": self.pipeline_correctness_count,
            "review_correctness_count": self.review_correctness_count,
            "policy_safety_case_count": self.policy_safety_case_count,
            "policy_bypass_count": self.policy_bypass_count,
            "protected_content_safety_count": self.protected_content_safety_count,
            "forbidden_configuration_execution_count": self.forbidden_configuration_execution_count,
            "average_duration_ms": self.average_duration_ms,
            "declared_capabilities": list(self.declared_capabilities),
            "capabilities_resolved": self.capabilities_resolved,
            "native_tools_supported": self.native_tools_supported,
            "structured_output_supported": self.structured_output_supported,
        }


def _summary_transport(
    *,
    capabilities_resolved: bool,
    native_tools_supported: bool,
    structured_output_supported: bool,
) -> str:
    if not capabilities_resolved:
        return TokenOptimizationRouterTransport.UNSUPPORTED.value
    if native_tools_supported:
        return TokenOptimizationRouterTransport.NATIVE_TOOLS.value
    if structured_output_supported:
        return TokenOptimizationRouterTransport.STRUCTURED_OUTPUT.value
    return TokenOptimizationRouterTransport.UNSUPPORTED.value


def _read_concrete_model_capabilities(adapter) -> tuple[bool, tuple[str, ...]]:
    concrete = unwrap_catalog_capability_adapter(adapter)
    caps = attribute_access.optional(concrete, "model_capabilities", None)
    if caps is None:
        return False, ()
    resolved = attribute_access.optional(caps, "resolved", None)
    if resolved is not True:
        return False, ()
    capabilities = attribute_access.optional(caps, "capabilities", None)
    if not isinstance(capabilities, frozenset):
        return False, ()
    return True, tuple(sorted(capabilities))


def _is_valid_native_tool_call(
    *,
    result,
    case,
    native_tools_supported: bool,
) -> bool:
    if not case.expected_llm_call:
        return False
    if not native_tools_supported:
        return False
    if result.transport is not TokenOptimizationRouterTransport.NATIVE_TOOLS:
        return False
    if result.reason in _INVALID_TOOL_CALL_REASONS:
        return False
    if result.status in _INVALID_TOOL_CALL_STATUSES:
        return False
    if not result.tool_call_id:
        return False
    if result.configuration_id is None:
        return False
    return True


def _is_policy_bypass(*, result, case) -> bool:
    if not case.policy.enabled:
        return result.executed
    if case.policy.profile.value == "off":
        return result.executed
    if not case.policy.allow_lossy and result.configuration_id is not None:
        if result.configuration_id.value in _LOSSY_CONFIGURATION_IDS and result.executed:
            return True
    return False


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

    if not case.expected_llm_call:
        metrics.execution_correct = (
            result.executed is False
            and result.tool_call_id is None
            and result.transport is TokenOptimizationRouterTransport.UNSUPPORTED
            and result.reason is case.expected_reason
        )
        metrics.pipeline_correct = result.pipeline_result is None
        metrics.review_correct = result.status is not TokenOptimizationRouterStatus.REVIEW_REQUIRED
        metrics.protected_safe = True
        metrics.policy_bypass = _is_policy_bypass(result=result, case=case)
        _ = token_optimization_router_result_to_safe_dict(result)
        return metrics

    if native_tools_supported:
        if result.reason is TokenOptimizationRouterReason.NO_TOOL_CALL:
            metrics.no_tool_call = True
        elif result.reason is TokenOptimizationRouterReason.MULTIPLE_TOOL_CALLS:
            metrics.multiple_tool_call = True
        elif result.reason in {
            TokenOptimizationRouterReason.INVALID_TOOL_ARGUMENTS,
            TokenOptimizationRouterReason.UNEXPECTED_TOOL,
            TokenOptimizationRouterReason.LLM_ERROR,
        }:
            metrics.invalid_tool_call = True

    if _is_valid_native_tool_call(
        result=result,
        case=case,
        native_tools_supported=native_tools_supported,
    ):
        metrics.valid_tool_call = True

    if case.evaluate_suitability and result.configuration_id is not None:
        metrics.suitable_configuration = (
            result.configuration_id in case.acceptable_configuration_ids
        )

    if result.configuration_id in case.forbidden_configuration_ids and result.executed:
        metrics.forbidden_configuration_executed = True

    if case.expected_review:
        metrics.review_correct = (
            result.status is TokenOptimizationRouterStatus.REVIEW_REQUIRED
            and result.executed is False
        )
    else:
        metrics.review_correct = result.status is not TokenOptimizationRouterStatus.REVIEW_REQUIRED

    if case.case_id == "router.lossy_disallowed":
        if result.configuration_id is TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION:
            metrics.execution_correct = not result.executed
            metrics.pipeline_correct = result.pipeline_result is None
        elif result.configuration_id is TokenOptimizationRouterConfigurationId.EXACT_ONLY:
            metrics.execution_correct = result.executed
            metrics.pipeline_correct = (
                result.executed
                and result.pipeline_result is not None
                and bool(result.pipeline_result.receipt_metadata.get("completed", False))
                and len(result.pipeline_result.failed_layer_ids) == 0
            )
        else:
            metrics.execution_correct = False
            metrics.pipeline_correct = False
    elif case.expected_execution:
        metrics.execution_correct = result.executed
        metrics.pipeline_correct = (
            result.executed
            and result.pipeline_result is not None
            and bool(result.pipeline_result.receipt_metadata.get("completed", False))
            and len(result.pipeline_result.failed_layer_ids) == 0
            and not metrics.forbidden_configuration_executed
        )
    else:
        metrics.execution_correct = not result.executed
        metrics.pipeline_correct = result.pipeline_result is None

    if case.case_id == "router.protected_noisy_output":
        metrics.protected_safe = not (
            result.executed
            and result.configuration_id in case.forbidden_configuration_ids
        )
    else:
        metrics.protected_safe = True

    metrics.policy_bypass = _is_policy_bypass(result=result, case=case)

    _ = token_optimization_router_result_to_safe_dict(result)
    return metrics


def _run_model_matrix(model: str) -> _ModelSummary:
    adapter = LLMAdapterRegistry.create(LLMProvider.OLLAMA, model=model)
    capabilities_resolved, declared_capabilities = _read_concrete_model_capabilities(adapter)
    if not capabilities_resolved:
        native_tools = False
        structured = False
    else:
        native_tools = "tools" in frozenset(declared_capabilities)
        structured = bool(adapter.supports_structured_output())
    transport = _summary_transport(
        capabilities_resolved=capabilities_resolved,
        native_tools_supported=native_tools,
        structured_output_supported=structured,
    )
    if not capabilities_resolved and any(case.expected_llm_call for case in LLM_ROUTER_CORPUS):
        pytest.fail(
            f"adapter capabilities unresolved for {model}; "
            "summary transport must remain unsupported"
        )
    summary = _ModelSummary(
        model=model,
        transport=transport,
        declared_capabilities=declared_capabilities,
        capabilities_resolved=capabilities_resolved,
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

            if case.expected_llm_call and native_tools:
                summary.tool_attempt_count += 1
                if metrics.valid_tool_call:
                    summary.valid_tool_call_count += 1
                if metrics.invalid_tool_call:
                    summary.invalid_tool_call_count += 1
                if metrics.no_tool_call:
                    summary.no_tool_call_count += 1
                if metrics.multiple_tool_call:
                    summary.multiple_tool_call_count += 1

            if case.evaluate_suitability:
                summary.routing_quality_case_count += 1
                if metrics.suitable_configuration:
                    summary.suitable_configuration_count += 1

            if not case.expected_llm_call:
                summary.policy_safety_case_count += 1

            if metrics.execution_correct:
                summary.execution_correctness_count += 1
            if metrics.pipeline_correct:
                summary.pipeline_correctness_count += 1
            if metrics.review_correct:
                summary.review_correctness_count += 1
            if metrics.protected_safe:
                summary.protected_content_safety_count += 1
            if metrics.forbidden_configuration_executed:
                summary.forbidden_configuration_execution_count += 1
            if metrics.policy_bypass:
                summary.policy_bypass_count += 1
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

    if summary.native_tools_supported:
        assert summary.tool_attempt_count > 0
        assert summary.valid_tool_call_count == summary.tool_attempt_count

    assert summary.policy_bypass_count == 0
    assert summary.forbidden_configuration_execution_count == 0
    assert summary.protected_content_safety_count == summary.case_count
    assert summary.execution_correctness_count == summary.case_count
    assert summary.pipeline_correctness_count == summary.case_count
    assert summary.review_correctness_count == summary.case_count

    suitability = (
        summary.suitable_configuration_count / summary.routing_quality_case_count
        if summary.routing_quality_case_count
        else 0.0
    )
    if suitability < _min_suitability():
        pytest.fail(
            f"routing suitability {suitability:.2f} below threshold {_min_suitability():.2f}"
        )
