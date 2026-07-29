# © Artur Czarnecki. All rights reserved.

"""Cold / warm / changed-prefix proof evaluation for vLLM prefix-cache reuse (TOKEN-10C)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.providers.vllm_diagnostics import VllmMetricDeltas
from intergrax.runtime.token_optimization.prompt_assembly import (
    PromptAssemblyMessageBlock,
    assemble_cache_stable_prompt,
    materialize_cache_stable_send_payload,
)


class VllmPrefixCacheProofCaseId(StrEnum):
    COLD = "COLD"
    WARM = "WARM"
    CHANGED_PREFIX = "CHANGED_PREFIX"


class VllmPrefixCacheProofReasonCode(StrEnum):
    PASSED = "PASSED"
    HEALTH_FAILED = "HEALTH_FAILED"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    REQUIRED_METRICS_MISSING = "REQUIRED_METRICS_MISSING"
    PROMPT_TOKENS_DETAILS_MISSING = "PROMPT_TOKENS_DETAILS_MISSING"
    PREFIX_HASH_MISMATCH = "PREFIX_HASH_MISMATCH"
    PREFIX_HASH_NOT_CHANGED = "PREFIX_HASH_NOT_CHANGED"
    WARM_CACHED_TOKENS_NOT_POSITIVE = "WARM_CACHED_TOKENS_NOT_POSITIVE"
    WARM_NOT_GREATER_THAN_COLD = "WARM_NOT_GREATER_THAN_COLD"
    WARM_HIT_DELTA_NOT_GREATER_THAN_COLD = "WARM_HIT_DELTA_NOT_GREATER_THAN_COLD"
    CHANGED_PREFIX_REUSE_NOT_LOWER_THAN_WARM = "CHANGED_PREFIX_REUSE_NOT_LOWER_THAN_WARM"
    INVALID_USAGE_COUNTS = "INVALID_USAGE_COUNTS"
    MISSING_CASE = "MISSING_CASE"


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheProofCaseObservation:
    case_id: VllmPrefixCacheProofCaseId
    prefix_hash: str
    tool_envelope_hash: str | None
    input_tokens: int
    cached_input_tokens: int
    uncached_input_tokens: int | None
    latency_ms: float
    prompt_tokens_details_reported: bool
    metric_deltas: VllmMetricDeltas | None = None


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheProofCaseResult:
    case_id: VllmPrefixCacheProofCaseId
    prefix_hash: str
    tool_envelope_hash: str | None
    input_tokens: int
    cached_input_tokens: int
    uncached_input_tokens: int | None
    latency_ms: float
    prompt_tokens_details_reported: bool
    metric_deltas: VllmMetricDeltas | None
    passed: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheProofResult:
    passed: bool
    reason_codes: tuple[str, ...]
    server_version: str | None
    health_ok: bool
    cases: tuple[VllmPrefixCacheProofCaseResult, ...]


def build_synthetic_stable_prefix_block(
    *,
    block_id: str,
    variant: str,
    minimum_chars: int,
) -> PromptAssemblyMessageBlock:
    if minimum_chars < 512:
        raise ValueError("minimum_chars must be >= 512 for reliable cache blocks")
    unit = (
        f"Synthetic cache-stable qualification prefix variant={variant}. "
        "This content is intentionally non-tenant and non-application specific. "
    )
    repeats = max(1, (minimum_chars + len(unit) - 1) // len(unit))
    content = (unit * repeats)[:minimum_chars]
    return PromptAssemblyMessageBlock(
        block_id=block_id,
        message=ChatMessage(role="system", content=content),
    )


def build_synthetic_proof_tool_schema() -> tuple[Mapping[str, Any], ...]:
    return (
        {
            "type": "function",
            "function": {
                "name": "token_optimization_proof_echo",
                "description": "Synthetic proof-only echo tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string"},
                    },
                    "required": ["note"],
                },
            },
        },
    )


def assemble_proof_case(
    *,
    case_id: VllmPrefixCacheProofCaseId,
    prefix_variant: str,
    dynamic_tail_text: str,
    minimum_prefix_chars: int,
    include_tools: bool = True,
    previous_state=None,
):
    stable_block = build_synthetic_stable_prefix_block(
        block_id=f"proof.prefix.{prefix_variant}",
        variant=prefix_variant,
        minimum_chars=minimum_prefix_chars,
    )
    tools_schema = build_synthetic_proof_tool_schema() if include_tools else ()
    return assemble_cache_stable_prompt(
        stable_prefix_blocks=(stable_block,),
        dynamic_tail=(ChatMessage(role="user", content=dynamic_tail_text),),
        tools_schema=tools_schema,
        previous_state=previous_state,
    )


def vllm_prefix_cache_proof_case_to_safe_dict(
    case: VllmPrefixCacheProofCaseResult,
) -> dict[str, object]:
    metric_deltas = None
    if case.metric_deltas is not None:
        metric_deltas = {
            "prefix_cache_queries": case.metric_deltas.prefix_cache_queries,
            "prefix_cache_hits": case.metric_deltas.prefix_cache_hits,
            "prompt_tokens_cached": case.metric_deltas.prompt_tokens_cached,
            "kv_cache_usage_perc": case.metric_deltas.kv_cache_usage_perc,
        }
    return {
        "case_id": case.case_id.value,
        "prefix_hash": case.prefix_hash,
        "tool_envelope_hash": case.tool_envelope_hash,
        "input_tokens": case.input_tokens,
        "cached_input_tokens": case.cached_input_tokens,
        "uncached_input_tokens": case.uncached_input_tokens,
        "latency_ms": case.latency_ms,
        "prompt_tokens_details_reported": case.prompt_tokens_details_reported,
        "metric_deltas": metric_deltas,
        "passed": case.passed,
        "reason_codes": list(case.reason_codes),
    }


def vllm_prefix_cache_proof_result_to_safe_dict(
    result: VllmPrefixCacheProofResult,
) -> dict[str, object]:
    return {
        "passed": result.passed,
        "reason_codes": list(result.reason_codes),
        "server_version": result.server_version,
        "health_ok": result.health_ok,
        "cases": [vllm_prefix_cache_proof_case_to_safe_dict(case) for case in result.cases],
    }


def _case_by_id(
    cases: Sequence[VllmPrefixCacheProofCaseObservation],
    case_id: VllmPrefixCacheProofCaseId,
) -> VllmPrefixCacheProofCaseObservation | None:
    for case in cases:
        if case.case_id == case_id:
            return case
    return None


def evaluate_vllm_prefix_cache_proof(
    *,
    health_ok: bool,
    server_version: str | None,
    expected_server_version: str,
    metrics_available: bool,
    cases: Sequence[VllmPrefixCacheProofCaseObservation],
) -> VllmPrefixCacheProofResult:
    reason_codes: list[str] = []
    if not health_ok:
        reason_codes.append(VllmPrefixCacheProofReasonCode.HEALTH_FAILED.value)
    if server_version != expected_server_version:
        reason_codes.append(VllmPrefixCacheProofReasonCode.VERSION_MISMATCH.value)
    if not metrics_available:
        reason_codes.append(VllmPrefixCacheProofReasonCode.REQUIRED_METRICS_MISSING.value)

    cold = _case_by_id(cases, VllmPrefixCacheProofCaseId.COLD)
    warm = _case_by_id(cases, VllmPrefixCacheProofCaseId.WARM)
    changed = _case_by_id(cases, VllmPrefixCacheProofCaseId.CHANGED_PREFIX)
    if cold is None or warm is None or changed is None:
        reason_codes.append(VllmPrefixCacheProofReasonCode.MISSING_CASE.value)

    case_results: list[VllmPrefixCacheProofCaseResult] = []
    for observation in cases:
        case_reasons: list[str] = []
        if observation.uncached_input_tokens is None:
            case_reasons.append(VllmPrefixCacheProofReasonCode.INVALID_USAGE_COUNTS.value)
        if not observation.prompt_tokens_details_reported:
            case_reasons.append(
                VllmPrefixCacheProofReasonCode.PROMPT_TOKENS_DETAILS_MISSING.value
            )
        case_results.append(
            VllmPrefixCacheProofCaseResult(
                case_id=observation.case_id,
                prefix_hash=observation.prefix_hash,
                tool_envelope_hash=observation.tool_envelope_hash,
                input_tokens=observation.input_tokens,
                cached_input_tokens=observation.cached_input_tokens,
                uncached_input_tokens=observation.uncached_input_tokens,
                latency_ms=observation.latency_ms,
                prompt_tokens_details_reported=observation.prompt_tokens_details_reported,
                metric_deltas=observation.metric_deltas,
                passed=not case_reasons,
                reason_codes=tuple(case_reasons),
            )
        )

    if cold is not None and warm is not None:
        if cold.prefix_hash != warm.prefix_hash:
            reason_codes.append(VllmPrefixCacheProofReasonCode.PREFIX_HASH_MISMATCH.value)
        if warm.cached_input_tokens <= 0:
            reason_codes.append(
                VllmPrefixCacheProofReasonCode.WARM_CACHED_TOKENS_NOT_POSITIVE.value
            )
        if warm.cached_input_tokens <= cold.cached_input_tokens:
            reason_codes.append(VllmPrefixCacheProofReasonCode.WARM_NOT_GREATER_THAN_COLD.value)
        cold_hits = cold.metric_deltas.prefix_cache_hits if cold.metric_deltas else 0.0
        warm_hits = warm.metric_deltas.prefix_cache_hits if warm.metric_deltas else 0.0
        if warm_hits <= cold_hits:
            reason_codes.append(
                VllmPrefixCacheProofReasonCode.WARM_HIT_DELTA_NOT_GREATER_THAN_COLD.value
            )

    if changed is not None:
        if cold is not None and changed.prefix_hash == cold.prefix_hash:
            reason_codes.append(VllmPrefixCacheProofReasonCode.PREFIX_HASH_NOT_CHANGED.value)
        if warm is not None and changed.cached_input_tokens >= warm.cached_input_tokens:
            reason_codes.append(
                VllmPrefixCacheProofReasonCode.CHANGED_PREFIX_REUSE_NOT_LOWER_THAN_WARM.value
            )

    deduped = tuple(dict.fromkeys(reason_codes))
    return VllmPrefixCacheProofResult(
        passed=not deduped,
        reason_codes=deduped,
        server_version=server_version,
        health_ok=health_ok,
        cases=tuple(case_results),
    )


def materialize_proof_send_payload(assembly):
    """Validate send-time integrity for a proof assembly."""
    return materialize_cache_stable_send_payload(assembly)
