# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.llm_adapters.providers.vllm_diagnostics import VllmMetricDeltas, VllmMetricsSnapshot
from intergrax.runtime.token_optimization.vllm_prefix_cache_proof import (
    VllmPrefixCacheProofCaseId,
    VllmPrefixCacheProofCaseObservation,
    VllmPrefixCacheProofReasonCode,
    assemble_proof_case,
    evaluate_vllm_prefix_cache_proof,
    materialize_proof_send_payload,
    vllm_prefix_cache_proof_result_to_safe_dict,
)


def _metrics() -> VllmMetricsSnapshot:
    return VllmMetricsSnapshot(
        prefix_cache_queries=10.0,
        prefix_cache_hits=4.0,
        prompt_tokens_cached=100.0,
        kv_cache_usage_perc=0.5,
    )


def _observation(
    case_id: VllmPrefixCacheProofCaseId,
    *,
    prefix_hash: str,
    cached_input_tokens: int,
    hit_delta: float,
    details_reported: bool = True,
) -> VllmPrefixCacheProofCaseObservation:
    return VllmPrefixCacheProofCaseObservation(
        case_id=case_id,
        prefix_hash=prefix_hash,
        tool_envelope_hash="tool-hash",
        input_tokens=500,
        cached_input_tokens=cached_input_tokens,
        uncached_input_tokens=500 - cached_input_tokens,
        latency_ms=120.0,
        prompt_tokens_details_reported=details_reported,
        metric_deltas=VllmMetricDeltas(
            prefix_cache_queries=1.0,
            prefix_cache_hits=hit_delta,
            prompt_tokens_cached=float(cached_input_tokens),
            kv_cache_usage_perc=0.01,
        ),
    )


def test_evaluator_success_for_cold_warm_changed_prefix() -> None:
    result = evaluate_vllm_prefix_cache_proof(
        health_ok=True,
        server_version="0.23.0",
        expected_server_version="0.23.0",
        metrics_available=True,
        cases=(
            _observation(VllmPrefixCacheProofCaseId.COLD, prefix_hash="a", cached_input_tokens=0, hit_delta=0.0),
            _observation(VllmPrefixCacheProofCaseId.WARM, prefix_hash="a", cached_input_tokens=300, hit_delta=2.0),
            _observation(
                VllmPrefixCacheProofCaseId.CHANGED_PREFIX,
                prefix_hash="b",
                cached_input_tokens=50,
                hit_delta=0.5,
            ),
        ),
    )
    assert result.passed is True
    assert result.reason_codes == ()


def test_unchanged_prefix_requirement_failure() -> None:
    result = evaluate_vllm_prefix_cache_proof(
        health_ok=True,
        server_version="0.23.0",
        expected_server_version="0.23.0",
        metrics_available=True,
        cases=(
            _observation(VllmPrefixCacheProofCaseId.COLD, prefix_hash="a", cached_input_tokens=0, hit_delta=0.0),
            _observation(VllmPrefixCacheProofCaseId.WARM, prefix_hash="b", cached_input_tokens=200, hit_delta=2.0),
            _observation(
                VllmPrefixCacheProofCaseId.CHANGED_PREFIX,
                prefix_hash="c",
                cached_input_tokens=10,
                hit_delta=0.1,
            ),
        ),
    )
    assert result.passed is False
    assert VllmPrefixCacheProofReasonCode.PREFIX_HASH_MISMATCH.value in result.reason_codes


def test_changed_prefix_negative_control_failure() -> None:
    result = evaluate_vllm_prefix_cache_proof(
        health_ok=True,
        server_version="0.23.0",
        expected_server_version="0.23.0",
        metrics_available=True,
        cases=(
            _observation(VllmPrefixCacheProofCaseId.COLD, prefix_hash="a", cached_input_tokens=0, hit_delta=0.0),
            _observation(VllmPrefixCacheProofCaseId.WARM, prefix_hash="a", cached_input_tokens=250, hit_delta=2.0),
            _observation(
                VllmPrefixCacheProofCaseId.CHANGED_PREFIX,
                prefix_hash="b",
                cached_input_tokens=300,
                hit_delta=1.0,
            ),
        ),
    )
    assert result.passed is False
    assert (
        VllmPrefixCacheProofReasonCode.CHANGED_PREFIX_REUSE_NOT_LOWER_THAN_WARM.value
        in result.reason_codes
    )


def test_safe_output_excludes_raw_prompt_and_tool_schema() -> None:
    assembly = assemble_proof_case(
        case_id=VllmPrefixCacheProofCaseId.COLD,
        prefix_variant="safe-output",
        dynamic_tail_text="tail",
        minimum_prefix_chars=1024,
    )
    payload = materialize_proof_send_payload(assembly)
    assert payload.messages
    assert payload.tools_schema

    safe = vllm_prefix_cache_proof_result_to_safe_dict(
        evaluate_vllm_prefix_cache_proof(
            health_ok=True,
            server_version="0.23.0",
            expected_server_version="0.23.0",
            metrics_available=True,
            cases=(
                VllmPrefixCacheProofCaseObservation(
                    case_id=VllmPrefixCacheProofCaseId.COLD,
                    prefix_hash=assembly.state.prefix_hash,
                    tool_envelope_hash=assembly.state.tool_envelope_hash,
                    input_tokens=100,
                    cached_input_tokens=0,
                    uncached_input_tokens=100,
                    latency_ms=10.0,
                    prompt_tokens_details_reported=True,
                    metric_deltas=_metrics().metric_delta(_metrics()),
                ),
                _observation(VllmPrefixCacheProofCaseId.WARM, prefix_hash=assembly.state.prefix_hash, cached_input_tokens=80, hit_delta=1.0),
                _observation(
                    VllmPrefixCacheProofCaseId.CHANGED_PREFIX,
                    prefix_hash="different",
                    cached_input_tokens=5,
                    hit_delta=0.1,
                ),
            ),
        )
    )
    dumped = str(safe)
    assert "Synthetic cache-stable qualification prefix" not in dumped
    assert "token_optimization_proof_echo" not in dumped
    assert "parameters" not in dumped
