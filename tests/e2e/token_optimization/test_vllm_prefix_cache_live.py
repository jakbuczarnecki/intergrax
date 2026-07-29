# © Artur Czarnecki. All rights reserved.

"""Gated live vLLM prefix-cache proof for Token Optimization (TOKEN-10C)."""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from intergrax.llm_adapters.providers.openai_compat_providers import VllmChatAdapter
from intergrax.llm_adapters.providers.vllm_diagnostics import (
    VLLM_PINNED_VERSION,
    collect_vllm_diagnostics,
    derive_vllm_server_root,
    fetch_vllm_metrics,
)
from intergrax.runtime.token_optimization.vllm_prefix_cache_proof import (
    VllmPrefixCacheProofCaseId,
    VllmPrefixCacheProofCaseObservation,
    assemble_proof_case,
    evaluate_vllm_prefix_cache_proof,
    materialize_proof_send_payload,
    vllm_prefix_cache_proof_result_to_safe_dict,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_E2E_FLAG = "INTERGRAX_TOKEN_OPTIMIZATION_VLLM_E2E"
_BASE_URL_ENV = "INTERGRAX_DEFAULT_VLLM_BASE_URL"
_MODEL_ENV = "INTERGRAX_DEFAULT_VLLM_MODEL"
_CONNECT_TIMEOUT_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_VLLM_CONNECT_TIMEOUT"
_READ_TIMEOUT_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_VLLM_READ_TIMEOUT"
_MIN_PREFIX_CHARS_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_VLLM_MIN_PREFIX_CHARS"
_REPORT_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_VLLM_E2E_REPORT"


def _enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _base_url() -> str:
    value = os.environ.get(_BASE_URL_ENV, "http://127.0.0.1:8100/v1").strip()
    if not value:
        pytest.fail(f"{_BASE_URL_ENV} must be set when {_E2E_FLAG}=1")
    return value


def _model() -> str:
    value = os.environ.get(_MODEL_ENV, "Qwen/Qwen2.5-7B-Instruct").strip()
    if not value:
        pytest.fail(f"{_MODEL_ENV} must be set when {_E2E_FLAG}=1")
    return value


def _connect_timeout() -> float:
    return float(os.environ.get(_CONNECT_TIMEOUT_ENV, "5"))


def _read_timeout() -> float:
    return float(os.environ.get(_READ_TIMEOUT_ENV, "120"))


def _minimum_prefix_chars() -> int:
    return max(512, int(os.environ.get(_MIN_PREFIX_CHARS_ENV, "4096")))


@pytest.mark.skipif(not _enabled(), reason=f"Set {_E2E_FLAG}=1 to run live vLLM prefix-cache proof")
def test_vllm_prefix_cache_live_proof() -> None:
    base_url = _base_url()
    model = _model()
    connect_timeout = _connect_timeout()
    read_timeout = _read_timeout()
    minimum_prefix_chars = _minimum_prefix_chars()

    diagnostics = collect_vllm_diagnostics(
        base_url,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    server_root = derive_vllm_server_root(base_url)
    client = httpx.Client(base_url=server_root)
    adapter = VllmChatAdapter(base_url=base_url, model=model)

    observations: list[VllmPrefixCacheProofCaseObservation] = []
    previous_state = None

    case_specs = (
        (VllmPrefixCacheProofCaseId.COLD, "proof-a", "dynamic tail one for cold proof"),
        (VllmPrefixCacheProofCaseId.WARM, "proof-a", "dynamic tail two for warm proof"),
        (
            VllmPrefixCacheProofCaseId.CHANGED_PREFIX,
            "proof-b",
            "dynamic tail three for changed-prefix proof",
        ),
    )

    try:
        for case_id, prefix_variant, tail_text in case_specs:
            assembly = assemble_proof_case(
                case_id=case_id,
                prefix_variant=prefix_variant,
                dynamic_tail_text=tail_text,
                minimum_prefix_chars=minimum_prefix_chars,
                previous_state=previous_state,
            )
            payload = materialize_proof_send_payload(assembly)
            metrics_before_case = fetch_vllm_metrics(
                client,
                server_root=server_root,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            started = time.perf_counter()
            if payload.tools_schema:
                response = adapter.generate_with_tools(
                    payload.messages,
                    list(payload.tools_schema),
                    max_tokens=64,
                    run_id=f"token-10c-{case_id.value.lower()}",
                )
            else:
                response = adapter.generate_messages(
                    payload.messages,
                    max_tokens=64,
                    run_id=f"token-10c-{case_id.value.lower()}",
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            metrics_after_case = fetch_vllm_metrics(
                client,
                server_root=server_root,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            usage = response.usage
            assert usage is not None
            extensions = response.provider_extensions
            prompt_details_reported = bool(
                extensions is not None
                and extensions.vllm is not None
                and extensions.vllm.prompt_tokens_details_reported
            )
            observations.append(
                VllmPrefixCacheProofCaseObservation(
                    case_id=case_id,
                    prefix_hash=assembly.state.prefix_hash,
                    tool_envelope_hash=assembly.state.tool_envelope_hash,
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    uncached_input_tokens=usage.uncached_input_tokens,
                    latency_ms=latency_ms,
                    prompt_tokens_details_reported=prompt_details_reported,
                    metric_deltas=metrics_after_case.metric_delta(metrics_before_case),
                )
            )
            previous_state = assembly.state
    finally:
        client.close()

    result = evaluate_vllm_prefix_cache_proof(
        health_ok=diagnostics.health.healthy,
        server_version=diagnostics.server_version,
        expected_server_version=VLLM_PINNED_VERSION,
        metrics_available=True,
        cases=observations,
    )
    safe_report = vllm_prefix_cache_proof_result_to_safe_dict(result)
    report_path = os.environ.get(_REPORT_ENV, "").strip()
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(safe_report, handle, indent=2, sort_keys=True)

    assert result.passed, safe_report
