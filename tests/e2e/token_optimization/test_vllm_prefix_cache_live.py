# © Artur Czarnecki. All rights reserved.

"""Gated live vLLM prefix-cache proof for Token Optimization (TOKEN-10C)."""

from __future__ import annotations

import os

import pytest

from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_live import (
    VllmPrefixCacheLiveProofConfig,
    run_vllm_prefix_cache_live_proof,
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
_OUTPUT_DIR_ENV = "INTERGRAX_TOKEN_OPTIMIZATION_VLLM_E2E_OUTPUT_DIR"


def _enabled() -> bool:
    return os.environ.get(_E2E_FLAG, "").strip() == "1"


def _proof_config() -> VllmPrefixCacheLiveProofConfig:
    output_dir = os.environ.get(
        _OUTPUT_DIR_ENV,
        "build/proofs/token_optimization/vllm_prefix_cache",
    ).strip()
    return VllmPrefixCacheLiveProofConfig(
        runs=1,
        output_dir=__import__("pathlib").Path(output_dir),
        base_url=os.environ.get(_BASE_URL_ENV, "http://127.0.0.1:8100/v1").strip(),
        model=os.environ.get(_MODEL_ENV, "Qwen/Qwen2.5-3B-Instruct").strip(),
        minimum_prefix_chars=max(
            512,
            int(os.environ.get(_MIN_PREFIX_CHARS_ENV, "4096")),
        ),
        connect_timeout_seconds=float(os.environ.get(_CONNECT_TIMEOUT_ENV, "5")),
        read_timeout_seconds=float(os.environ.get(_READ_TIMEOUT_ENV, "120")),
        startup_timeout_seconds=1800.0,
        manage_vllm=False,
        force_recreate_vllm=False,
        keep_vllm_running=False,
    )


@pytest.mark.skipif(not _enabled(), reason=f"Set {_E2E_FLAG}=1 to run live vLLM prefix-cache proof")
def test_vllm_prefix_cache_live_proof() -> None:
    result = run_vllm_prefix_cache_live_proof(_proof_config())
    assert result.aggregate.completed_runs == 1
    assert result.aggregate.all_runs_passed is True, list(result.aggregate.reason_codes)
