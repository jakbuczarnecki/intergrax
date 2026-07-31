# © Artur Czarnecki. All rights reserved.

"""Opt-in live proof for LKW Ollama / vLLM model runtime portability."""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_FLAG = "INTERGRAX_LKW_MODEL_RUNTIME_PROOF"


def _enabled() -> bool:
    return os.environ.get(_FLAG, "").strip() == "1"


def _require_config() -> None:
    missing = []
    for name in (
        "LKW_MODEL_RUNTIME_PROOF_OLLAMA_MODEL",
        "LKW_MODEL_RUNTIME_PROOF_VLLM_MODEL",
        "LKW_MODEL_RUNTIME_PROOF_VLLM_PROVISIONING_CLASSIFICATION",
    ):
        if not os.environ.get(name, "").strip():
            missing.append(name)
    if missing:
        pytest.fail(f"Missing required env when {_FLAG}=1: {', '.join(missing)}")


@pytest.mark.asyncio
async def test_model_runtime_portability_live() -> None:
    if not _enabled():
        pytest.skip(f"{_FLAG} is not set")
    _require_config()

    from local_workspace_application.model_runtime_proof.config import (
        load_proof_config_from_env,
    )
    from local_workspace_application.model_runtime_proof.runner import (
        run_model_runtime_proof,
    )

    config = load_proof_config_from_env()
    result = await run_model_runtime_proof(config)
    assert result.overall_status.value == "PASS", result.model_dump()
