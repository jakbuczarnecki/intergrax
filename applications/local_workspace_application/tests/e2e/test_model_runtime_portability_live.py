# © Artur Czarnecki. All rights reserved.

"""Opt-in live proof for LKW Ollama / vLLM model runtime portability."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.no_ci,
]

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_FLAG = "INTERGRAX_LKW_MODEL_RUNTIME_PROOF"


def _enabled() -> bool:
    return os.environ.get(_FLAG, "").strip() == "1"


def _load_env() -> None:
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)


def _require_config() -> None:
    _load_env()
    missing = []
    for name in (
        "LKW_MODEL_RUNTIME_PROOF_OLLAMA_MODEL",
        "LKW_MODEL_RUNTIME_PROOF_VLLM_MODEL",
    ):
        if not (os.environ.get(name) or os.environ.get("INTERGRAX_LLM_MODEL")):
            missing.append(name)
    if missing and not os.environ.get("INTERGRAX_LLM_MODEL"):
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
