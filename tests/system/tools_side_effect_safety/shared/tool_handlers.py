# © Artur Czarnecki. All rights reserved.

"""HTTP side-effect tool handlers for Docker proof."""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest
from tests.system.tools_side_effect_safety.shared.contracts import (
    ChargeInput,
    ChargeOutput,
)


class ChargeHandler:
    def __init__(self, *, effect_service_url: str, worker_source: str) -> None:
        self._effect_service_url = effect_service_url.rstrip("/")
        self._worker_source = worker_source

    def execute(self, request: ToolExecutionRequest[ChargeInput]) -> ChargeOutput:
        inp = request.input
        mode = inp.proof_mode
        delay_ms = inp.proof_delay_ms
        headers = {
            "X-Proof-Mode": str(mode),
            "X-Proof-Delay-Ms": str(delay_ms),
            "X-Worker-Source": self._worker_source,
        }
        payload = {
            "business_operation_id": inp.business_operation_id,
            "amount": inp.amount,
            "worker_source": self._worker_source,
        }
        timeout_s = max(inp.http_timeout_s, 1.0)
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                f"{self._effect_service_url}/charge",
                json=payload,
                headers=headers,
            )
        if response.status_code >= 400:
            response.raise_for_status()
        data = response.json()
        if mode == "bad_output_after_commit":
            raise ValueError("output validation failure after external commit")
        return ChargeOutput.model_validate(data)


class FailBeforeHandler:
    def execute(self, request: ToolExecutionRequest[ChargeInput]) -> ChargeOutput:
        del request
        raise RuntimeError("executor failure before external effect")


class BadOutputHandler(ChargeHandler):
    def execute(self, request: ToolExecutionRequest[ChargeInput]) -> BaseModel:
        patched_input = request.input.model_copy(update={"proof_mode": "bad_output_after_commit"})
        patched_request = ToolExecutionRequest(
            run_id=request.run_id,
            step_id=request.step_id,
            tool_id=request.tool_id,
            input=patched_input,
            idempotency_key=request.idempotency_key,
            declarative_hitl_invocation_scope_id=request.declarative_hitl_invocation_scope_id,
        )
        super().execute(patched_request)
        return ChargeInput(business_operation_id="invalid", amount=-1)


def resolve_effect_service_url() -> str:
    return os.environ.get("EFFECT_SERVICE_URL", "http://external-effect-service:8080")
