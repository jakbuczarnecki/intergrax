# © Artur Czarnecki. All rights reserved.

"""HTTP side-effect tool handlers for Docker proof."""

from __future__ import annotations

import contextvars
import os

import httpx
from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest
from tests.system.tools_side_effect_safety.shared.contracts import (
    ChargeInput,
    ChargeOutput,
)

proof_mode_var: contextvars.ContextVar[str] = contextvars.ContextVar("proof_mode", default="normal")
proof_delay_var: contextvars.ContextVar[int] = contextvars.ContextVar("proof_delay_ms", default=0)
http_timeout_var: contextvars.ContextVar[float] = contextvars.ContextVar("http_timeout_s", default=120.0)


class ChargeHandler:
    def __init__(self, *, effect_service_url: str, worker_source: str) -> None:
        self._effect_service_url = effect_service_url.rstrip("/")
        self._worker_source = worker_source

    def execute(self, request: ToolExecutionRequest[ChargeInput]) -> ChargeOutput:
        mode = proof_mode_var.get()
        delay_ms = proof_delay_var.get()
        headers = {
            "X-Proof-Mode": str(mode),
            "X-Proof-Delay-Ms": str(delay_ms),
            "X-Worker-Source": self._worker_source,
        }
        payload = {
            "business_operation_id": request.input.business_operation_id,
            "amount": request.input.amount,
            "worker_source": self._worker_source,
        }
        timeout_s = max(http_timeout_var.get(), 1.0)
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
        token = proof_mode_var.set("bad_output_after_commit")
        try:
            super().execute(request)
            return ChargeInput(business_operation_id="invalid", amount=-1)
        finally:
            proof_mode_var.reset(token)


def resolve_effect_service_url() -> str:
    return os.environ.get("EFFECT_SERVICE_URL", "http://external-effect-service:8080")
