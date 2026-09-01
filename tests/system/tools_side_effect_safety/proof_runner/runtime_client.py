# © Artur Czarnecki. All rights reserved.

"""HTTP client for proof runtime workers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from tests.system.tools_side_effect_safety.shared.contracts import (
    DEFAULT_TENANT,
    InvokeRequest,
    InvokeResponse,
    PROOF_TOOL_BAD_OUTPUT,
    PROOF_TOOL_CHARGE,
    PROOF_TOOL_CHARGE_ALT,
    PROOF_TOOL_FAIL_BEFORE,
    PROOF_TOOL_SLOW_CHARGE,
)


@dataclass(frozen=True, slots=True)
class RuntimeEndpoint:
    name: str
    base_url: str


class RuntimeClient:
    def __init__(self, endpoint: RuntimeEndpoint) -> None:
        self.endpoint = endpoint
        self._base = endpoint.base_url.rstrip("/")

    def wait_healthy(self, timeout_s: float = 180.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=3.0) as client:
                    response = client.get(f"{self._base}/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"runtime {self.endpoint.name} not healthy at {self._base}")

    def invoke(self, request: InvokeRequest) -> InvokeResponse:
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                f"{self._base}/invoke",
                json=request.model_dump(),
            )
            if response.status_code >= 400:
                return InvokeResponse(
                    success=False,
                    error_type=f"HTTPStatusError:{response.status_code}",
                )
            return InvokeResponse.model_validate(response.json())

    def ledger_status(self, *, tenant_id: str = DEFAULT_TENANT, key: str) -> str | None:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self._base}/ledger/{tenant_id}/{key}")
            response.raise_for_status()
            return response.json().get("status")


def build_invoke(
    *,
    run_id: str,
    business_operation_id: str,
    idempotency_key: str,
    tool_id: str = PROOF_TOOL_CHARGE,
    proof_mode: str = "normal",
    proof_delay_ms: int = 0,
    governance_action: str | None = None,
    require_hitl: bool = False,
    hitl_resume: bool = False,
) -> InvokeRequest:
    return InvokeRequest(
        run_id=run_id,
        tool_id=tool_id,
        business_operation_id=business_operation_id,
        idempotency_key=idempotency_key,
        proof_mode=proof_mode,
        proof_delay_ms=proof_delay_ms,
        governance_action=governance_action,
        require_hitl=require_hitl,
        hitl_resume=hitl_resume,
        worker_source=None,
    )
