# © Artur Czarnecki. All rights reserved.

"""Bridge RuntimeRequest ↔ AgentRunRequest (ACP-DX-4)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import (
    AgentRunRequest,
    AgentRunResult,
    RequestIdentity,
)
from intergrax.contracts.agent_run_enums import (
    AgentRunStatus,
    PrincipalType,
    TerminalReason,
)
from intergrax.contracts.memory_scope import MemoryScope
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest


def runtime_request_to_agent_run(
    request: RuntimeRequest,
    *,
    contract: AgentContract,
) -> AgentRunRequest:
    """Map Nexus ``RuntimeRequest`` into typed ``AgentRunRequest``."""
    tenant_id = str(request.tenant_id or request.metadata.get("tenant_id") or "default")
    user_id = request.metadata.get("user_id")
    user_id_str = str(user_id) if user_id else None
    principal_raw = str(request.metadata.get("principal_type") or "user")
    try:
        principal_type = PrincipalType(principal_raw)
    except ValueError:
        principal_type = PrincipalType.USER

    input_payload: str | dict[str, Any]
    if request.message:
        input_payload = request.message
    else:
        input_payload = {"metadata": dict(request.metadata)}

    return AgentRunRequest(
        input=input_payload,
        identity=RequestIdentity(
            tenant_id=tenant_id,
            user_id=user_id_str,
            principal_type=principal_type,
            auth_subject=str(request.metadata.get("auth_subject") or "") or None,
        ),
        session_id=str(request.session_id or "") or None,
        correlation_id=str(request.metadata.get("correlation_id") or "") or None,
        agent_id=contract.id,
        metadata=dict(request.metadata),
        state=(
            dict(request.metadata["acp.state.v1"])
            if isinstance(request.metadata.get("acp.state.v1"), dict)
            else None
        ),
    )


def agent_run_result_to_runtime_answer(result: AgentRunResult) -> RuntimeAnswer:
    """Map ``AgentRunResult`` to legacy ``RuntimeAnswer`` for Nexus parity."""
    if isinstance(result.output, str):
        answer = result.output
    else:
        answer = str(result.output.get("summary", result.output))
    run_id = result.run_id or f"run_{uuid4().hex}"
    return RuntimeAnswer(
        run_id=run_id,
        answer=answer,
        route=None,
    )


def acp_session_enabled(request: RuntimeRequest) -> bool:
    """Whether Nexus should route through the typed ACP session loop."""
    flag = request.metadata.get("acp.session.v1")
    return flag is True or flag == "true" or flag == "1"
