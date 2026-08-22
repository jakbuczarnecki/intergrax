# © Artur Czarnecki. All rights reserved.

"""Bridge RuntimeRequest ↔ AgentRunRequest (ACP-DX-4)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.acp_state import ACP_STATE_KEY
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import (
    AgentExecutionOptions,
    AgentRunRequest,
    AgentRunResult,
    RequestIdentity,
)
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.request_identity_spine import (
    assert_untrusted_metadata_identity_compatible,
)
from intergrax.llm.messages import final_user_message_content, model_input_messages_from_metadata
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest


def runtime_request_to_agent_run(
    request: RuntimeRequest,
    *,
    contract: AgentContract,
) -> AgentRunRequest:
    """Map Nexus ``RuntimeRequest`` into typed ``AgentRunRequest``."""
    if request.canonical_identity is not None:
        identity = request.canonical_identity
        assert_untrusted_metadata_identity_compatible(identity, request.metadata)
    else:
        tenant_id = str(request.tenant_id or request.metadata.get("tenant_id") or "default")
        user_id = request.metadata.get("user_id") or request.user_id
        user_id_str = str(user_id) if user_id else None
        principal_raw = str(request.metadata.get("principal_type") or "user")
        try:
            principal_type = PrincipalType(principal_raw)
        except ValueError:
            principal_type = PrincipalType.USER
        identity = RequestIdentity(
            tenant_id=tenant_id,
            user_id=user_id_str,
            principal_type=principal_type,
            auth_subject=str(request.metadata.get("auth_subject") or "") or None,
        )

    input_payload: str | dict[str, Any]
    model_messages = model_input_messages_from_metadata(request.metadata)
    if model_messages:
        input_payload = final_user_message_content(model_messages)
    elif request.message:
        input_payload = request.message
    else:
        input_payload = {"metadata": dict(request.metadata)}

    execution_options: AgentExecutionOptions | None = None
    raw_options = request.metadata.get("acp.execution_options.v1")
    if isinstance(raw_options, dict):
        execution_options = AgentExecutionOptions.model_validate(raw_options)

    metadata = dict(request.metadata)
    metadata.setdefault("task_id", request.task_id)
    metadata.setdefault("run_id", request.run_id)

    return AgentRunRequest(
        input=input_payload,
        identity=identity,
        session_id=str(request.session_id or "") or None,
        correlation_id=str(request.metadata.get("correlation_id") or request.run_id) or None,
        agent_id=contract.id,
        metadata=metadata,
        state=(
            dict(request.metadata[ACP_STATE_KEY])
            if isinstance(request.metadata.get(ACP_STATE_KEY), dict)
            else None
        ),
        execution_options=execution_options,
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
    flag = request.metadata.get(AcpMetadataKey.SESSION_ENABLED)
    return flag is True or flag == "true" or flag == "1"
