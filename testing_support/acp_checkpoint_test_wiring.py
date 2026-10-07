# © Artur Czarnecki. All rights reserved.

"""Test-only helpers mirroring host-composed checkpoint injection (not public API)."""

from __future__ import annotations

from intergrax.agents.authoring.acp_run import run_acp_session
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.checkpoint_wiring import attach_checkpoint_wiring
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest


def wire_acp_run_request(
    request: AgentRunRequest,
    store: AgentCheckpointStore,
    *,
    resume: bool = False,
) -> AgentRunRequest:
    """Return request with resume metadata only (store is not attached to request)."""
    return request.model_copy(
        update={
            "metadata": attach_checkpoint_wiring(
                dict(request.metadata),
                store,
                resume=resume,
            ),
        },
    )


async def run_acp_with_host_checkpoint_store(
    agent: object,
    request: AgentRunRequest,
    store: AgentCheckpointStore,
    *,
    resume: bool = False,
) -> object:
    """Execute ACP with sanctioned host-composed checkpoint store (AgentEngine-equivalent)."""
    wired = wire_acp_run_request(request, store, resume=resume)
    return await run_acp_session(
        agent,
        wired,
        agent_checkpoint_store=store,
    )
