# © Artur Czarnecki. All rights reserved.

"""Wire checkpoint store and ledger into ACP session (ACP-PROD-1 · ACP-PROD-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.agents.persistence.checkpoint_store import (
    AgentCheckpointStore,
    build_checkpoint,
)
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest

@dataclass(frozen=True, slots=True)
class SessionResumeState:
    state_root: dict[str, Any]
    start_step_index: int
    trace_step_count: int


@dataclass(frozen=True, slots=True)
class AgentSessionPersistence:
    checkpoint_store: AgentCheckpointStore | None
    side_effect_ledger: SideEffectLedger
    resume_enabled: bool


def resolve_checkpoint_store(metadata: dict[str, Any]) -> AgentCheckpointStore | None:
    store = metadata.get(AcpMetadataKey.CHECKPOINT_STORE)
    if store is None:
        return None
    if isinstance(store, AgentCheckpointStore):
        return store
    raise TypeError("checkpoint store metadata must be AgentCheckpointStore instance")


def resolve_session_persistence(
    request: AgentRunRequest,
    *,
    run_id: str,
    tenant_id: str,
) -> tuple[AgentSessionPersistence, SessionResumeState | None]:
    metadata = request.metadata
    store = resolve_checkpoint_store(metadata)
    resume_enabled = bool(metadata.get(AcpMetadataKey.RESUME_FROM_CHECKPOINT))
    ledger = SideEffectLedger()

    if store is None or not resume_enabled:
        return AgentSessionPersistence(store, ledger, resume_enabled), None

    checkpoint = store.get_latest(run_id, tenant_id)
    if checkpoint is None:
        return AgentSessionPersistence(store, ledger, resume_enabled), None

    ledger = SideEffectLedger(checkpoint.side_effect_ledger)
    resume = SessionResumeState(
        state_root=dict(checkpoint.state_root),
        start_step_index=checkpoint.step_index + 1,
        trace_step_count=checkpoint.trace_step_count,
    )
    return AgentSessionPersistence(store, ledger, resume_enabled), resume


def make_checkpoint_hook(
    *,
    persistence: AgentSessionPersistence,
    run_id: str,
    tenant_id: str,
    agent_id: str,
    trace_step_count_fn: Any,
) -> Any:
    store = persistence.checkpoint_store
    if store is None:
        return None

    initial_revision: int | None = None
    if store is not None and resume_enabled:
        existing = store.get_latest(run_id, tenant_id)
        if existing is not None:
            initial_revision = existing.revision

    async def _hook(state_root: dict[str, Any], step_index: int) -> None:
        checkpoint = build_checkpoint(
            run_id=run_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            step_index=step_index,
            state_root=state_root,
            side_effect_ledger=persistence.side_effect_ledger.records(),
            trace_step_count=int(trace_step_count_fn()),
        )
        saved = store.save(checkpoint, expected_revision=current_revision)
        current_revision = saved.revision

    current_revision: int | None = initial_revision

    return _hook
