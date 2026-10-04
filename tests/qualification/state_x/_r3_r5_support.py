# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R5 shared qualification helpers (Agent checkpoint persistence parity)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from intergrax.agents.persistence.checkpoint_store import (
    AgentCheckpointStore,
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
    build_checkpoint,
    validate_agent_checkpoint_for_persistence,
)
from intergrax.contracts.side_effect import AgentRunCheckpoint, SideEffectKind, SideEffectRecord

_REPO_ROOT = Path(__file__).resolve().parents[3]


def agent_checkpoint_store_implementations() -> List[Tuple[str, type]]:
    return [
        ("InMemoryAgentCheckpointStore", InMemoryAgentCheckpointStore),
        ("SQLiteAgentCheckpointStore", SQLiteAgentCheckpointStore),
    ]


def build_valid_checkpoint(**overrides: object) -> AgentRunCheckpoint:
    base = build_checkpoint(
        run_id="run-r5",
        tenant_id="tenant-a",
        agent_id="agent-a",
        step_index=0,
        state_root={"acp.state.v1": {"_version": 1}},
        side_effect_ledger=[],
        trace_step_count=1,
    )
    if overrides:
        return base.model_copy(update=overrides)  # type: ignore[arg-type]
    return base


def build_side_effect_record(
    *,
    run_id: str = "run-r5",
    step_index: int = 0,
) -> SideEffectRecord:
    return SideEffectRecord(
        side_effect_id="se-1",
        idempotency_key="idem-1",
        run_id=run_id,
        step_index=step_index,
        kind=SideEffectKind.TOOL,
        target="tool://probe",
    )


def post_construction_invalid_checkpoint(
    valid: AgentRunCheckpoint,
    *,
    ledger: list[SideEffectRecord] | None = None,
) -> AgentRunCheckpoint:
    if ledger is not None:
        return valid.model_copy(update={"side_effect_ledger": ledger})
    return valid.model_copy(update={"step_index": -1})


def parity_checkpoint_store_factories(
    tmp_path: Path,
) -> Iterable[Tuple[str, Callable[[], AgentCheckpointStore]]]:
    db = tmp_path / "r5parity.db"

    def sqlite_factory() -> AgentCheckpointStore:
        return SQLiteAgentCheckpointStore(db)

    def memory_factory() -> AgentCheckpointStore:
        return InMemoryAgentCheckpointStore()

    return (
        ("sqlite", sqlite_factory),
        ("memory", memory_factory),
    )


def save_source_uses_canonical_validator(class_type: type) -> bool:
    source = inspect.getsource(class_type.save)
    return "validate_agent_checkpoint_for_persistence" in source


def canonical_validator_definition_path() -> Path:
    return _REPO_ROOT / "intergrax/agents/persistence/checkpoint_store.py"
