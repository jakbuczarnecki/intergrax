# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.checkpoint_store import (
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
    build_checkpoint,
)
from intergrax.agents.persistence.idempotency_keys import build_default_idempotency_key
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.agents.persistence.tool_action_validation import (
    ToolActionValidationError,
    validate_requested_actions,
)
from intergrax.contracts.agent_run_enums import SideEffectMode
from intergrax.contracts.side_effect import SideEffectKind, SideEffectStatus
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.tool_execution_profile import build_profile_map, profile_from_tool_contract
from pydantic import BaseModel


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


_MUTATING = ToolContract(
    tool_id="email.send",
    name="email.send",
    description="send email",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.HIGH,
)


@pytest.mark.unit
@pytest.mark.gate
def test_checkpoint_store_roundtrip_in_memory() -> None:
    store = InMemoryAgentCheckpointStore()
    checkpoint = build_checkpoint(
        run_id="run-1",
        tenant_id="tenant-a",
        agent_id="legal",
        step_index=2,
        state_root={"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 3}},
        side_effect_ledger=[],
        trace_step_count=3,
    )
    store.save(checkpoint)
    loaded = store.get_latest("run-1", "tenant-a")
    assert loaded is not None
    assert loaded.step_index == 2
    assert loaded.state_root["acp.state.v1"]["_version"] == 3


@pytest.mark.unit
@pytest.mark.gate
def test_sqlite_checkpoint_store_roundtrip(tmp_path) -> None:
    store = SQLiteAgentCheckpointStore(tmp_path / "agent_ckpt.db")
    checkpoint = build_checkpoint(
        run_id="run-sqlite",
        tenant_id="tenant-a",
        agent_id="legal",
        step_index=1,
        state_root={"acp.state.v1": {"_version": 1}},
        side_effect_ledger=[],
        trace_step_count=2,
    )
    store.save(checkpoint)
    loaded = store.get_latest("run-sqlite", "tenant-a")
    assert loaded is not None
    assert loaded.agent_id == "legal"


@pytest.mark.unit
@pytest.mark.gate
def test_side_effect_ledger_dedupes_committed_keys() -> None:
    ledger = SideEffectLedger()
    key = build_default_idempotency_key(
        run_id="run-1",
        step_index=0,
        kind=SideEffectKind.TOOL,
        target="email.send",
        args={"to": "a@example.com"},
    )
    ledger.register(
        idempotency_key=key,
        run_id="run-1",
        step_index=0,
        target="email.send",
    )
    ledger.commit(key, external_ref="msg-42")
    assert ledger.should_skip_replay(key)
    reloaded = SideEffectLedger(ledger.records())
    assert reloaded.is_committed(key)
    assert reloaded.records()[0].status == SideEffectStatus.COMMITTED


@pytest.mark.unit
@pytest.mark.gate
def test_mutating_tool_requires_idempotency_key() -> None:
    profiles = build_profile_map([_MUTATING])
    with pytest.raises(ToolActionValidationError, match="idempotency_key"):
        validate_requested_actions(
            requested_actions=[{"tool_id": "email.send", "args": {}}],
            side_effect_mode=SideEffectMode.DECLARATIVE,
            tool_profiles=profiles,
            run_id="run-1",
            step_index=0,
            ledger=SideEffectLedger(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_tool_execution_profile_marks_mutating_tools() -> None:
    profile = profile_from_tool_contract(_MUTATING)
    assert profile.requires_idempotency_key is True
    assert profile.requires_approval is True
