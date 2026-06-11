# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.checkpoint_wiring import (
    attach_checkpoint_wiring,
    inject_acp_checkpoint_metadata,
    should_resume_acp_checkpoint,
)
from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore, build_checkpoint
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey


@pytest.mark.unit
@pytest.mark.gate
def test_inject_acp_checkpoint_metadata_when_session_enabled() -> None:
    store = InMemoryAgentCheckpointStore()
    metadata: dict[str, object] = {AcpMetadataKey.SESSION_ENABLED: True}
    inject_acp_checkpoint_metadata(
        metadata,
        store=store,
        run_id="run-1",
        tenant_id="tenant-a",
    )
    assert metadata[AcpMetadataKey.CHECKPOINT_STORE] is store


@pytest.mark.unit
@pytest.mark.gate
def test_should_resume_when_checkpoint_exists() -> None:
    store = InMemoryAgentCheckpointStore()
    store.save(
        build_checkpoint(
            run_id="run-2",
            tenant_id="tenant-a",
            agent_id="probe",
            step_index=0,
            state_root={"acp.state.v1": {"_version": 1}},
            side_effect_ledger=[],
            trace_step_count=1,
        )
    )
    assert should_resume_acp_checkpoint(
        {},
        store=store,
        run_id="run-2",
        tenant_id="tenant-a",
    )


@pytest.mark.unit
@pytest.mark.gate
def test_attach_checkpoint_wiring_sets_resume_flag() -> None:
    store = InMemoryAgentCheckpointStore()
    wired = attach_checkpoint_wiring({}, store, resume=True)
    assert wired[AcpMetadataKey.RESUME_FROM_CHECKPOINT] is True
