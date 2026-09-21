# © Artur Czarnecki. All rights reserved.

"""GR-10-A3-R1 — RuntimeCheckpoint canonical UAEP resume; ACP AgentCheckpointStore session-only."""

from __future__ import annotations

import inspect

import pytest

from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.agents.persistence.session_persistence import (
    make_checkpoint_hook,
    resolve_session_persistence,
)
from intergrax.applications._shared.task_control_wiring import build_reliability_task_enricher
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.runtime.execution import host_task as host_task_module
from intergrax.runtime.long_running.checkpoint_builder import apply_runtime_checkpoint_to_task
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint, UaepStepCursor
from intergrax.runtime.nexus.agents import agent_engine as agent_engine_module
from intergrax.runtime.nexus.uaep import uaep_executor as uaep_executor_module
from intergrax.runtime.task.task import Task
from tests.qualification.governance.strategy.catalog import (
    GR10_A2_CHECKPOINT_SESSION_COUPLING,
    GR10_A3_R1_NEXT_REMEDIATION,
    Gr10CheckpointSessionCouplingStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr10_a3_r1_ssot_runtime_checkpoint_authority_for_uaep() -> None:
    assert GR10_A2_CHECKPOINT_SESSION_COUPLING is (
        Gr10CheckpointSessionCouplingStatus.DEPRECATED_MIGRATION_TARGET
    )
    host_source = inspect.getsource(host_task_module)
    assert "apply_runtime_checkpoint_to_task" in host_source
    assert "resume_checkpoint" in host_source
    assert "AgentCheckpointStore" not in host_source

    uaep_source = inspect.getsource(uaep_executor_module)
    assert "RuntimeCheckpoint" in uaep_source
    assert "AgentCheckpointStore" not in uaep_source

    engine_source = inspect.getsource(agent_engine_module)
    assert "AgentCheckpointStore" not in engine_source


def test_gr10_a3_r1_reliability_enricher_does_not_require_agent_checkpoint_store() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    enricher = build_reliability_task_enricher(env, agent_checkpoint_store=InMemoryAgentCheckpointStore())
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="legal",
        message="review",
        metadata={},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is None


def test_gr10_a3_r1_runtime_checkpoint_resume_restores_uaep_cursor() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    root_execution_id = mint_execution_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="legal",
        message="review",
        metadata={},
    )
    runtime = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=ExecutionTreeSnapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=root_execution_id,
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                )
            ],
        ),
        uaep_step_index=2,
        uaep_step_id="step-2",
        uaep_step_cursor=UaepStepCursor(values={"step-0": True, "step-1": True}),
    )
    apply_runtime_checkpoint_to_task(task, runtime)
    restored = task.runtime.orchestration.runtime_checkpoint
    assert restored is not None
    assert restored.uaep_step_index == 2
    assert restored.uaep_step_id == "step-2"
    assert restored.uaep_step_cursor is not None
    assert restored.uaep_step_cursor.values["step-1"] is True


@pytest.mark.asyncio
async def test_gr10_a3_r1_acp_session_checkpoint_save_and_resume() -> None:
    store = InMemoryAgentCheckpointStore()
    run_id = mint_run_id()
    tenant_id = "tenant-a"
    request = AgentRunRequest(
        input="hello",
        identity=RequestIdentity(tenant_id=tenant_id, user_id="user-1"),
        agent_id="probe",
        metadata={
            AcpMetadataKey.SESSION_ENABLED: True,
            AcpMetadataKey.CHECKPOINT_STORE: store,
            AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
        },
    )
    persistence, resume = resolve_session_persistence(
        request,
        run_id=run_id,
        tenant_id=tenant_id,
    )
    assert resume is None

    hook = make_checkpoint_hook(
        persistence=persistence,
        run_id=run_id,
        tenant_id=tenant_id,
        agent_id="probe",
        trace_step_count_fn=lambda: 1,
    )
    assert hook is not None
    state_root = {"acp.state.v1": {"_version": 1, "counter": 1}}
    await hook(state_root, step_index=0)

    loaded = store.get_latest(run_id, tenant_id)
    assert loaded is not None
    assert loaded.step_index == 0
    assert loaded.state_root["acp.state.v1"]["counter"] == 1

    resume_request = request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                AcpMetadataKey.RESUME_FROM_CHECKPOINT: True,
            },
        },
    )
    _persistence, resume_state = resolve_session_persistence(
        resume_request,
        run_id=run_id,
        tenant_id=tenant_id,
    )
    assert resume_state is not None
    assert resume_state.start_step_index == 1
    assert resume_state.state_root["acp.state.v1"]["counter"] == 1


def test_gr10_a3_r1_next_remediation_points_to_gr10_closure() -> None:
    assert "GR-10 AGENTIC Final Recertification" in GR10_A3_R1_NEXT_REMEDIATION.task_name
