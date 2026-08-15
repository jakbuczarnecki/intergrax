# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.events.context_skill_recording import (
    record_context_assembly,
    record_context_candidate_collected,
    record_context_candidate_dropped,
    record_context_validation_failed,
    record_skill_resolved,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.context.context_budget import ContextTrimResult
from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolver


@pytest.fixture
def execution_identity():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        yield run_id, mint_task_id(), attempt_id
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
def test_record_skill_resolved_appends_history(execution_identity) -> None:
    run_id, task_id, _ = execution_identity
    bus = RuntimeEventBus(record_history=True)
    skills = SkillRegistry()
    skills.register(SkillManifest(skill_id="demo", description="d", tool_ids=("t1",)))
    pack = SkillResolver(skills).resolve(["demo"])
    record_skill_resolved(
        bus,
        agent_id="agent_x",
        pack=pack,
        task_id=task_id,
        run_id=run_id,
    )
    assert bus.history[-1].event_type == RuntimeEventType.SKILL_RESOLVED
    assert bus.history[-1].payload["data"]["skill_ids"] == ["demo"]
    assert bus.history[-1].task_id == task_id
    assert bus.history[-1].run_id == run_id


@pytest.mark.unit
def test_record_skill_resolved_requires_active_execution_identity() -> None:
    bus = RuntimeEventBus(record_history=True)
    skills = SkillRegistry()
    skills.register(
        SkillManifest(
            skill_id="demo",
            description="d",
            tool_ids=("t1",),
            risk_tier=SkillRiskTier.LOW,
        )
    )
    pack = SkillResolver(skills).resolve(["demo"])
    with pytest.raises(RuntimeError, match="active execution identity required"):
        record_skill_resolved(bus, agent_id="agent_x", pack=pack)


@pytest.mark.unit
def test_record_context_assembly_and_trim(execution_identity) -> None:
    run_id, task_id, _ = execution_identity
    bus = RuntimeEventBus(record_history=True)
    trim = ContextTrimResult(message="x", trimmed=True, original_chars=100, final_chars=10)
    record_context_assembly(
        bus,
        task_id=task_id,
        run_id=run_id,
        node_id="n1",
        agent_id="a1",
        trim=trim,
        metadata={},
    )
    types = [event.event_type for event in bus.history]
    assert RuntimeEventType.CONTEXT_ASSEMBLED in types
    assert RuntimeEventType.CONTEXT_TRIMMED in types


@pytest.mark.unit
def test_record_context_candidate_events(execution_identity) -> None:
    run_id, task_id, _ = execution_identity
    bus = RuntimeEventBus(record_history=True)
    record_context_candidate_collected(
        bus,
        task_id=task_id,
        run_id=run_id,
        node_id="n1",
        agent_id="a1",
        provider_id="builtin.workspace",
        fragment_count=2,
        engine_id="default",
    )
    record_context_candidate_dropped(
        bus,
        task_id=task_id,
        run_id=run_id,
        provider_id="dup-1",
        drop_reason="duplicate_content_hash",
        engine_id="default",
    )
    record_context_validation_failed(
        bus,
        task_id=task_id,
        run_id=run_id,
        errors=("budget exceeded",),
        stage="assembled_validation",
    )
    types = [event.event_type for event in bus.history]
    assert RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED in types
    assert RuntimeEventType.CONTEXT_CANDIDATE_DROPPED in types
    assert RuntimeEventType.CONTEXT_VALIDATION_FAILED in types
    assert bus.history[0].payload["provider_id"] == "builtin.workspace"
    assert bus.history[1].payload["drop_reason"] == "duplicate_content_hash"
