# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.context_skill_recording import (
    record_context_assembly,
    record_skill_resolved,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.context.context_budget import ContextTrimResult
from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolver


@pytest.mark.unit
def test_record_skill_resolved_appends_history() -> None:
    bus = RuntimeEventBus(record_history=True)
    skills = SkillRegistry()
    skills.register(SkillManifest(skill_id="demo", description="d", tool_ids=("t1",)))
    pack = SkillResolver(skills).resolve(["demo"])
    record_skill_resolved(bus, agent_id="agent_x", pack=pack)
    assert bus.history[-1].event_type == RuntimeEventType.SKILL_RESOLVED
    assert bus.history[-1].payload["skill_ids"] == ["demo"]


@pytest.mark.unit
def test_record_context_assembly_and_trim() -> None:
    bus = RuntimeEventBus(record_history=True)
    trim = ContextTrimResult(message="x", trimmed=True, original_chars=100, final_chars=10)
    record_context_assembly(
        bus,
        task_id="t1",
        run_id="r1",
        node_id="n1",
        agent_id="a1",
        trim=trim,
        metadata={},
    )
    types = [event.event_type for event in bus.history]
    assert RuntimeEventType.CONTEXT_ASSEMBLED in types
    assert RuntimeEventType.CONTEXT_TRIMMED in types
