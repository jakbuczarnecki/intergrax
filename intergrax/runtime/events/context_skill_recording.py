# © Artur Czarnecki. All rights reserved.

"""Synchronous runtime event recording helpers (Phase R-Skill.10, R-Context.2)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.nexus.context.context_budget import ContextTrimResult
from intergrax.skills.resolver import ResolvedSkillPack


def record_skill_resolved(
    bus: RuntimeEventBus,
    *,
    agent_id: str,
    pack: ResolvedSkillPack,
    task_id: str = "",
    run_id: str = "",
    correlation_id: str = "",
) -> None:
    bus.record(
        RuntimeEvent(
            task_id=task_id or agent_id,
            run_id=run_id or task_id or agent_id,
            agent_id=agent_id,
            event_type=RuntimeEventType.SKILL_RESOLVED,
            phase=ExecutionPhase.AGENT_SELECTION,
            correlation_id=correlation_id or agent_id,
            payload={
                "skill_ids": list(pack.skill_ids),
                "tool_ids": sorted(pack.tool_ids),
                "prompt_instruction_ids": sorted(pack.prompt_instruction_ids),
                "policy_fragment_ids": sorted(pack.policy_fragment_ids),
                "risk_tier": pack.risk_tier.value,
            },
        )
    )


def record_skill_import_failed(
    bus: RuntimeEventBus,
    *,
    source: str,
    reason: str,
    task_id: str = "",
    run_id: str = "",
    correlation_id: str = "",
) -> None:
    bus.record(
        RuntimeEvent(
            task_id=task_id or "skill-import",
            run_id=run_id or task_id or "skill-import",
            event_type=RuntimeEventType.SKILL_IMPORT_FAILED,
            phase=ExecutionPhase.AGENT_SELECTION,
            correlation_id=correlation_id or source,
            payload={"source": source, "reason": reason},
        )
    )


def record_context_assembly(
    bus: RuntimeEventBus,
    *,
    task_id: str,
    run_id: str,
    node_id: str,
    agent_id: Optional[str],
    trim: ContextTrimResult,
    metadata: Mapping[str, Any],
) -> None:
    base_payload: dict[str, Any] = {
        "node_id": node_id,
        "summary_tier": metadata.get("summary_tier"),
        "context_original_chars": trim.original_chars,
        "context_final_chars": trim.final_chars,
    }
    bus.record(
        RuntimeEvent(
            tenant_id=metadata.get("tenant_id") if isinstance(metadata.get("tenant_id"), str) else None,
            task_id=task_id,
            run_id=run_id,
            node_id=node_id,
            agent_id=agent_id,
            event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
            phase=ExecutionPhase.CONTEXT_BUILDING,
            correlation_id=task_id,
            payload=dict(base_payload),
        )
    )
    if trim.trimmed:
        bus.record(
            RuntimeEvent(
                task_id=task_id,
                run_id=run_id,
                node_id=node_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.CONTEXT_TRIMMED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=task_id,
                payload=dict(base_payload, trimmed=True),
            )
        )
