# © Artur Czarnecki. All rights reserved.

"""Synchronous runtime event recording helpers (Phase R-Skill.10, R-Context.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from intergrax.runtime.nexus.context.context_budget import ContextTrimResult

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads import (
    ContextAssemblyPayloadV2,
    ContextCandidatePayloadV1,
    SkillResolvedPayloadV1,
    ValidationPayloadV1,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.skills.resolver import ResolvedSkillPack

if TYPE_CHECKING:
    from intergrax.context.contracts import AssembledContext


def record_skill_resolved(
    bus: RuntimeEventBus,
    *,
    agent_id: str,
    pack: ResolvedSkillPack,
    task_id: str = "",
    run_id: str = "",
    correlation_id: str = "",
) -> None:
    typed = SkillResolvedPayloadV1(
        skill_ids=tuple(pack.skill_ids),
        tool_ids=tuple(sorted(pack.tool_ids)),
        prompt_instruction_ids=tuple(sorted(pack.prompt_instruction_ids)),
        policy_fragment_ids=tuple(sorted(pack.policy_fragment_ids)),
        risk_tier=pack.risk_tier.value,
    )
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                task_id=task_id or agent_id,
                run_id=run_id or task_id or agent_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.SKILL_RESOLVED,
                phase=ExecutionPhase.AGENT_SELECTION,
                correlation_id=correlation_id or agent_id,
            ),
            typed,
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


def record_context_candidate_collected(
    bus: RuntimeEventBus,
    *,
    task_id: str,
    run_id: str,
    node_id: str = "",
    agent_id: Optional[str] = None,
    provider_id: str,
    fragment_count: int,
    engine_id: str = "",
    correlation_id: str = "",
) -> None:
    typed = ContextCandidatePayloadV1(
        provider_id=provider_id,
        fragment_count=fragment_count,
        engine_id=engine_id,
    )
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                task_id=task_id,
                run_id=run_id,
                node_id=node_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=correlation_id or task_id,
            ),
            typed,
            promote_fields={
                "provider_id": provider_id,
                "fragment_count": fragment_count,
                "engine_id": engine_id,
            },
        )
    )


def record_context_candidate_dropped(
    bus: RuntimeEventBus,
    *,
    task_id: str,
    run_id: str,
    node_id: str = "",
    agent_id: Optional[str] = None,
    provider_id: str,
    drop_reason: str,
    engine_id: str = "",
    correlation_id: str = "",
) -> None:
    typed = ContextCandidatePayloadV1(
        provider_id=provider_id,
        fragment_count=1,
        engine_id=engine_id,
        drop_reason=drop_reason,
    )
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                task_id=task_id,
                run_id=run_id,
                node_id=node_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.CONTEXT_CANDIDATE_DROPPED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=correlation_id or task_id,
            ),
            typed,
            promote_fields={
                "provider_id": provider_id,
                "drop_reason": drop_reason,
                "engine_id": engine_id,
            },
        )
    )


def record_context_validation_failed(
    bus: RuntimeEventBus,
    *,
    task_id: str,
    run_id: str,
    node_id: str = "",
    agent_id: Optional[str] = None,
    errors: Sequence[str],
    stage: str = "context_assembly",
    engine_id: str = "",
    correlation_id: str = "",
) -> None:
    typed = ValidationPayloadV1(
        valid=False,
        error_count=len(errors),
        stage=stage,
        rule_ids_failed=tuple(errors),
    )
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                task_id=task_id,
                run_id=run_id,
                node_id=node_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.CONTEXT_VALIDATION_FAILED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=correlation_id or task_id,
            ),
            typed,
            promote_fields={
                "engine_id": engine_id,
                "stage": stage,
                "error_count": len(errors),
            },
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
    engine_id: str = "",
    step_index: int | None = None,
    step_kind: str | None = None,
) -> None:
    base_payload: dict[str, Any] = {
        "node_id": node_id,
        "summary_tier": metadata.get("summary_tier"),
        "context_original_chars": trim.original_chars,
        "context_final_chars": trim.final_chars,
        "engine_id": engine_id,
    }
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                tenant_id=metadata.get("tenant_id") if isinstance(metadata.get("tenant_id"), str) else None,
                task_id=task_id,
                run_id=run_id,
                node_id=node_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=task_id,
            ),
            ContextAssemblyPayloadV2(
                node_id=node_id,
                summary_tier=str(metadata.get("summary_tier"))
                if metadata.get("summary_tier") is not None
                else None,
                context_original_chars=trim.original_chars,
                context_final_chars=trim.final_chars,
                trimmed=False,
                engine_id=engine_id,
                step_index=step_index,
                step_kind=step_kind,
            ),
            promote_fields=base_payload,
        )
    )
    if trim.trimmed:
        bus.record(
            runtime_event_with_payload(
                RuntimeEvent(
                    task_id=task_id,
                    run_id=run_id,
                    node_id=node_id,
                    agent_id=agent_id,
                    event_type=RuntimeEventType.CONTEXT_TRIMMED,
                    phase=ExecutionPhase.CONTEXT_BUILDING,
                    correlation_id=task_id,
                ),
                ContextAssemblyPayloadV2(
                    node_id=node_id,
                    summary_tier=str(metadata.get("summary_tier"))
                    if metadata.get("summary_tier") is not None
                    else None,
                    context_original_chars=trim.original_chars,
                    context_final_chars=trim.final_chars,
                    trimmed=True,
                    engine_id=engine_id,
                    step_index=step_index,
                    step_kind=step_kind,
                ),
                promote_fields={**base_payload, "trimmed": True},
            )
        )


def record_context_assembled_from_engine(
    bus: RuntimeEventBus,
    *,
    assembled: AssembledContext,
    task_id: str,
    run_id: str = "",
    node_id: str = "",
    agent_id: str | None = None,
    engine_id: str = "",
    step_index: int | None = None,
    step_kind: str | None = None,
) -> None:
    """Record CONTEXT_ASSEMBLED with per-fragment cost attribution (CE-MAINT-02)."""
    from intergrax.context.tracking.assembly_cost import assembly_cost_from_assembled

    original_chars = sum(len(fragment.content) for fragment in assembled.fragments_included)
    final_chars = sum(len(msg.content or "") for msg in assembled.messages)
    cost = assembly_cost_from_assembled(assembled)
    base_payload: dict[str, Any] = {
        "node_id": node_id,
        "context_original_chars": original_chars,
        "context_final_chars": final_chars,
        "engine_id": engine_id,
        "fragment_token_cost": cost.fragment_token_cost,
        "estimated_cost_microusd": cost.estimated_cost_microusd,
    }
    bus.record(
        runtime_event_with_payload(
            RuntimeEvent(
                task_id=task_id,
                run_id=run_id or task_id,
                node_id=node_id,
                agent_id=agent_id,
                event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=task_id,
            ),
            ContextAssemblyPayloadV2(
                node_id=node_id,
                context_original_chars=original_chars,
                context_final_chars=final_chars,
                trimmed=assembled.degradation_steps > 0,
                engine_id=engine_id,
                step_index=step_index,
                step_kind=step_kind,
                fragment_token_cost=cost.fragment_token_cost,
                estimated_cost_microusd=cost.estimated_cost_microusd,
            ),
            promote_fields=base_payload,
        )
    )
