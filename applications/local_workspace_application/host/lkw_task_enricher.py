# © Artur Czarnecki. All rights reserved.

"""LKW application task enrichment and capability policy (LKW.6A)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.applications._shared.task_control_wiring import build_reliability_task_enricher
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers
from intergrax.runtime.task.task import Task
from local_workspace_application.host.run_task_enricher import build_lkw_http_run_task_enricher

TaskEnricher = Callable[[Task], Task]

_LKW_CORE_CAPABILITIES = frozenset(
    {
        "local.workspace.search",
        "local.workspace.index",
        "local.workspace.synthesize",
    }
)


def resolve_lkw_allowed_capabilities(env: ApplicationEnvironmentProfile) -> frozenset[str]:
    allowed = set(_LKW_CORE_CAPABILITIES)
    spec = env.graph_spec
    if spec is not None:
        allowed.update(spec.trigger_capabilities or ())
        allowed.update(
            orchestration_capabilities_from_triggers(spec.trigger_capabilities),
        )
    return frozenset(allowed)


def build_lkw_application_task_enricher(
    env: ApplicationEnvironmentProfile,
    *,
    default_capability: str,
    allowed_capabilities: Iterable[str] | None = None,
) -> TaskEnricher:
    allowed = frozenset(allowed_capabilities or resolve_lkw_allowed_capabilities(env))

    def enricher(task: Task) -> Task:
        capability = (task.context.capability or "").strip()
        if not capability:
            capability = default_capability
        elif capability not in allowed:
            raise ValueError(f"unsupported_lkw_capability:{capability}")
        return task.model_copy(
            update={
                "context": task.context.model_copy(update={"capability": capability}),
            }
        )

    return enricher


def build_lkw_combined_task_enricher(
    env: ApplicationEnvironmentProfile,
    *,
    default_capability: str,
    agent_checkpoint_store: AgentCheckpointStore | None = None,
    compensation_queue_store: CompensationQueueStore | None = None,
    idempotency_store: IdempotencyStore | None = None,
) -> TaskEnricher:
    """Apply LKW defaults, shared reliability enrichment, then orchestration ACP enrichment."""
    application_enricher = build_lkw_application_task_enricher(
        env,
        default_capability=default_capability,
    )
    reliability_enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=agent_checkpoint_store,
        compensation_queue_store=compensation_queue_store,
        idempotency_store=idempotency_store,
    )
    orchestration_enricher = build_lkw_http_run_task_enricher(
        env,
        agent_checkpoint_store=agent_checkpoint_store,
    )

    def enricher(task: Task) -> Task:
        enriched = application_enricher(task)
        enriched = reliability_enricher(enriched)
        if orchestration_enricher is not None:
            enriched = orchestration_enricher(enriched)
        return enriched

    return enricher
