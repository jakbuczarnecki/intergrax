# © Artur Czarnecki. All rights reserved.

"""LKW application task enrichment and capability policy (LKW.6A)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from collections.abc import Callable

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.agents.persistence.compensation_queue_wiring import (
    make_acp_compensation_queue_task_enricher,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.agents.persistence.idempotency_store_wiring import (
    make_acp_idempotency_store_task_enricher,
)
from intergrax.agents.persistence.tool_invoker_wiring import attach_declarative_tool_invoker
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
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
        "local.workspace.tool_selection_qualification",
        "local.workspace.web_search_qualification",
        "local.workspace.model_routing_qualification",
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
    declarative_tool_invoker_factory: Callable[[], DeclarativeToolInvoker | None] | None = None,
) -> TaskEnricher:
    """Apply LKW defaults, shared reliability defaults, then orchestration ACP enrichment.

    Direct LKW.1 capabilities keep the UAEP reflex path (no ``acp.session.v1``).
    Orchestration capabilities receive typed ACP session wiring via the HTTP run enricher.
    """
    application_enricher = build_lkw_application_task_enricher(
        env,
        default_capability=default_capability,
    )
    compensation_enricher = make_acp_compensation_queue_task_enricher(compensation_queue_store)
    idempotency_enricher = make_acp_idempotency_store_task_enricher(idempotency_store)
    orchestration_enricher = build_lkw_http_run_task_enricher(
        env,
        agent_checkpoint_store=agent_checkpoint_store,
    )

    def enricher(task: Task) -> Task:
        enriched = application_enricher(task)
        enriched = apply_reliability_task_defaults(enriched, env)
        if declarative_tool_invoker_factory is not None:
            declarative_tool_invoker = declarative_tool_invoker_factory()
            if declarative_tool_invoker is not None:
                enriched = enriched.model_copy(
                    update={
                        "metadata": attach_declarative_tool_invoker(
                            dict(enriched.metadata),
                            declarative_tool_invoker,
                        ),
                    },
                )
        if orchestration_enricher is not None:
            enriched = orchestration_enricher(enriched)
        if compensation_enricher is not None:
            enriched = compensation_enricher(enriched)
        if idempotency_enricher is not None:
            enriched = idempotency_enricher(enriched)
        return enriched

    return enricher
