# © Artur Czarnecki. All rights reserved.

"""HTTP /run task enricher — ACP session only for graph orchestration capabilities."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.applications._shared.acp_checkpoint_task_enricher import (
    make_acp_checkpoint_task_enricher,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.nexus.orchestration_capabilities import (
    is_orchestration_capability,
    orchestration_capabilities_from_triggers,
)
from intergrax.runtime.task.task import Task

TaskEnricher = Callable[[Task], Task]


def build_lkw_http_run_task_enricher(
    env: ApplicationEnvironmentProfile,
    *,
    agent_checkpoint_store: AgentCheckpointStore | None,
) -> TaskEnricher | None:
    """Enable typed ACP sessions for graph triggers (e.g. ``local.workspace.pipeline``).

    Direct LKW.1 capabilities keep the UAEP reflex path so Plane B step diagnostics
    remain available on curated HTTP metadata.
    """
    acp_enricher = make_acp_checkpoint_task_enricher(agent_checkpoint_store)
    if acp_enricher is None:
        return None

    spec = env.graph_spec
    triggers = orchestration_capabilities_from_triggers(
        spec.trigger_capabilities if spec is not None else None,
    )
    suffix = spec.pipeline_capability_suffix if spec is not None else ".pipeline"

    def enricher(task: Task) -> Task:
        capability = (task.context.capability or "").strip()
        if not is_orchestration_capability(
            capability,
            trigger_capabilities=triggers,
            pipeline_capability_suffix=suffix,
        ):
            return task
        return acp_enricher(task)

    return enricher
