# © Artur Czarnecki. All rights reserved.

"""Initialized scenario runtime spine proofs (testing only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    ScenarioRuntimeComposition,
    execute_scenario_task,
)
from intergrax.applications._shared.scenario_runtime_profiles import (
    build_scenario_lab_runtime,
    cleanup_scenario_runtime_workspace,
)
from intergrax.applications._shared.diagnostic_read_wiring import build_diagnostic_read_service
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeProofResult:
    slug: str
    tenant_id: str
    task_id: str
    run_id: str
    terminal_state: TaskState
    has_runtime_events: bool
    has_terminal_diagnostic_trigger: bool
    reconstruction_complete: bool


def _tenant_for_slug(slug: str) -> str:
    return f"tenant-obs-spine-{slug.replace('_', '-')}"


async def prove_initialized_scenario_runtime(
    slug: str,
    *,
    workspace_root: Path,
) -> ScenarioRuntimeProofResult:
    tenant_id = _tenant_for_slug(slug)
    registry = AgentRegistry()
    registry.register(EchoAgent())
    composition: ScenarioRuntimeComposition = build_scenario_lab_runtime(
        tenant_id=tenant_id,
        scenario_slug=slug,
        workspace_root=workspace_root,
        registry=registry,
    )
    try:
        platform_result = await execute_scenario_task(
            composition,
            ScenarioExecutionRequest(
                tenant_id=tenant_id,
                user_id="obs-spine-proof",
                message=f"OBS universal spine qualification for {slug}",
                capability="echo.basic",
            ),
        )
        store = composition.nexus_loop.runtime_event_store
        if store is None:
            raise AssertionError("scenario runtime must expose runtime_event_store")
        events = store.list_for_run(platform_result.run_id, tenant_id=tenant_id)
        terminal_types = {RuntimeEventType.TASK_COMPLETED, RuntimeEventType.TASK_FAILED}
        assert any(event.event_type in terminal_types for event in events)

        causal = InMemoryCausalEvidencePersistence()
        reconstructor = ExecutionReconstructor(
            runtime_events=store,
            causal_evidence=causal,
        )
        reconstruction = reconstructor.reconstruct_execution(
            tenant_id,
            platform_result.task_id,
            platform_result.run_id,
        )
        read_deps = resolve_host_diagnostic_runtime_dependencies(
            env_wiring=composition.env_wiring,
            observability=composition.observability,
        )
        if read_deps is not None:
            read_service = build_diagnostic_read_service(read_deps)
            _ = read_service.list_problems(tenant_id=tenant_id)

        return ScenarioRuntimeProofResult(
            slug=slug,
            tenant_id=tenant_id,
            task_id=platform_result.task_id,
            run_id=str(platform_result.run_id),
            terminal_state=platform_result.task_result.state,
            has_runtime_events=reconstruction.has_runtime_events,
            has_terminal_diagnostic_trigger=composition.has_terminal_diagnostic_trigger,
            reconstruction_complete=reconstruction.is_runtime_history_complete,
        )
    finally:
        if composition.workspace is not None:
            cleanup_scenario_runtime_workspace(composition.workspace)

