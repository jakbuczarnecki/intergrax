# © Artur Czarnecki. All rights reserved.

"""APP-CON-2/4 — ApplicationEnvironmentState v2 and artifact contracts."""

from __future__ import annotations

from intergrax.applications.contracts.application_artifacts import (
    ApplicationArtifactRef,
    ArtifactRetentionPolicy,
    ArtifactSecurityClass,
    RunArtifactBundle,
    WorkspaceArtifactRef,
)
from intergrax.applications.contracts.environment_state import (
    ActiveBudgetState,
    ApplicationEnvironmentState,
    EnvironmentHealthStatus,
    EnvironmentTaskPhase,
    HitlEscalationState,
    PendingNotification,
    PolicyOverlayState,
    SandboxIsolationRef,
    WorkspaceIsolationRef,
    seed_application_environment_state,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode


def test_seed_application_environment_state_v2_shape() -> None:
    runtime = seed_application_environment_state(
        app_id="lab",
        profile_id="lab-default",
        execution_mode=ExecutionMode.BALANCED,
        task_id="task-1",
        organization_id="org-1",
        active_scenario_id="scenario-a",
    )
    state = ApplicationEnvironmentState.from_runtime_state(runtime)
    assert state is not None
    assert state.schema_version == "app_env_state.v2"
    assert state.task_id == "task-1"
    assert state.phase == EnvironmentTaskPhase.INTAKE
    assert state.health == EnvironmentHealthStatus.HEALTHY
    assert state.policy_overlays.organization_id == "org-1"
    assert state.policy_overlays.active_scenario_id == "scenario-a"


def test_environment_state_round_trip_and_patch() -> None:
    state = ApplicationEnvironmentState(
        app_id="legal",
        profile_id="legal-prod",
        task_id="t-99",
        phase=EnvironmentTaskPhase.GRAPH_EXECUTION,
        health=EnvironmentHealthStatus.BUDGET_PRESSURE,
        hitl=HitlEscalationState(pending=True, ticket_id="hitl-1"),
        budget=ActiveBudgetState(agent_tokens_total=900, agent_tokens_limit=1000, warn_emitted=True),
        shadow_workspace=WorkspaceIsolationRef(
            workspace_id="ws-1",
            tenant_id="tenant",
            task_id="t-99",
            root_path="/tmp/ws",
        ),
        sandbox_session=SandboxIsolationRef(
            session_id="sb-1",
            tenant_id="tenant",
            task_id="t-99",
        ),
        pending_notifications=[
            PendingNotification(channel="slack", template_id="budget_warn"),
        ],
        policy_overlays=PolicyOverlayState(effective_tool_denies=["dangerous.tool"]),
    )
    runtime = state.apply_to_runtime_state({})
    restored = ApplicationEnvironmentState.from_runtime_state(runtime)
    assert restored is not None
    assert restored.phase == EnvironmentTaskPhase.GRAPH_EXECUTION
    assert restored.hitl.ticket_id == "hitl-1"
    assert restored.budget.warn_emitted is True
    assert restored.shadow_workspace is not None
    assert restored.shadow_workspace.workspace_id == "ws-1"
    patch = restored.patch_runtime_state()
    assert "app_env_state.v1" in patch
    assert patch["app_env_state.v1"]["health"] == "budget_pressure"


def test_run_artifact_bundle_links_task_and_run() -> None:
    ref = ApplicationArtifactRef(
        artifact_id="art-1",
        kind="export",
        uri="file:///tmp/out.json",
        task_id="task-1",
        owner_app_id="research",
        tenant_id="tenant-1",
        security_class=ArtifactSecurityClass.CONFIDENTIAL,
    )
    ws = WorkspaceArtifactRef(
        artifact_id="ws-art-1",
        workspace_id="ws-1",
        relative_path="reports/summary.md",
        uri="file:///tmp/ws/reports/summary.md",
        task_id="task-1",
        tenant_id="tenant-1",
        retention=ArtifactRetentionPolicy(retain_hours=24, delete_on_task_complete=False),
    )
    bundle = RunArtifactBundle(
        task_id="task-1",
        graph_id="graph-1",
        application=[ref],
        workspace=[ws],
    )
    assert bundle.schema_version == "run_artifact_bundle.v1"
    assert bundle.application[0].security_class == ArtifactSecurityClass.CONFIDENTIAL
    assert bundle.workspace[0].retention.retain_hours == 24
