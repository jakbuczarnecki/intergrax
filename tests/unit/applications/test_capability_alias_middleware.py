# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-3 — capability alias intake middleware and lifecycle persist."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.applications.contracts.capability_alias import (
    CAPABILITY_ALIAS_REDIRECT_KEY,
    CapabilityAlias,
    CapabilityGovernanceProfile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications._shared.capability_alias_middleware import CapabilityAliasMiddleware
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.hooks.hook_context import HookAction
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.nexus_lifecycle_hooks import (
    NexusLifecycleHookCoordinator,
    nexus_lifecycle_hook_context,
)
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_middleware_redirects_capability_on_intake() -> None:
    env = ApplicationEnvironmentProfile(
        execution_mode=ExecutionMode.STRICT,
        capability_governance_profile=CapabilityGovernanceProfile(
            aliases=[
                CapabilityAlias(
                    alias="research.pipeline",
                    canonical="research.orchestrate",
                    effective_from="2026-01-01T00:00:00Z",
                    sunset_at="2027-01-01T00:00:00Z",
                ),
            ],
        ),
    )
    middleware = CapabilityAliasMiddleware(environment=env)
    ctx = nexus_lifecycle_hook_context(
        Task(
            task_id="run-1",
            tenant_id="t1",
            user_id="u1",
            message="hello",
            context=TaskContext(capability="research.pipeline"),
        ),
        phase=ExecutionPhase.INTAKE,
    )
    result = await middleware.before(HookPoint.BEFORE_TASK_INTAKE, ctx)
    assert result.action == HookAction.ALLOW
    assert ctx.runtime_state["capability"] == "research.orchestrate"
    redirect = ctx.runtime_state[CAPABILITY_ALIAS_REDIRECT_KEY]
    assert redirect["redirected"] is True
    assert redirect["canonical_capability"] == "research.orchestrate"


@pytest.mark.asyncio
async def test_lifecycle_coordinator_persists_redirect_to_task() -> None:
    env = ApplicationEnvironmentProfile(
        execution_mode=ExecutionMode.BALANCED,
        capability_governance_profile=CapabilityGovernanceProfile(
            aliases=[
                CapabilityAlias(
                    alias="legacy.cap",
                    canonical="modern.cap",
                    effective_from="2026-01-01T00:00:00Z",
                    sunset_at="2027-01-01T00:00:00Z",
                ),
            ],
        ),
    )
    pipeline = MiddlewarePipeline(middleware=[CapabilityAliasMiddleware(environment=env)])
    coordinator = NexusLifecycleHookCoordinator(pipeline)
    task = Task(
        task_id="run-2",
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="legacy.cap"),
    )
    await coordinator.before(
        HookPoint.BEFORE_TASK_INTAKE,
        task,
        phase=ExecutionPhase.INTAKE,
    )
    assert task.context.capability == "modern.cap"
    assert task.metadata[CAPABILITY_ALIAS_REDIRECT_KEY]["requested_capability"] == "legacy.cap"
