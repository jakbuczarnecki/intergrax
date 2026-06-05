# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.applications._shared.application_security_wiring import register_application_security_hooks
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.security.task_security_context import RESOURCE_TENANT_ID_METADATA_KEY
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_tenant_security_blocks_cross_tenant_resource_access() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry)
    register_application_security_hooks(
        loop,
        ApplicationSecurityProfile(
            prompt_defense_enabled=False,
            tool_injection_defense_enabled=False,
            retrieval_poisoning_defense_enabled=False,
            tenant_security_verify_enabled=True,
        ),
    )

    task = Task(
        tenant_id="tenant-a",
        user_id="user-1",
        message="hello",
        context=TaskContext(
            capability="echo.basic",
            metadata={RESOURCE_TENANT_ID_METADATA_KEY: "tenant-b"},
        ),
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.FAILED
    assert result.summary.validation.valid is False
    assert any("Tenant isolation" in error for error in result.summary.validation.errors)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_tenant_security_allows_matching_tenant() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(registry)
    register_application_security_hooks(
        loop,
        ApplicationSecurityProfile(
            prompt_defense_enabled=False,
            tool_injection_defense_enabled=False,
            retrieval_poisoning_defense_enabled=False,
            tenant_security_verify_enabled=True,
        ),
    )

    task = Task(
        tenant_id="tenant-a",
        user_id="user-1",
        message="hello harness",
        context=TaskContext(
            capability="echo.basic",
            metadata={RESOURCE_TENANT_ID_METADATA_KEY: "tenant-a"},
        ),
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.COMPLETED
    assert "hello harness" in (result.answer or "")
