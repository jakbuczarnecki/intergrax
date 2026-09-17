# © Artur Czarnecki. All rights reserved.

"""ME-16 mixed execution via public ``HarnessHostRuntime.execution``."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.application_owned_tool_conformance import (
    application_owned_tool_declarations,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry import ToolProfile, ToolRegistry
from testing_support.agent_platform_admin_harness import lifecycle_proof_durable_profile_stores
from testing_support.builder import FakeLLMAdapter
from testing_support.canonical_agent_lifecycle_composition import (
    CanonicalAgentLifecycleProofStack,
)
from testing_support.canonical_me16_mixed_agent import (
    ME16_MIXED_CAPABILITY,
    ME16_MIXED_CONTRACT_ID,
    ME16_MIXED_TASK_INPUT,
    ME16_MIXED_TENANT,
)
from testing_support.canonical_me14_echo_tool import ME14_TOOL_LOGICAL_ID

def me16_mixed_execution_manifest(
    *,
    logical_agent_id: str,
    tool_logical_id: str,
    application_id: str,
) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=application_id,
        name="ME-16 Mixed Capability Proof",
        route_prefix="/v1/me16_mixed",
        env_prefix="ME16_MIXED_",
        agents=[
            AgentBinding(
                contract_id=logical_agent_id,
                builder_key=logical_agent_id,
                tool_allowlist_extra=[tool_logical_id],
            ),
        ],
        application_owned_tools=application_owned_tool_declarations([tool_logical_id]),
    )


def me16_mixed_execution_environment(
    *,
    tool_logical_id: str,
    skill_profile: SkillProfile,
    environment_id: str,
):
    from testing_support.canonical_agent_lifecycle_composition import (
        _stage15_proof_environment,
    )

    env = _stage15_proof_environment(environment_id)
    return env.model_copy(
        update={
            "tool_profile": ToolProfile(enabled=[tool_logical_id]),
            "skill_profile": skill_profile,
        },
    )


async def run_me16_mixed_host_execution(
    *,
    agent_stack: CanonicalAgentLifecycleProofStack,
    tool_registry: ToolRegistry,
    skill_lifecycle: SkillHostLifecycleService,
    tmp_path: Path,
) -> TaskResult:
    projection = agent_stack.resolve_serving_projection()
    config = agent_stack.config
    manifest = me16_mixed_execution_manifest(
        logical_agent_id=config.logical_agent_id,
        tool_logical_id=ME14_TOOL_LOGICAL_ID,
        application_id=config.application_id,
    )
    environment = me16_mixed_execution_environment(
        tool_logical_id=ME14_TOOL_LOGICAL_ID,
        skill_profile=skill_lifecycle.skill_profile,
        environment_id=config.environment_id,
    )
    profile_stores = lifecycle_proof_durable_profile_stores(agent_stack.runtime_root)
    host_runtime = build_harness_host_runtime(
        manifest,
        environment,
        tenant_id=ME16_MIXED_TENANT,
        registry_projection=projection,
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "runtime_events.db",
        document_store=InMemoryDocumentStore(),
        revision_store=profile_stores.revision_store,
        pinning_store=profile_stores.pinning_store,
        active_store=profile_stores.active_store,
        application_tool_registry=tool_registry,
        application_skill_registry=skill_lifecycle.registry,
        llm_adapter=FakeLLMAdapter(),
    )
    task = Task(
        tenant_id=ME16_MIXED_TENANT,
        user_id="me16-proof-user",
        message=ME16_MIXED_TASK_INPUT,
        agent_id=ME16_MIXED_CONTRACT_ID,
        context=TaskContext(capability=ME16_MIXED_CAPABILITY),
    )
    return await host_runtime.execution.execute(task)


async def execute_me16_mixed_via_host_execution_engine(
    *,
    agent_stack: CanonicalAgentLifecycleProofStack,
    tool_registry: ToolRegistry,
    skill_lifecycle: SkillHostLifecycleService,
    tmp_path: Path,
) -> tuple[str, str, str | None]:
    result = await run_me16_mixed_host_execution(
        agent_stack=agent_stack,
        tool_registry=tool_registry,
        skill_lifecycle=skill_lifecycle,
        tmp_path=tmp_path,
    )
    assert result.state is TaskState.COMPLETED
    assert result.answer is not None
    assert result.agent_id == ME16_MIXED_CONTRACT_ID
    return result.agent_id or ME16_MIXED_CONTRACT_ID, result.answer, (
        str(result.task_id) if result.task_id else None
    )


__all__ = [
    "execute_me16_mixed_via_host_execution_engine",
    "me16_mixed_execution_environment",
    "me16_mixed_execution_manifest",
    "run_me16_mixed_host_execution",
]
