# © Artur Czarnecki. All rights reserved.

"""ME-14 canonical tool execution via public ``HarnessHostRuntime.execution``."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_config,
    default_reference_harness,
)
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.authoring.runtime_tool_helpers import invoke_catalog_tool
from intergrax.applications._shared.application_owned_tool_conformance import (
    application_owned_tool_declarations,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry import ToolProfile, ToolRegistry
from testing_support.builder import FakeLLMAdapter


ME14_PROOF_AGENT_ID = "me14_tool_proof"
ME14_PROOF_TENANT = "tenant-me14"


class Me14ToolProofAgent(HarnessReferenceAgent):
    """Single-step UAEP agent that invokes one activated catalog tool."""

    contract_id = ME14_PROOF_AGENT_ID
    agent_name = "ME-14 Tool Proof Agent"
    agent_description = "Invokes activated marketplace tool via UAEP"
    capabilities = ("me14.tool.invoke",)
    _execution_registry: ClassVar[ToolRegistry | None] = None

    def __init__(self, *, tool_id: str | None = None) -> None:
        from testing_support.canonical_me14_echo_tool import ME14_TOOL_LOGICAL_ID

        self._tool_id = tool_id or ME14_TOOL_LOGICAL_ID

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self.contract_id,
            name=self.agent_name,
            description=self.agent_description,
            version="0.1.0",
            capabilities=list(self.capabilities),
            skills=[],
            extra_tools=[],
            allowed_tools=[self._tool_id],
            risk_level=AgentRiskLevel.LOW,
            lifecycle_state=AgentLifecycleState.PRODUCTION,
            production_eligible=True,
            owner_team="platform",
            owner_contact="harness@intergrax",
            on_call_contact="harness@intergrax",
            runbook_ref="docs/project/architecture/TOOLS.md",
            modality_profile_id="me14.tool_proof",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            validation_rules=["structured_output"],
            failure_modes=["tool_invoke_failed"],
            max_steps=1,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability in (None, "me14.tool.invoke"):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=["me14.tool.invoke"],
                score=1.0,
                rationale="me14 tool proof agent",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
        from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
        from testing_support.builder import FakeLLMAdapter

        harness = default_reference_harness()
        if Me14ToolProofAgent._execution_registry is not None:
            registry = Me14ToolProofAgent._execution_registry
            tool_wiring = ToolWiringContext(extras={"me14_proof_registry": registry})
            harness = LabHarnessContext(
                policy_bundle=harness.policy_bundle,
                tool_wiring_context=tool_wiring,
            )
        config = build_lab_agent_runtime_config(
            request=request,
            llm_adapter=FakeLLMAdapter(),
            harness=harness,
        )
        if Me14ToolProofAgent._execution_registry is not None:
            registry = Me14ToolProofAgent._execution_registry
            config.tool_invoker = RuntimeToolInvoker(
                registry=registry,
                executor=RegistryToolExecutor(registry),
            )
        return RuntimeContext.build(
            config=config,
            session_manager=SessionManager(storage=InMemorySessionStorage()),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        del context
        return [AgentStep(step_id="invoke_tool", step_name="Invoke ME-14 proof tool")]

    async def run_step(
        self,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepOutput:
        del step
        payload = await invoke_catalog_tool(
            ctx,
            tool_name=self._tool_id,
            agent_id=ME14_PROOF_AGENT_ID,
            step_id="invoke_tool",
            tool_input={"message": "ping"},
        )
        result_text = str(payload.get("result", ""))
        if not result_text and payload.get("status") != "success":
            raise RuntimeError(f"catalog tool invocation failed: {payload}")
        if not result_text:
            result_text = str(payload.get("reason", payload))
        return StepOutput(
            step_id="invoke_tool",
            summary=result_text,
            data={"answer": result_text, "result": result_text},
        )


def me14_tool_proof_manifest(tool_logical_id: str) -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id="me14_tool_proof",
        name="ME-14 Tool Proof",
        route_prefix="/v1/me14_tool_proof",
        env_prefix="ME14_TOOL_PROOF_",
        agents=[
            AgentBinding.mount(
                Me14ToolProofAgent,
                contract_id=ME14_PROOF_AGENT_ID,
                builder_key=ME14_PROOF_AGENT_ID,
                requires_uaep=True,
                tool_allowlist_extra=[tool_logical_id],
            ),
        ],
        application_owned_tools=application_owned_tool_declarations([tool_logical_id]),
    )


def me14_tool_proof_environment(tool_logical_id: str) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.lab_defaults(
        profile_id="me14.tool_proof",
        harness_tools=False,
    ).model_copy(
        update={
            "tool_profile": ToolProfile(enabled=[tool_logical_id]),
            "skill_profile": SkillProfile(enabled=[]),
        },
    )


def build_me14_proof_agent(**_: object) -> Me14ToolProofAgent:
    from testing_support.canonical_me14_echo_tool import ME14_TOOL_LOGICAL_ID

    return Me14ToolProofAgent(tool_id=ME14_TOOL_LOGICAL_ID)


async def execute_me14_tool_via_host_execution_engine(
    *,
    registry: ToolRegistry,
    tool_logical_id: str,
    tmp_path: Path,
) -> tuple[str, str, str | None]:
    """Returns ``(tool_id, execution_output, task_id)`` via ``HostTaskExecution.execute``."""
    manifest = me14_tool_proof_manifest(tool_logical_id)
    environment = me14_tool_proof_environment(tool_logical_id)
    Me14ToolProofAgent._execution_registry = registry
    try:
        host_runtime = build_harness_host_runtime(
            manifest,
            environment,
            tenant_id=ME14_PROOF_TENANT,
            trace_db_path=tmp_path / "trace.db",
            runtime_events_db_path=tmp_path / "runtime_events.db",
            application_tool_registry=registry,
            llm_adapter=FakeLLMAdapter(),
        )
        from intergrax.agents.persistence.tool_invoker_wiring import inject_acp_tool_invoker_metadata
        from intergrax.contracts.execution_identity import mint_run_id

        task = Task(
            tenant_id=ME14_PROOF_TENANT,
            user_id="me14-proof-user",
            message="invoke-me14-tool",
            agent_id=ME14_PROOF_AGENT_ID,
            context=TaskContext(capability="me14.tool.invoke"),
        )
        run_id = mint_run_id()
        metadata: dict[str, object] = {}
        inject_acp_tool_invoker_metadata(
            metadata,
            host_runtime._internal_composition._orchestration_backend._declarative_tool_invoker,
            task_id=task.task_id,
            run_id=run_id,
            agent_id=ME14_PROOF_AGENT_ID,
            tenant_id=ME14_PROOF_TENANT,
        )
        task = task.model_copy(update={"metadata": metadata})
        result: TaskResult = await host_runtime.execution.execute(task)
        assert result.state is TaskState.COMPLETED
        assert result.answer is not None
        return tool_logical_id, result.answer, str(result.task_id) if result.task_id else None
    finally:
        Me14ToolProofAgent._execution_registry = None


__all__ = [
    "ME14_PROOF_AGENT_ID",
    "Me14ToolProofAgent",
    "build_me14_proof_agent",
    "execute_me14_tool_via_host_execution_engine",
    "me14_tool_proof_environment",
    "me14_tool_proof_manifest",
]
