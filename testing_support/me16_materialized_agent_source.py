# © Artur Czarnecki. All rights reserved.

"""Materialized ME-16 mixed worker source emitted into agent distribution bundles."""

from __future__ import annotations

import textwrap


def render_me16_materialized_agent_module(*, function_name: str) -> str:
    return (
        textwrap.dedent(
            f'''
            from intergrax.agents.authoring.runtime_tool_helpers import invoke_catalog_tool
            from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
            from intergrax.agents.reference_harness import (
                build_lab_agent_runtime_context,
                default_reference_harness,
            )
            from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
            from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
            from intergrax.contracts.agent_step import AgentStep, StepOutput
            from intergrax.contracts.capability import CapabilityMatchResult
            from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
            from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
            from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
            from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
            from intergrax.runtime.task.task import TaskContext
            from intergrax.skills.execution_binding import resolve_bound_skill_pack
            from intergrax.skills.registry.profile import SkillProfile
            from testing_support.builder import MeteringFakeLLMAdapter

            _CONTRACT_ID = "me16-mixed-capability-agent"
            _CAPABILITY = "me16.mixed.capability"
            _TOOL_ID = "tools.me14.canonical-echo"
            _SKILL_ID = "skills.me15.canonical-instruction"
            _TENANT_ID = "tenant-me16"


            class _MaterializedMe16MixedAgent(HarnessReferenceAgent):
                contract_id = _CONTRACT_ID
                capabilities = (_CAPABILITY, _TOOL_ID)
                agent_name = "ME-16 Mixed Capability Worker"
                agent_description = "Consumes bound skill context and invokes catalog tool."

                def get_contract(self) -> AgentContract:
                    return AgentContract(
                        id=self.contract_id,
                        name=self.agent_name,
                        description=self.agent_description,
                        version="1.0.0",
                        capabilities=[_CAPABILITY, _TOOL_ID],
                        skills=[],
                        extra_tools=[],
                        allowed_tools=[_TOOL_ID],
                        risk_level=AgentRiskLevel.LOW,
                        lifecycle_state=AgentLifecycleState.PRODUCTION,
                        production_eligible=True,
                        owner_team="platform",
                        owner_contact="harness@intergrax",
                        on_call_contact="harness@intergrax",
                        runbook_ref="docs/project/architecture/CAPABILITY_MARKETPLACE_ENGINE.md",
                        modality_profile_id="me16.mixed",
                        input_schema={{"type": "object", "properties": {{"message": {{"type": "string"}}}}}},
                        output_schema={{"type": "object", "properties": {{"answer": {{"type": "string"}}}}}},
                        validation_rules=["structured_output"],
                        failure_modes=["tool_invoke_failed", "skill_binding_missing"],
                        max_steps=1,
                    )

                def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
                    if task_context.capability in (None, _CAPABILITY):
                        return CapabilityMatchResult(
                            matched=True,
                            agent_id=self.contract_id,
                            matched_capabilities=[_CAPABILITY],
                            score=1.0,
                            rationale="me16 mixed capability worker",
                        )
                    return CapabilityMatchResult(matched=False, rationale="capability not supported")

                def build_context(self, request: RuntimeRequest) -> RuntimeContext:
                    return build_lab_agent_runtime_context(
                        request=request,
                        llm_adapter=MeteringFakeLLMAdapter(),
                        harness=default_reference_harness(),
                    )

                def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
                    del context
                    return [AgentStep(step_id="mixed_step", step_name="Skill context + tool invoke")]

                async def run_step(
                    self,
                    step: AgentStep,
                    ctx: RuntimeExecutionContext,
                ) -> StepOutput:
                    del step
                    runtime_state = ctx.metadata.get("runtime_state")
                    if not isinstance(runtime_state, RuntimeState):
                        raise RuntimeError("runtime_state missing for skill resolution")
                    config = runtime_state.context.config
                    pack = resolve_bound_skill_pack(
                        tenant_id=_TENANT_ID,
                        skill_profile=SkillProfile(enabled=[_SKILL_ID]),
                        skill_registry=config.skill_registry,
                        pinning_store=config.skill_pinning_store,
                    )
                    if not pack.prompt_instruction_ids:
                        raise RuntimeError("bound skill pack lacks instruction markers")
                    marker = sorted(pack.prompt_instruction_ids)[0]
                    payload = await invoke_catalog_tool(
                        ctx,
                        tool_name=_TOOL_ID,
                        agent_id=_CONTRACT_ID,
                        step_id="mixed_step",
                        tool_input={{"message": "ping"}},
                    )
                    tool_result = str(payload.get("result", ""))
                    if not tool_result:
                        raise RuntimeError(f"tool invocation failed: {{payload}}")
                    answer = f"{{marker}}|{{tool_result}}"
                    return StepOutput(
                        step_id="mixed_step",
                        summary=answer,
                        data={{"answer": answer, "marker": marker, "tool_result": tool_result}},
                    )


            def {function_name}(ctx, binding):
                del ctx, binding
                return _MaterializedMe16MixedAgent()
            '''
        ).strip()
        + "\n"
    )


__all__ = ["render_me16_materialized_agent_module"]
