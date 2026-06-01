# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
from __future__ import annotations
from dataclasses import replace
from typing import Optional

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from legal.config.legal_agent_config import LegalAgentConfig
from intergrax.agents.uaep_pipeline import pipeline_step_complete
from legal.uaep.dynamic_steps import (
    FINAL_DYNAMIC_STEP_ID,
    legal_dynamic_agent_steps,
    run_legal_dynamic_uaep_step,
)
from legal.uaep.thin_steps import (
    FINAL_SEQUENTIAL_STEP_ID,
    legal_sequential_agent_steps,
    run_legal_uaep_step,
)
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.ingestion.attachments import FileSystemAttachmentResolver
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.task.task import TaskContext
from legal.config.tool_planner_wiring import resolve_legal_tool_planner
from intergrax.applications._shared.runtime_config_bridge import apply_policy_bundle_to_runtime_config
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy, RuntimePolicies
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle

from legal.pipeline.legal_dynamic_pipeline import LegalDynamicPipeline
from legal.pipeline.legal_agent_pipeline import LegalAnalysisPipeline


class LegalAgent(Agent):
    """
    Real business agent: contract analysis.

    Sequential and dynamic modes expose UAEP macro-steps (Phase E). Sequential
    runs fixed domain stages; dynamic runs setup → tool plan → route → waves → finalize.
    """

    def __init__(
        self,
        *,
        config: LegalAgentConfig,
    ) -> None:
        self._config = config

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="legal",
            name="Legal Agent",
            description="Contract analysis and legal document review.",
            version="1.0.0",
            capabilities=["legal.contract_review"],
            skill_ids=["legal.contract_review"],
            allowed_tools=["rag", "websearch", "tools"],
            required_adapters=["llm"],
            risk_level=AgentRiskLevel.HIGH,
            max_steps=20,
            validation_rules=["non_empty_answer"],
            failure_modes=["governance_abort", "budget_exceeded"],
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, "legal.contract_review"):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.get_contract().id,
                matched_capabilities=["legal.contract_review"],
                score=1.0,
                rationale="legal contract review capability",
            )
        return CapabilityMatchResult(matched=False, rationale="not a legal task")

    @property
    def data_compliance_policy(self) -> DataCompliancePolicy:
        """Egress policy for product API surfaces (trace, tool args); same object as on :class:`LegalAgentConfig`."""
        return self._config.data_compliance

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:

        cfg = self._config

        runtime_policies = replace(
            RuntimePolicies(),
            data_compliance=cfg.data_compliance,
        )

        runtime_config = RuntimeConfig(
            llm_adapter=cfg.llm_adapter,
            enable_rag=cfg.enable_rag
            and cfg.embedding_manager is not None
            and cfg.vectorstore_manager is not None,
            enable_websearch=cfg.enable_websearch,
            production_mode=cfg.production_mode,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            embedding_manager=cfg.embedding_manager,
            vectorstore_manager=cfg.vectorstore_manager,
            tool_planner=resolve_legal_tool_planner(cfg),
            tools_mode=cfg.tools_mode,
            tool_providers=tuple(cfg.tool_providers),
            tool_profile=cfg.tool_profile,
            tool_wiring_context=cfg.tool_wiring_context,
            websearch_executor=cfg.websearch_executor,
            websearch_config=cfg.websearch_config,
            run_budget=cfg.run_budget,
            budget_policy=cfg.budget_policy,
            runtime_policies=runtime_policies,
        )
        if isinstance(cfg.policy_bundle, RuntimePolicyBundle):
            apply_policy_bundle_to_runtime_config(runtime_config, cfg.policy_bundle)

        if cfg.enable_sequential_legal_pipeline:
            runtime_config.pipeline = LegalAnalysisPipeline(config=cfg)
        else:
            runtime_config.pipeline = LegalDynamicPipeline(config=cfg)

        ingestion_service: Optional[AttachmentIngestionService] = None

        if runtime_config.enable_rag:
            ingestion_service = AttachmentIngestionService(
                embedding_manager=cfg.embedding_manager,
                vectorstore_manager=cfg.vectorstore_manager,
                resolver=FileSystemAttachmentResolver(),
                loader=cfg.documents_loader,
                splitter=cfg.documents_splitter,
            )

        context = RuntimeContext.build(
            config=runtime_config,
            session_manager=cfg.session_manager,
            ingestion_service=ingestion_service,
            governance_service=cfg.governance_service,
        )

        return context

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        allowed = list(contract.allowed_tools)
        if self._config.enable_sequential_legal_pipeline:
            return legal_sequential_agent_steps(allowed_tools=allowed)
        return legal_dynamic_agent_steps(allowed_tools=allowed)

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        if self._config.enable_sequential_legal_pipeline:
            return await run_legal_uaep_step(step, ctx, config=self._config)
        return await run_legal_dynamic_uaep_step(step, ctx, config=self._config)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = output, ctx
        final_step_id = (
            FINAL_SEQUENTIAL_STEP_ID
            if self._config.enable_sequential_legal_pipeline
            else FINAL_DYNAMIC_STEP_ID
        )
        if step.step_id != final_step_id:
            return AgentDecision(
                type=AgentDecisionType.CONTINUE,
                reason=f"{step.step_id} finished",
            )
        return pipeline_step_complete(reason="legal pipeline finished")
