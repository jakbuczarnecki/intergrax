# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
from __future__ import annotations
from dataclasses import replace
from typing import Optional

from intergrax.agents.agent_contract import Agent
from legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.ingestion.attachments import FileSystemAttachmentResolver
from intergrax.runtime.nexus.ingestion.ingestion_service import AttachmentIngestionService
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy, RuntimePolicies

from legal_agent.pipeline.legal_dynamic_pipeline import LegalDynamicPipeline
from legal_agent.pipeline.legal_agent_pipeline import LegalAnalysisPipeline


class LegalAgent(Agent):
    """
    Real business agent: contract analysis.
    """

    def __init__(
        self,
        *,
        config: LegalAgentConfig,
    ) -> None:
        self._config = config

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
            tools_agent=cfg.tools_agent,
            tools_mode=cfg.tools_mode,
            tool_providers=tuple(cfg.tool_providers),
            websearch_executor=cfg.websearch_executor,
            websearch_config=cfg.websearch_config,
            run_budget=cfg.run_budget,
            budget_policy=cfg.budget_policy,
            runtime_policies=runtime_policies,
        )

        # --- PIPELINE ---
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

        # --- BUILD CONTEXT ---
        context = RuntimeContext.build(
            config=runtime_config,
            session_manager=cfg.session_manager,
            ingestion_service=ingestion_service,
            governance_service=cfg.governance_service,
        )

        return context