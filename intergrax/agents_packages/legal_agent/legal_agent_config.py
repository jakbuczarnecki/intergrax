# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.document_loaders.contracts.base_document_loader import BaseDocumentsLoader
from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.agents_packages.legal_agent.legal_agent_llm_prompts import (
    DEFAULT_ORGANIZATION_COMPLIANCE_POLICY,
)
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.session.session_manager import SessionManager


class LegalAgentConfig(BaseModel):
    """
    Full configuration for the Legal Agent (tier-2).

    This is a single source of truth for the Legal Agent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_manager: SessionManager
    llm_adapter: LLMAdapter

    production_mode: bool = True
    
    enable_websearch: bool = False


    enable_rag: bool = False
    embedding_manager: Optional[BaseEmbeddingManager] = None
    vectorstore_manager: Optional[BaseVectorstoreManager] = None
    documents_loader: Optional[BaseDocumentsLoader] = None
    documents_splitter: Optional[BaseDocumentsSplitter] = None

    governance_service: Optional[GovernanceService] = None

    organization_compliance_policy: str = Field(
        default=DEFAULT_ORGANIZATION_COMPLIANCE_POLICY,
        description=(
            "Full policy text for LegalPolicyComplianceStep. "
            "Override per tenant/org. Set to empty string to skip that step."
        ),
    )

    enable_sequential_legal_pipeline: bool = Field(
        default=False,
        description=(
            "If True, use fixed-order LegalAnalysisPipeline (no SETUP_STEPS, no LLM routing). "
            "If False (default), use LegalDynamicPipeline: session/history setup + routed stages."
        ),
    )

    use_llm_legal_route_planner: bool = Field(
        default=True,
        description=(
            "When using LegalDynamicPipeline: if True, LLM selects which stages to run "
            "(with deterministic dependency closure). If False, run all stages except "
            "those that self-skip inside steps."
        ),
    )

    use_legal_run_evaluator: bool = Field(
        default=True,
        description=(
            "When using LegalDynamicPipeline: after executing the routed stages, call an "
            "LLM evaluator. If it requests replanning (and limits allow), merge a new "
            "routing plan and run only stages not yet completed this run."
        ),
    )

    use_legal_route_replanner: bool = Field(
        default=True,
        description=(
            "If True (and use_llm_legal_route_planner), replanning uses an LLM to propose "
            "additional stages. If False, replan falls back to union with a full routing "
            "(all stages on) once per loop."
        ),
    )

    legal_loop_max_iterations: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Max plan→execute cycles per LegalDynamicPipeline run (includes first pass).",
    )

    legal_loop_max_same_routing_repeats: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Stop replanning when the merged routing fingerprint repeats this many times.",
    )
