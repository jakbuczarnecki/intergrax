# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import ToolChoiceMode
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy
from intergrax.rag.document_loaders.contracts.base_document_loader import BaseDocumentsLoader
from intergrax.rag.document_splitters.contracts.base_documents_splitter import BaseDocumentsSplitter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from legal.prompts.legal_agent_llm_prompts import (
    DEFAULT_ORGANIZATION_COMPLIANCE_POLICY,
)
from legal.config.legal_failure_policy import LegalFailurePolicy
from legal.memory.legal_memory_policy import (
    LegalMemoryPolicy,
    LegalMemoryPolicyPresets,
)
from legal.governance.legal_response_governance_port import (
    LegalResponseGovernancePort,
)
from legal.governance.legal_tool_plan_governance_port import (
    LegalToolPlanGovernancePort,
)
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.budget.budget_models import BudgetPolicy, RunBudget
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.tools.core.provider import ToolProvider
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.websearch.service.websearch_config import WebSearchConfig
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


class LegalAgentConfig(BaseModel):
    """
    Full configuration for the Legal Agent (tier-2).

    This is a single source of truth for the Legal Agent.
    Optional SKU defaults: :class:`~legal.config.legal_agent_product_profiles.LegalAgentProductProfile`.
    Failure & degradation contract: :class:`~legal.config.legal_failure_policy.LegalFailurePolicy`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_manager: SessionManager
    llm_adapter: LLMAdapter

    production_mode: bool = True

    enable_websearch: bool = False

    websearch_executor: Optional[WebSearchExecutor] = None
    websearch_config: Optional[WebSearchConfig] = None

    enable_rag: bool = False
    embedding_manager: Optional[BaseEmbeddingManager] = None
    vectorstore_manager: Optional[BaseVectorstoreManager] = None
    documents_loader: Optional[BaseDocumentsLoader] = None
    documents_splitter: Optional[BaseDocumentsSplitter] = None

    governance_service: Optional[GovernanceService] = None

    run_budget: Optional[RunBudget] = None
    budget_policy: Optional[BudgetPolicy] = Field(
        default=None,
        description="Required when run_budget is set; forwarded to RuntimeConfig for engine RunBudget enforcement.",
    )

    legal_response_governance: Optional[LegalResponseGovernancePort] = Field(
        default=None,
        description=(
            "Optional product-layer transform after LegalFinalizeAnswerStep draft: disclaimers, "
            "uncertainty summary, format version (see LegalShapedClientResponse). None keeps draft as "
            "RuntimeAnswer.answer."
        ),
    )

    legal_tool_plan_governance: Optional[LegalToolPlanGovernancePort] = Field(
        default=None,
        description=(
            "Optional dynamic clamp of LegalToolPlan after static organization rules and before "
            "the Nexus bridge. Implement :class:`LegalToolPlanGovernancePort`; often the same "
            "instance as governance_service when your platform class implements both evaluate() "
            "and adjust_legal_tool_plan()."
        ),
    )

    tool_planner: Optional[ToolPlannerProtocol] = Field(
        default=None,
        description=(
            "Optional explicit tool planner. When None and tool_profile is set, "
            "LegalAgent builds CatalogToolPlanner (catalog ToolRuntime path)."
        ),
    )
    tools_mode: ToolChoiceMode = "off"
    tool_providers: List[ToolProvider] = Field(default_factory=list)
    tool_profile: Optional[ToolProfile] = Field(
        default=None,
        description="Catalog tools enabled for Nexus RuntimeConfig (Phase O.5+).",
    )
    tool_wiring_context: Optional[ToolWiringContext] = Field(
        default=None,
        description="Dependencies for catalog tool handlers (integrations, RAG managers).",
    )

    use_legal_tool_decision: bool = Field(
        default=False,
        description=(
            "If True, run Tier-2 LegalToolDecision (LLM) then Nexus RagStep/WebsearchStep/ToolsStep "
            "before legal stage routing. Enable in production when RAG/tools/websearch are wired."
        ),
    )

    organization_compliance_policy: str = Field(
        default=DEFAULT_ORGANIZATION_COMPLIANCE_POLICY,
        description=(
            "Full policy text for LegalPolicyComplianceStep. "
            "Override per tenant/org. Set to empty string to skip that step."
        ),
    )

    organization_allow_rag: bool = Field(
        default=True,
        description=(
            "If False, organization governance clamps LegalToolPlan.use_rag before the runtime "
            "bridge (Nexus RagStep will not run for this agent even if tool-decision requests it)."
        ),
    )
    organization_allow_websearch: bool = Field(
        default=True,
        description=(
            "If False, governance clamps use_websearch before the runtime bridge "
            "(Nexus WebsearchStep will not run)."
        ),
    )
    organization_allow_tools: bool = Field(
        default=True,
        description=(
            "If False, governance clamps use_tools before the runtime bridge "
            "(Nexus ToolsStep will not run)."
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
        description=(
            "Execution budget (Tier-2): max plan→execute waves per LegalDynamicPipeline run "
            "(includes first pass). Caps runaway replan/evaluator loops; pair with RuntimeConfig "
            "run_budget for RAG/websearch/tools/token limits on RuntimeConfig."
        ),
    )

    legal_loop_max_same_routing_repeats: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Stop replanning when the merged routing fingerprint repeats this many times.",
    )

    legal_loop_early_exit: bool = Field(
        default=True,
        description=(
            "After a wave, skip evaluator/replan when decision ran this wave, confidence is high, "
            "and there are no policy violations or blocking issues."
        ),
    )

    legal_loop_early_exit_min_confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum decision.confidence for legal_loop_early_exit (requires run_decision this wave).",
    )

    memory_policy: LegalMemoryPolicy = Field(
        default_factory=LegalMemoryPolicyPresets.default,
        description=(
            "Tier-2 memory contract: conversation snippet limits for routing/tool/replan LLM calls, "
            "and whether workspace metrics snapshot is written/read on the session. Fully host-controlled "
            "(build :class:`~legal.memory.legal_memory_policy.LegalMemoryPolicy` "
            "or use presets from the same module). Ignored by sequential pipeline except finalize persistence flags."
        ),
    )

    failure_policy: LegalFailurePolicy = Field(
        default_factory=LegalFailurePolicy.product_default,
        description=(
            "Prescriptive product contract for empty retrieval, Nexus/tool degradation, governance clamps, "
            "low-confidence decisions, and evaluator/finalize failures. Hosts replace only when explicitly "
            "changing SKU guarantees; see :class:`~legal.config.legal_failure_policy.LegalFailurePolicy`."
        ),
    )

    data_compliance: DataCompliancePolicy = Field(
        default_factory=DataCompliancePolicy,
        description=(
            "Product data egress rules forwarded to :class:`~intergrax.runtime.nexus.config.RuntimeConfig` "
            "(via :class:`~intergrax.runtime.nexus.policies.runtime_policies.RuntimePolicies`) and applied "
            "when shaping Legal HTTP responses (trace serialization, tool argument exposure)."
        ),
    )

    @model_validator(mode="after")
    def _budget_policy_when_run_budget(self) -> "LegalAgentConfig":
        if self.run_budget is not None and self.budget_policy is None:
            raise ValueError("budget_policy is required when run_budget is set.")
        return self
