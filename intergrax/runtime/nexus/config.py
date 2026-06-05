# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Sequence, TYPE_CHECKING

from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.config_types import ToolChoiceMode, ToolsContextScope

if TYPE_CHECKING:
    from intergrax.integrations.registry.profile import IntegrationProfile
    from intergrax.runtime.events.event_bus import RuntimeEventBus
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile

from intergrax.rag.profiles.runtime_rag_sync import sync_rag_profile_from_runtime_config
from intergrax.runtime.nexus.config_sections import (
    ModelRuntimeConfig,
    PlanningRuntimeConfig,
    RetrievalRuntimeConfig,
    ToolsRuntimeConfig,
    TraceRuntimeConfig,
)

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.runtime.nexus.budget.budget_models import BudgetPolicy, RunBudget
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.planning.engine_plan_models import PlannerPromptConfig
from intergrax.runtime.nexus.planning.plan_loop_models import PlanLoopPolicy
from intergrax.runtime.nexus.planning.plan_sources import PlanSource
from intergrax.runtime.nexus.planning.step_executor_models import StepExecutorConfig
from intergrax.runtime.nexus.planning.step_planner import StepPlannerConfig
from intergrax.runtime.nexus.policies.runtime_policies import RuntimePolicies
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.runtime.modality.modality_profile import ModalityProfile
from intergrax.runtime.tools.scope_policy import ToolScopePolicy
from intergrax.tools.core.provider import ToolProvider
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry import ToolProfile, ToolRegistry, ToolWiringContext, build_registry_from_profile
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.websearch.service.websearch_config import WebSearchConfig
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


@dataclass
class RuntimeConfig:
    """
    Global configuration object for the nexus Runtime.

    This configuration defines:
      - Which LLM is used for generation.
      - How RAG (vectorstore-based retrieval) is applied.
      - Whether web search is available as an additional context source.
      - Whether a tools agent (for function/tool calling) can be used.

    The runtime is backend-agnostic and only depends on the abstract
    interfaces defined in the Intergrax framework.
    """

    # ------------------------------------------------------------------
    # CORE MODEL & RAG BACKENDS
    # ------------------------------------------------------------------

    # Primary LLM adapter used for chat-style generation.
    llm_adapter: LLMAdapter

    # Embedding manager used for RAG/document indexing and retrieval.
    embedding_manager: Optional[BaseEmbeddingManager] = None

    # Vectorstore manager providing semantic search over stored chunks.
    vectorstore_manager: Optional[BaseVectorstoreManager] = None

    # Optional full RAG stack (unified retrieval for Nexus ContextBuilder).
    retriever_manager: Optional[BaseRetrieverManager] = None
    reranker_manager: Optional[BaseRerankerManager] = None
    rag_profile: Optional[RagProfile] = None
    retrieval_service: Optional[RetrievalService] = None

    # ------------------------------------------------------------------
    # FEATURE FLAGS
    # ------------------------------------------------------------------

    # Enables Retrieval-Augmented Generation based on stored documents.
    enable_rag: bool = True

    # Enables real-time web search as an additional context layer.
    enable_websearch: bool = True
    

    # ------------------------------------------------------------------
    # MULTI-TENANCY
    # ------------------------------------------------------------------

    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # ------------------------------------------------------------------
    # RAG CONFIGURATION
    # ------------------------------------------------------------------

    # Maximum number of retrieved chunks per query.
    max_docs_per_query: int = 8

    # Maximum token budget reserved for RAG content.
    max_rag_tokens: int = 4096

    # Optional semantic score threshold for filtering low-quality hits.
    rag_score_threshold: Optional[float] = None


    # ------------------------------------------------------------------
    # LONG-TERM MEMORY (USER) RETRIEVAL CONFIGURATION
    # ------------------------------------------------------------------

    # Maximum number of long-term memory entries retrieved per query.
    max_longterm_entries_per_query: int = 8

    # Maximum token budget reserved for long-term memory context.
    max_longterm_tokens: int = 4096

    # Optional semantic score threshold for filtering low-quality long-term hits.
    longterm_score_threshold: Optional[float] = None


    # ------------------------------------------------------------------
    # WEB SEARCH CONFIGURATION
    # ------------------------------------------------------------------

    # Pre-configured executor capable of performing web search queries.
    # If None, web search is effectively unavailable.
    websearch_executor: Optional[WebSearchExecutor] = None

    websearch_config: Optional[WebSearchConfig] = None

    # ------------------------------------------------------------------
    # TOOLS / AGENT EXECUTION
    # ------------------------------------------------------------------

    # Optional tools agent responsible for:
    #   - planning tool calls,
    #   - invoking tools,
    #   - merging tool results into the final answer.
    #
    # If None, tools cannot be used regardless of tools_mode.
    tool_planner: Optional[ToolPlannerProtocol] = None

    # High-level policy defining whether tools may or must be used:
    #   - "off": do not use tools at all.
    #   - "auto": runtime may call tools if useful.
    #   - "required": runtime must use at least one tool.
    tools_mode: ToolChoiceMode = "auto"

    # Determines how much contextual information the tool planner receives:
    #
    #   - "current_message_only":
    #       Planner sees only the newest user query.
    #       Useful for strict function-calling, cost optimization
    #       and predictable single-turn behavior.
    #
    #   - "conversation":
    #       Planner sees full conversation history up to this point.
    #
    #   - "full":
    #       Planner receives the same context as the LLM:
    #       system → profile → history → RAG → websearch.
    #
    tools_context_scope: ToolsContextScope = ToolsContextScope.CURRENT_MESSAGE_ONLY

    tool_invoker: Optional[RuntimeToolInvoker] = None

    idempotency_store: Optional[IdempotencyStore] = None
    
    tool_providers: Sequence[ToolProvider] = ()

    tool_profile: Optional[ToolProfile] = None

    skill_profile: Optional[SkillProfile] = None

    tool_wiring_context: Optional[ToolWiringContext] = None

    modality_profile: Optional[ModalityProfile] = None

     # Optional capability-level tool authorization policy.
    # If None → all tools are allowed (backward compatible behavior).
    tool_scope_policy: Optional["ToolScopePolicy"] = None

    # Memory toggles
    enable_user_profile_memory: bool = True
    enable_org_profile_memory: bool = True
    enable_user_longterm_memory: bool = True
    enable_task_memory: bool = False
    memory_retention_days: Optional[int] = None
    memory_scope_boundary: str = "tenant"

    # Context assembly (Phase MEM-1.2, MEM-CTX.1)
    context_budget_policy: Optional["ContextBudgetPolicy"] = None
    task_context_assembly_options: Optional[TaskContextAssemblyOptions] = None
    context_decision_profile: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # MISC METADATA
    # ------------------------------------------------------------------

    # Arbitrary metadata for app-specific instrumentation or tags.
    metadata: Dict[str, Any] = field(default_factory=dict)



    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------
    enable_llm_usage_collection: bool = True


    # ------------------------------------------------------------------
    # PLANNING
    # ------------------------------------------------------------------

    # Optional explicit pipeline instance.
    # If provided, Runtime will run it.
    pipeline: Optional[RuntimePipeline] = None

    step_planner_cfg: Optional[StepPlannerConfig] = None

    step_executor_cfg: Optional[StepExecutorConfig] = None
    
    planner_prompt_config: Optional[PlannerPromptConfig] = None

    plan_loop_policy: Optional[PlanLoopPolicy] = None

    plan_source: Optional[PlanSource] = None


    # ------------------------------------------------------------------
    # RUNTIME POLICIES
    # ------------------------------------------------------------------

    # Hard timeout for a single runtime.run() execution.
    # If set, the entire pipeline execution is cancelled after this duration.
    runtime_timeout_ms: Optional[int] = None

    # Run-level retry (safety net). Defaults to OFF.
    max_run_retries: int = 0

    retry_run_on: FrozenSet[RuntimeErrorCode] = field(
        default_factory=lambda: frozenset(
            {RuntimeErrorCode.LLM_ERROR, RuntimeErrorCode.TOOL_ERROR}
        )
    )

    runtime_policies: RuntimePolicies = RuntimePolicies()

    # Warn if execution semaphore slot is held longer than this threshold (ms).
    # If None → disabled.
    execution_slot_warn_threshold_ms: Optional[int] = None

    hitl_default_message: Optional[str] = None


    # ------------------------------------------------------------------
    # BUDGET CONTROL
    # ------------------------------------------------------------------

    run_budget: Optional[RunBudget] = None
    budget_policy: Optional[BudgetPolicy] = None
    
    # ------------------------------------------------------------------
    # TRACING
    # ------------------------------------------------------------------
    trace_db_path: Optional[str] = None
    integration_profile: Optional["IntegrationProfile"] = None

    # Optional sync bus for planner/context events (Phase Q+-N.5, R-Context.2).
    runtime_event_bus: Optional["RuntimeEventBus"] = None

    # Tier-3 composed policy (Phase R-Policy); set via applications runtime_config_bridge.
    policy_bundle: Optional["RuntimePolicyBundle"] = None


    # ------------------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------------------
    production_mode: bool = True
    security_profile: Optional["ApplicationSecurityProfile"] = None

    prompt_catalog_path: Optional[str] = None

    # ------------------------------------------------------------------
    # COMPOSED SECTIONS (Phase Q-N.8)
    # ------------------------------------------------------------------

    @property
    def model_section(self) -> ModelRuntimeConfig:
        return ModelRuntimeConfig(
            llm_adapter=self.llm_adapter,
            enable_llm_usage_collection=self.enable_llm_usage_collection,
        )

    @property
    def retrieval_section(self) -> RetrievalRuntimeConfig:
        return RetrievalRuntimeConfig(
            embedding_manager=self.embedding_manager,
            vectorstore_manager=self.vectorstore_manager,
            retriever_manager=self.retriever_manager,
            reranker_manager=self.reranker_manager,
            rag_profile=self.rag_profile,
            retrieval_service=self.retrieval_service,
            enable_rag=self.enable_rag,
            max_docs_per_query=self.max_docs_per_query,
            max_rag_tokens=self.max_rag_tokens,
            rag_score_threshold=self.rag_score_threshold,
            max_longterm_entries_per_query=self.max_longterm_entries_per_query,
            max_longterm_tokens=self.max_longterm_tokens,
            longterm_score_threshold=self.longterm_score_threshold,
            enable_user_profile_memory=self.enable_user_profile_memory,
            enable_org_profile_memory=self.enable_org_profile_memory,
            enable_user_longterm_memory=self.enable_user_longterm_memory,
        )

    @property
    def tools_section(self) -> ToolsRuntimeConfig:
        return ToolsRuntimeConfig(
            websearch_executor=self.websearch_executor,
            websearch_config=self.websearch_config,
            enable_websearch=self.enable_websearch,
            tool_planner=self.tool_planner,
            tools_mode=self.tools_mode,
            tools_context_scope=self.tools_context_scope,
            tool_invoker=self.tool_invoker,
            idempotency_store=self.idempotency_store,
            tool_providers=tuple(self.tool_providers),
            tool_profile=self.tool_profile,
            tool_wiring_context=self.tool_wiring_context,
            tool_scope_policy=self.tool_scope_policy,
            modality_profile=self.modality_profile,
        )

    @property
    def planning_section(self) -> PlanningRuntimeConfig:
        return PlanningRuntimeConfig(
            pipeline=self.pipeline,
            step_planner_cfg=self.step_planner_cfg,
            step_executor_cfg=self.step_executor_cfg,
            planner_prompt_config=self.planner_prompt_config,
            plan_loop_policy=self.plan_loop_policy,
            plan_source=self.plan_source,
        )

    @property
    def trace_section(self) -> TraceRuntimeConfig:
        return TraceRuntimeConfig(
            trace_db_path=self.trace_db_path,
            integration_profile=self.integration_profile,
            runtime_timeout_ms=self.runtime_timeout_ms,
            max_run_retries=self.max_run_retries,
            retry_run_on=self.retry_run_on,
            runtime_policies=self.runtime_policies,
            execution_slot_warn_threshold_ms=self.execution_slot_warn_threshold_ms,
            hitl_default_message=self.hitl_default_message,
            production_mode=self.production_mode,
        )

    def ensure_rag_profile(self) -> RagProfile:
        """Single RAG config surface: RuntimeConfig fields → RagProfile."""
        profile = sync_rag_profile_from_runtime_config(self)
        self.rag_profile = profile
        return profile

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Validates config consistency. Keeps the runtime fail-fast and predictable.
        """

        if self.runtime_timeout_ms is not None:
            if not isinstance(self.runtime_timeout_ms, int):
                raise TypeError("runtime_timeout_ms must be an int or None.")
            
            if self.runtime_timeout_ms<=0:
                raise ValueError("runtime_timeout_ms must be > 0 when provided.")
            
        
        if not isinstance(self.max_run_retries, int):
            raise TypeError("max_run_retries must be an int.")
        
        if self.max_run_retries < 0:
            raise ValueError("max_run_retries must be >= 0.")

        if not isinstance(self.retry_run_on, frozenset):
            raise TypeError("retry_run_on must be a frozenset[RuntimeErrorCode].")

        for code in self.retry_run_on:
            if not isinstance(code, RuntimeErrorCode):
                raise TypeError("retry_run_on must contain RuntimeErrorCode items only.")


        if self.pipeline is not None and not isinstance(self.pipeline, RuntimePipeline):
            raise TypeError("pipeline must be an instance of RuntimePipeline.")
        
        if self.enable_rag:
            if self.embedding_manager is None or self.vectorstore_manager is None:
                raise ValueError(
                    "enable_rag=True requires embedding_manager and vectorstore_manager."
                )
            self.ensure_rag_profile()
            
        if self.run_budget is not None:
            if not isinstance(self.run_budget, RunBudget):
                raise TypeError("run_budget must be RunBudget or None.")
            self.run_budget.validate()

            if self.budget_policy is None:
                raise ValueError("budget_policy must be provided when run_budget is set.")

        if self.budget_policy is not None:
            if not isinstance(self.budget_policy, BudgetPolicy):
                raise TypeError("budget_policy must be BudgetPolicy or None.")
        
        
        if self.execution_slot_warn_threshold_ms is not None:
            if not isinstance(self.execution_slot_warn_threshold_ms, int):
                raise TypeError("execution_slot_warn_threshold_ms must be int or None.")
            if self.execution_slot_warn_threshold_ms < 0:
                raise ValueError("execution_slot_warn_threshold_ms must be >= 0.")
