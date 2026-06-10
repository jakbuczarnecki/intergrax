# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composed runtime config sections (Phase Q-N.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
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
from intergrax.tools.registry import ToolProfile, ToolWiringContext
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.websearch.service.websearch_config import WebSearchConfig
from intergrax.websearch.service.websearch_executor import WebSearchExecutor

from intergrax.runtime.nexus.config_types import ToolChoiceMode, ToolSelectionMode, ToolsContextScope


@dataclass
class ModelRuntimeConfig:
    llm_adapter: LLMAdapter
    enable_llm_usage_collection: bool = True


@dataclass
class RetrievalRuntimeConfig:
    embedding_manager: Optional[BaseEmbeddingManager] = None
    vectorstore_manager: Optional[BaseVectorstoreManager] = None
    retriever_manager: Optional[BaseRetrieverManager] = None
    reranker_manager: Optional[BaseRerankerManager] = None
    rag_profile: Optional[RagProfile] = None
    retrieval_service: Optional[RetrievalService] = None
    enable_rag: bool = True
    max_docs_per_query: int = 8
    max_rag_tokens: int = 4096
    rag_score_threshold: Optional[float] = None
    max_longterm_entries_per_query: int = 8
    max_longterm_tokens: int = 4096
    longterm_score_threshold: Optional[float] = None
    enable_user_profile_memory: bool = True
    enable_org_profile_memory: bool = True
    enable_user_longterm_memory: bool = True


@dataclass
class ToolsRuntimeConfig:
    websearch_executor: Optional[WebSearchExecutor] = None
    websearch_config: Optional[WebSearchConfig] = None
    enable_websearch: bool = True
    tool_planner: Optional[ToolPlannerProtocol] = None
    tools_mode: ToolChoiceMode = "auto"
    tools_context_scope: ToolsContextScope = ToolsContextScope.CURRENT_MESSAGE_ONLY
    tool_selection_mode: ToolSelectionMode = ToolSelectionMode.STATIC
    tool_selection_top_k: int = 20
    tool_invoker: Optional[RuntimeToolInvoker] = None
    idempotency_store: Optional[IdempotencyStore] = None
    tool_providers: tuple[ToolProvider, ...] = ()
    tool_profile: Optional[ToolProfile] = None
    tool_wiring_context: Optional[ToolWiringContext] = None
    tool_scope_policy: Optional[ToolScopePolicy] = None
    modality_profile: ModalityProfile | None = None


@dataclass
class PlanningRuntimeConfig:
    pipeline: Optional[RuntimePipeline] = None
    step_planner_cfg: Optional[StepPlannerConfig] = None
    step_executor_cfg: Optional[StepExecutorConfig] = None
    planner_prompt_config: Optional[PlannerPromptConfig] = None
    plan_loop_policy: Optional[PlanLoopPolicy] = None
    plan_source: Optional[PlanSource] = None


@dataclass
class TraceRuntimeConfig:
    trace_db_path: Optional[str] = None
    integration_profile: Optional[IntegrationProfile] = None
    runtime_timeout_ms: Optional[int] = None
    max_run_retries: int = 0
    retry_run_on: FrozenSet[RuntimeErrorCode] = frozenset(
        {RuntimeErrorCode.LLM_ERROR, RuntimeErrorCode.TOOL_ERROR}
    )
    runtime_policies: RuntimePolicies = RuntimePolicies()
    execution_slot_warn_threshold_ms: Optional[int] = None
    hitl_default_message: Optional[str] = None
    production_mode: bool = True
