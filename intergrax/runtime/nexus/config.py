# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Literal

from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.rag.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore_manager import VectorstoreManager
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.planning.engine_plan_models import PlannerPromptConfig
from intergrax.runtime.nexus.planning.plan_loop_models import PlanLoopPolicy
from intergrax.runtime.nexus.planning.plan_sources import PlanSource
from intergrax.runtime.nexus.planning.step_executor_models import StepExecutorConfig
from intergrax.runtime.nexus.planning.step_planner import StepPlannerConfig
from intergrax.tools.tools_agent import ToolsAgent
from intergrax.websearch.service.websearch_config import WebSearchConfig
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


# Defines how the runtime should interact with tools.
# - "off": tools are never used, even if a tools_agent is provided.
# - "auto": runtime may decide to call tools when appropriate.
# - "required": runtime must use tools to answer the request.
ToolChoiceMode = Literal["off", "auto", "required"]


class ToolsContextScope(str, Enum):
    CURRENT_MESSAGE_ONLY = "current_message_only"
    
    CONVERSATION = "conversation"
    
    FULL = "full"


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
    embedding_manager: Optional[EmbeddingManager] = None

    # Vectorstore manager providing semantic search over stored chunks.
    vectorstore_manager: Optional[VectorstoreManager] = None

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
    tools_agent: Optional[ToolsAgent] = None

    # High-level policy defining whether tools may or must be used:
    #   - "off": do not use tools at all.
    #   - "auto": runtime may call tools if useful.
    #   - "required": runtime must use at least one tool.
    tools_mode: ToolChoiceMode = "auto"

    # Determines how much contextual information the tools agent receives:
    #
    #   - "current_message_only":
    #       ToolsAgent sees only the newest user query.
    #       Useful for strict function-calling, cost optimization
    #       and predictable single-turn behavior.
    #
    #   - "conversation":
    #       ToolsAgent sees full conversation history up to this point.
    #
    #   - "full":
    #       ToolsAgent receives the same context as the LLM:
    #       system → profile → history → RAG → websearch.
    #
    tools_context_scope: ToolsContextScope = ToolsContextScope.CURRENT_MESSAGE_ONLY


    # Memory toggles
    enable_user_profile_memory: bool = True
    enable_org_profile_memory: bool = True
    enable_user_longterm_memory: bool = True

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
    # VALIDATION
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Validates config consistency. Keeps the runtime fail-fast and predictable.
        """

        if self.pipeline is not None and not isinstance(self.pipeline, RuntimePipeline):
            raise TypeError("pipeline must be an instance of RuntimePipeline.")
        
        if self.enable_rag:
            if self.embedding_manager is None or self.vectorstore_manager is None:
                raise ValueError(
                    "enable_rag=True requires embedding_manager and vectorstore_manager."
                )