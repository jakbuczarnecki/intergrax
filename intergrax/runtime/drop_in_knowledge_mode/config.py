# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

from intergrax.llm_adapters.base import LLMAdapter
from intergrax.rag.embedding_manager import IntergraxEmbeddingManager
from intergrax.rag.vectorstore_manager import IntergraxVectorstoreManager
from intergrax.tools.tools_agent import IntergraxToolsAgent
from intergrax.websearch.service.websearch_executor import WebSearchExecutor


# Defines how the runtime should interact with tools.
# - "off": tools are never used, even if a tools_agent is provided.
# - "auto": runtime may decide to call tools when appropriate.
# - "required": runtime must use tools to answer the request.
ToolChoiceMode = Literal["off", "auto", "required"]


@dataclass
class RuntimeConfig:
    """
    Global configuration object for the Drop-In Knowledge Runtime.

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
    embedding_manager: IntergraxEmbeddingManager

    # Vectorstore manager providing semantic search over stored chunks.
    vectorstore_manager: IntergraxVectorstoreManager

    # Optional labels for observability/logging only.
    llm_label: str = "default-llm"
    embedding_label: str = "default-embedding"
    vectorstore_label: str = "default-vectorstore"

    # ------------------------------------------------------------------
    # FEATURE FLAGS
    # ------------------------------------------------------------------

    # Enables Retrieval-Augmented Generation based on stored documents.
    enable_rag: bool = True

    # Enables real-time web search as an additional context layer.
    enable_websearch: bool = True

    # Enables long-term memory (not yet implemented).
    enable_long_term_memory: bool = False

    # Enables short-term user profile / conversational memory.
    enable_user_profile_memory: bool = True

    # ------------------------------------------------------------------
    # MULTI-TENANCY
    # ------------------------------------------------------------------

    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # ------------------------------------------------------------------
    # CHAT HISTORY LIMITS
    # ------------------------------------------------------------------

    max_history_messages: int = 20
    max_history_tokens: int = 4096

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
    # WEB SEARCH CONFIGURATION
    # ------------------------------------------------------------------

    # Pre-configured executor capable of performing web search queries.
    # If None, web search is effectively unavailable.
    websearch_executor: Optional[WebSearchExecutor] = None

    # ------------------------------------------------------------------
    # TOOLS / AGENT EXECUTION
    # ------------------------------------------------------------------

    # Optional tools agent responsible for:
    #   - planning tool calls,
    #   - invoking tools,
    #   - merging tool results into the final answer.
    #
    # If None, tools cannot be used regardless of tools_mode.
    tools_agent: Optional[IntergraxToolsAgent] = None

    # High-level policy defining whether tools may or must be used:
    #   - "off": do not use tools at all.
    #   - "auto": runtime may call tools if useful.
    #   - "required": runtime must use at least one tool.
    tools_mode: ToolChoiceMode = "auto"

    # ------------------------------------------------------------------
    # TOKEN LIMITS
    # ------------------------------------------------------------------

    max_total_tokens: int = 8192
    max_output_tokens: int = 2048

    # ------------------------------------------------------------------
    # MISC METADATA
    # ------------------------------------------------------------------

    # Arbitrary metadata for app-specific instrumentation or tags.
    metadata: Dict[str, Any] = field(default_factory=dict)
