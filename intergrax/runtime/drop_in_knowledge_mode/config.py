# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Configuration objects for the Drop-In Knowledge Mode runtime.

These dataclasses define the main knobs for controlling how the runtime
behaves: which LLM adapter to use, which embedding and vector store managers,
which features are enabled (RAG, web search, tools, long-term memory),
and token budgets for context construction.

The goal is to integrate tightly with existing Intergrax components:
- LLM adapters from `intergrax.llm_adapters`
- IntergraxEmbeddingManager from `intergrax.rag.embedding_manager`
- IntergraxVectorstoreManager from `intergrax.rag.vectorstore_manager`
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Type

from intergrax.llm_adapters.base import LLMAdapter
from intergrax.rag.embedding_manager import IntergraxEmbeddingManager
from intergrax.rag.vectorstore_manager import IntergraxVectorstoreManager
from intergrax.websearch.providers.base import WebSearchProvider
from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
from intergrax.websearch.providers.bing_provider import BingWebProvider


@dataclass
class RuntimeConfig:
    """
    High-level configuration for the Drop-In Knowledge Mode runtime.

    Applications should construct this object once and pass it to the
    DropInKnowledgeRuntime. All low-level components (LLM adapters,
    vector stores, memory stores, tools, etc.) are wired based on this
    configuration.

    The design is:
      - strongly typed: runtime receives concrete adapter/manager instances,
      - model-agnostic: any LLMAdapter implementation can be used,
      - store-agnostic: any IntergraxVectorstoreManager / EmbeddingManager
        configuration is accepted.
    """

    # Core model / LLM configuration: concrete adapter instance
    llm_adapter: LLMAdapter

    # Embeddings and vector store: concrete manager instances used by RAG
    embedding_manager: IntergraxEmbeddingManager
    vectorstore_manager: IntergraxVectorstoreManager

    # Human-readable identifiers (optional, for logging/observability only)
    llm_label: str = "default-llm"
    embedding_label: str = "default-embedding"
    vectorstore_label: str = "default-vectorstore"

    # Feature toggles
    enable_rag: bool = True
    enable_websearch: bool = True
    enable_tools: bool = True
    enable_long_term_memory: bool = False
    enable_user_profile_memory: bool = True

    # Scoping and multi-tenancy
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Session / history parameters
    max_history_messages: int = 20
    max_history_tokens: int = 4096

    # RAG / retrieval parameters
    max_docs_per_query: int = 8
    max_rag_tokens: int = 4096
    rag_score_threshold: Optional[float] = None

    # Web search providers configuration (provider classes to be instantiated
    # by the WebSearchExecutor). Use provider classes that inherit from
    # `intergrax.websearch.providers.base.WebSearchProvider`.
    websearch_providers: Tuple[Type[WebSearchProvider], ...] = (
        GoogleCSEProvider,
        BingWebProvider,
    )

    # Tools / data sources: names understood by ToolRegistry / app-level config
    enabled_tools: Sequence[str] = field(default_factory=list)
    enabled_data_sources: Sequence[str] = field(default_factory=list)

    # Token budgets and limits for the whole runtime call
    max_total_tokens: int = 8192
    max_output_tokens: int = 2048

    # Optional arbitrary metadata / app-specific config
    metadata: Dict[str, Any] = field(default_factory=dict)
