# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Request and response data models for the Drop-In Knowledge Mode runtime.

These dataclasses define the high-level contract between applications
(FastAPI, Streamlit, CLI, MCP, etc.) and the DropInKnowledgeRuntime.

They intentionally hide low-level implementation details while keeping
enough structure to expose citations, routing information, tool calls,
and basic statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from intergrax.llm.messages import AttachmentRef


@dataclass
class Citation:
    """
    Represents a single citation/reference used in the final answer.

    This can point to:
      - a document chunk in a vector store,
      - a specific file and location,
      - a web page,
      - an internal knowledge base entry.
    """

    source_id: str
    source_type: str  # e.g. "vectorstore", "file", "web", "db"
    source_label: Optional[str] = None  # human-readable label
    url: Optional[str] = None
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteInfo:
    """
    Describes how the runtime decided to answer the question.

    Useful for debugging, observability, and UI explanations.
    """

    used_rag: bool = False
    used_websearch: bool = False
    used_tools: bool = False
    used_long_term_memory: bool = False
    used_user_profile: bool = False
    strategy: Optional[str] = None  # e.g. "simple", "agentic", "fallback_websearch"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallInfo:
    """
    Describes a single tool call executed during the runtime request.
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result_summary: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeStats:
    """
    Basic statistics about a runtime call.

    This is intentionally simple and can be extended over time.
    """

    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    rag_tokens: Optional[int] = None
    websearch_tokens: Optional[int] = None
    tool_tokens: Optional[int] = None
    duration_ms: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class HistoryCompressionStrategy(str, Enum):
    """
    High-level strategy describing how the runtime should reduce
    conversation history when it no longer fits into the model context.
    """

    # Do not modify or compress history at all.
    # (Risk: context window overflow for very long conversations.)
    OFF = "off"

    # Drop the oldest messages first until the history fits within
    # the computed token budget.
    TRUNCATE_OLDEST = "truncate_oldest"

    # NOTE:
    # Additional strategies like SUMMARIZE_OLDEST / HYBRID can be added
    # later when summarization is implemented.
    

@dataclass
class RuntimeRequest:
    """
    High-level request structure for the Drop-In Knowledge runtime.

    This object is built by the application layer and passed into the
    DropInKnowledgeRuntime. It can be created directly or via helper
    functions/wrappers in web frameworks.
    """

    user_id: str
    session_id: str
    message: str

    attachments: List[AttachmentRef] = field(default_factory=list)

    # Optional tenant/workspace scoping
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Optional UI / app metadata (channel, app name, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # User-provided instructions (ChatGPT/Gemini-style)
    instructions: Optional[str] = None


    # Strategy used to keep the conversation history within the model
    # context window for THIS request.
    #
    # If you don't specify anything when constructing the request,
    # TRUNCATE_OLDEST will be used as a reasonable default.
    history_compression_strategy: HistoryCompressionStrategy = HistoryCompressionStrategy.TRUNCATE_OLDEST

    # Maximum number of output tokens for a single model response
    # for THIS request.
    #
    # If None, the runtime/adapter will use its own internal default.
    #
    # NOTE:
    # This is *not* the context window size. The maximum context window
    # is defined by the underlying LLM adapter (context_window_tokens).
    max_output_tokens: Optional[int] = None


@dataclass
class RuntimeAnswer:
    """
    High-level response structure returned by the Drop-In Knowledge runtime.

    This contains the final answer, along with citations, routing info,
    tool call summaries, and basic statistics.
    """

    answer: str
    citations: List[Citation] = field(default_factory=list)
    route: RouteInfo = field(default_factory=RouteInfo)
    tool_calls: List[ToolCallInfo] = field(default_factory=list)
    stats: RuntimeStats = field(default_factory=RuntimeStats)

    # Optional raw model output or intermediate artifacts
    raw_model_output: Optional[Any] = None
    debug_trace: Dict[str, Any] = field(default_factory=dict)
