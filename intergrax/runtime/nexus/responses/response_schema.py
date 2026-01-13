# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Request and response data models for the nexus Mode runtime.

These dataclasses define the high-level contract between applications
(FastAPI, Streamlit, CLI, MCP, etc.) and the RuntimeEngine.

They intentionally hide low-level implementation details while keeping
enough structure to expose citations, routing information, tool calls,
and basic statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from intergrax.llm.messages import AttachmentRef
from intergrax.llm_adapters.llm_usage_track import LLMUsageReport
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent


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
    used_user_profile: bool = False
    used_user_longterm_memory: bool = False
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


class HistoryCompressionStrategy(Enum):
    """
    Strategy for compressing the conversation history before sending it
    to the LLM.

    - OFF
        Do not modify or compress history at all.
        (Risk: context window overflow for very long conversations.)

    - TRUNCATE_OLDEST:
        Drop the oldest messages until the history fits into the budget.
    - SUMMARIZE_OLDEST:
        Summarize the oldest portion of the history into a compact
        synthetic message and keep more recent turns verbatim.
    - HYBRID:
        Combine truncation and summarization, e.g. truncate very old noise
        and summarize the remaining older block.
    """

    OFF = "off"
    TRUNCATE_OLDEST = "truncate_oldest"
    SUMMARIZE_OLDEST = "summarize_oldest"
    HYBRID = "hybrid"
    

@dataclass
class RuntimeRequest:
    """
    High-level request structure for the nexus runtime.

    This object is built by the application layer and passed into the
    RuntimeEngine. It can be created directly or via helper
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
    High-level response structure returned by the nexus runtime.

    This contains the final answer, along with citations, routing info,
    tool call summaries, and basic statistics.
    """

    answer: str
    run_id: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    route: RouteInfo = field(default_factory=RouteInfo)
    tool_calls: List[ToolCallInfo] = field(default_factory=list)
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    llm_usage_report: Optional[LLMUsageReport] = None

    # Optional raw model output or intermediate artifacts
    raw_model_output: Optional[Any] = None
    trace_events: List[TraceEvent] = field(default_factory=list)
