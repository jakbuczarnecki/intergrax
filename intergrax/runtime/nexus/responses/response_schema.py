# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Request and response data models for the nexus Mode runtime.

These dataclasses define the high-level contract between applications
(FastAPI, Streamlit, CLI, MCP, etc.) and AgentEngine.

They intentionally hide low-level implementation details while keeping
enough structure to expose citations, routing information, tool calls,
and basic statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
from intergrax.runtime.task.task_contract import HumanApprovalResolution, TaskPauseRecord
from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.llm.messages import AttachmentRef
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageReport
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


class StopReason(Enum):
    COMPLETED = "completed"
    NEEDS_USER_INPUT = "needs_user_input"
    ABORTED = "aborted"
    ERROR = "error"

    
@dataclass
class RuntimeRequest:
    """
    High-level request structure for the nexus runtime.

    This object is built by the application layer and passed into the
    AgentEngine. It can be created directly or via helper
    functions/wrappers in web frameworks.
    """

    agent_id: str

    user_id: str
    session_id: str
    message: str

    task_id: TaskId
    run_id: RunId

    attachments: List[AttachmentRef] = field(default_factory=list)

    # Workspace scoping    
    workspace_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Optional UI / app metadata (channel, app name, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Typed generic HITL resume authority (canonical HumanApprovalResolution only).
    hitl_resolution: Optional[HumanApprovalResolution] = None

    # Verified authenticated principal for execution identity (IDT-FIX-A).
    canonical_identity: Optional[RequestIdentity] = None

    # Trusted root execution authority minted by host/runtime policy (IDT-FIX-B R1).
    execution_authority: Optional[ParentExecutionAuthority] = None

    # Typed effective delegation authority for delegated child execution (IDT-FIX-B R1).
    effective_delegation_authority: Optional[EffectiveDelegationAuthority] = None

    # Typed pause lifecycle identity for resume validation (not authorization).
    hitl_pause_record: Optional[TaskPauseRecord] = None

    # Typed declarative HITL grant transport (approval evidence only).
    declarative_hitl_grant: Optional[DeclarativeHitlApprovalGrant] = None

    def __post_init__(self) -> None:
        self.task_id = validate_task_id(self.task_id)
        self.run_id = validate_run_id(self.run_id)

    def to_envelope(self) -> TaskEnvelope:
        tenant = self.tenant_id or self.metadata.get("tenant_id") or "default"
        return TaskEnvelope(
            tenant_id=str(tenant),
            user_id=self.user_id,
            message=self.message,
            session_id=self.session_id,
            agent_id=self.agent_id,
            workspace_id=self.workspace_id,
            metadata=dict(self.metadata),
            canonical_identity=self.canonical_identity,
        )

    @classmethod
    def from_envelope(
        cls,
        envelope: TaskEnvelope,
        *,
        task_id: TaskId,
        run_id: RunId,
    ) -> RuntimeRequest:
        """Materialize execute-boundary runtime request from intake envelope."""
        return cls(
            agent_id=envelope.agent_id or "",
            user_id=envelope.user_id,
            session_id=envelope.session_id or f"sess_{envelope.tenant_id}",
            message=envelope.message,
            task_id=task_id,
            run_id=run_id,
            workspace_id=envelope.workspace_id,
            tenant_id=envelope.tenant_id,
            metadata=dict(envelope.metadata),
            canonical_identity=envelope.canonical_identity,
        )

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
    stop_reason: StopReason = StopReason.COMPLETED
    run_id: Optional[str] = None
    citations: List[Citation] = field(default_factory=list)
    route: RouteInfo = field(default_factory=RouteInfo)
    tool_calls: List[ToolCallInfo] = field(default_factory=list)
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    llm_usage_report: Optional[LLMUsageReport] = None

    # Optional raw model output or intermediate artifacts
    raw_model_output: Optional[Any] = None
    trace_events: List[TraceEvent] = field(default_factory=list)
