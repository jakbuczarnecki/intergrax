# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.ingestion import IngestionResult
from intergrax.runtime.drop_in_knowledge_mode.response_schema import RuntimeRequest
from intergrax.runtime.drop_in_knowledge_mode.session_store import ChatSession


@dataclass
class RuntimeState:
    """
    Mutable state object passed through the runtime pipeline.

    It aggregates:
      - request and session metadata,
      - ingestion results,
      - conversation history and model-ready messages,
      - flags indicating which subsystems were used (RAG, websearch, tools, memory),
      - tools traces and agent answer,
      - full debug_trace for observability & diagnostics.
    """

    # Input
    request: RuntimeRequest

    # Session and ingestion
    session: Optional[ChatSession] = None
    ingestion_results: List[IngestionResult] = field(default_factory=list)

    # Conversation / context
    base_history: List[ChatMessage] = field(default_factory=list)
    messages_for_llm: List[ChatMessage] = field(default_factory=list)
    tools_context_parts: List[str] = field(default_factory=list)
    built_history_messages: List[ChatMessage] = field(default_factory=list)
    history_includes_current_user: bool = False

    # ContextBuilder intermediate result (history + retrieved chunks)
    context_builder_result: Optional[Any] = None

    # Memory layer (will be filled by _step_memory_layer)
    user_memory_messages: List[ChatMessage] = field(default_factory=list)
    org_memory_messages: List[ChatMessage] = field(default_factory=list)
    ltm_memory_messages: List[ChatMessage] = field(default_factory=list)

    # Profile-based instruction fragments prepared by the memory layer
    profile_user_instructions: Optional[str] = None
    profile_org_instructions: Optional[str] = None

    # Usage flags
    used_rag: bool = False
    used_websearch: bool = False
    used_tools: bool = False
    used_long_term_memory: bool = False  # reserved for future integration
    used_user_profile: bool = False      # reserved for future integration

    # Tools
    tools_agent_answer: Optional[str] = None
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)

    # Debug / diagnostics
    debug_trace: Dict[str, Any] = field(default_factory=dict)
    websearch_debug: Dict[str, Any] = field(default_factory=dict)

    # Token accounting (filled in _step_build_base_history)
    history_token_count: Optional[int] = None