# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Versioned public API schemas (v1) for Legal Agent HTTP layer."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


API_VERSION: Literal["1"] = "1"


class AttachmentRefV1(BaseModel):
    """Maps to :class:`~intergrax.llm.messages.AttachmentRef`."""

    id: str
    type: str
    uri: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LegalChatRequestV1(BaseModel):
    """
    Product chat request — maps to :class:`~intergrax.runtime.nexus.responses.response_schema.RuntimeRequest`.

    Identity: ``tenant_id`` / ``user_id`` may come from auth (``RequestContext``); body fields override when set.
    """

    message: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    workspace_id: Optional[str] = None
    tenant_id: Optional[str] = Field(
        default=None,
        description="Overrides RequestContext.tenant_id when your gateway does not set auth context.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Overrides RequestContext.user_id when needed.",
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Registered Tier-2 agent id; defaults to server configured default.",
    )
    attachments: List[AttachmentRefV1] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    instructions: Optional[str] = None
    history_compression: Optional[str] = Field(
        default=None,
        description="Enum name e.g. TRUNCATE_OLDEST; default runtime default applies when omitted.",
    )
    max_output_tokens: Optional[int] = None
    include_trace: bool = Field(
        default=False,
        description="If true, include trace_events (may be large; trust gateway in prod).",
    )


class LegalChatResponseV1(BaseModel):
    """Stable JSON shape returned to API clients (subset of runtime answer)."""

    api_version: Literal["1"] = "1"
    request_id: str = Field(description="Correlation id from X-Request-ID / RequestContext.")
    run_id: Optional[str] = None
    stop_reason: str
    answer: str
    route: Dict[str, Any]
    stats: Dict[str, Any]
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    llm_usage: Optional[Dict[str, Any]] = None
    trace_events: Optional[List[Dict[str, Any]]] = None
