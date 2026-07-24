# © Artur Czarnecki. All rights reserved.

"""Typed product models for the LKW Slack companion."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SlackDedupeStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SlackDedupeRecord(BaseModel):
    """Persisted product dedupe claim for one Slack Events API identity."""

    model_config = ConfigDict(extra="forbid")

    dedupe_key: str
    status: SlackDedupeStatus
    claim_token: str
    first_seen_at: datetime
    updated_at: datetime
    expires_at: datetime
    ask_run_id: str | None = None


class AuthorizedSlackAskContext(BaseModel):
    """Fail-closed authorization + configured tenant/workspace mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team_id: str
    user_id: str
    tenant_id: str
    workspace_id: str
    question: str
    event_id: str


class SlackAskHttpStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class SlackAskCitationLabel(BaseModel):
    """Only ``file_name`` is retained for Slack rendering; other Ask fields ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    file_name: str = ""


class SlackAskHttpResponse(BaseModel):
    """Typed Ask Workspace HTTP response used by the companion (safe subset)."""

    model_config = ConfigDict(extra="ignore")

    run_id: str = ""
    workspace_id: str = ""
    status: Literal["completed", "insufficient_evidence", "failed"]
    question: str = ""
    answer: str | None = None
    citations: list[SlackAskCitationLabel] = Field(default_factory=list)


class SlackWorkspaceListItem(BaseModel):
    """Safe subset of Managed Workspace list items for Slack rendering."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    workspace_id: str = ""
    name: str = ""
    status: str = ""


class SlackWorkspaceListResponse(BaseModel):
    """Typed GET /workspaces response used by the companion (safe subset)."""

    model_config = ConfigDict(extra="ignore")

    workspaces: list[SlackWorkspaceListItem] = Field(default_factory=list)


class SlackWorkspaceCreateResponse(BaseModel):
    """Typed POST /workspaces create response (safe subset for selection)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    workspace_id: str = ""
    name: str = ""
    status: str = ""


class SlackAskClientError(Exception):
    """Controlled Ask/list HTTP failure (no internal diagnostics for Slack rendering)."""

    def __init__(self, *, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind
