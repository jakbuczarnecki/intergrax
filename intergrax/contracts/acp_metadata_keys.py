# © Artur Czarnecki. All rights reserved.

"""Typed metadata keys for ACP session and orchestration planes."""

from __future__ import annotations

from enum import StrEnum


class AcpMetadataKey(StrEnum):
    """ACP-related Task / RuntimeRequest metadata keys — no ad-hoc strings."""

    SESSION_ENABLED = "acp.session.v1"
    HOST_CONTEXT = "acp.host.v1"
    APPLICATION_RUN_SUMMARY = "application_run_summary.v1"


class AcpStructuredDataKey(StrEnum):
    """Keys inside ``AgentExecutionResult.structured_data`` for ACP payloads."""

    TRACE_SUMMARY = "acp.trace.v1"
