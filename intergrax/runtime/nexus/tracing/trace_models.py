# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Runtime compatibility re-exports for Plane B trace contracts.

Canonical definitions live in ``intergrax.contracts.tracing``. New application,
agent, and scenario code should import from that package.
"""

from __future__ import annotations

from intergrax.contracts.tracing import (
    DEFAULT_REDACTED_TEXT,
    DiagnosticPayload,
    ToolCallTrace,
    TraceArtifactRef,
    TraceComponent,
    TraceEvent,
    TraceLevel,
)
from intergrax.utils.time_provider import SystemTimeProvider

# Historical import path for nexus artifact embedding in traces.
ArtifactRef = TraceArtifactRef

__all__ = [
    "DEFAULT_REDACTED_TEXT",
    "ArtifactRef",
    "DiagnosticPayload",
    "ToolCallTrace",
    "TraceArtifactRef",
    "TraceComponent",
    "TraceEvent",
    "TraceLevel",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    return SystemTimeProvider.utc_now().isoformat()
