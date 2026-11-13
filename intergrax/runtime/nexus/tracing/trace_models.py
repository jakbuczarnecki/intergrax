# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Runtime compatibility re-exports for Plane B trace contracts.

Canonical definitions live in ``intergrax.contracts.tracing``. New application,
agent, and scenario code should import from that package.
"""

from __future__ import annotations

from typing import Any

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

__all__ = [
    "DEFAULT_REDACTED_TEXT",
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


def __getattr__(name: str) -> Any:
    """COMPAT-X shim: ``ArtifactRef`` was an alias of ``TraceArtifactRef`` (PDC-D8-R018).

    The alias is no longer bound at module level, so the only ``ArtifactRef`` is the artifact
    reference contract in ``intergrax.contracts.artifact_ref``. Existing imports keep resolving.
    """
    if name == "ArtifactRef":
        return TraceArtifactRef
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
