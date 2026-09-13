# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.tracing import (
    DiagnosticPayload,
    TraceArtifactRef,
    TraceEvent,
)
from intergrax.runtime.nexus.tracing.trace_models import (
    ArtifactRef,
    TraceEvent as RuntimeTraceEvent,
)

pytestmark = pytest.mark.unit


def test_trace_event_is_single_canonical_class() -> None:
    assert TraceEvent is RuntimeTraceEvent


def test_nexus_artifact_ref_aliases_contract() -> None:
    assert ArtifactRef is TraceArtifactRef


def test_diagnostic_payload_reexport_matches_contract() -> None:
    from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload as RuntimeDiag

    assert DiagnosticPayload is RuntimeDiag
