# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Harness wiring for functional evidence persistence (contract-first composition)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.observability.functional_evidence.document_store_functional_evidence_persistence import (
    build_document_store_functional_evidence_persistence,
)
from intergrax.runtime.observability.functional_evidence.in_memory_functional_evidence_persistence import (
    build_in_memory_functional_evidence_persistence,
)
from intergrax.runtime.observability.functional_evidence_recorder import (
    attach_functional_evidence_recorder,
    FunctionalEvidenceRecorder,
)

if TYPE_CHECKING:
    from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
    from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceRuntimeWiring:
    persistence: FunctionalEvidencePersistence
    recorder: FunctionalEvidenceRecorder


def wire_functional_evidence_runtime(
    *,
    persistence: FunctionalEvidencePersistence | None = None,
    cursor_secret: str | bytes | None = None,
    document_store: DocumentStore | None = None,
    producer_component: str = "agents.local_search",
) -> FunctionalEvidenceRuntimeWiring:
    """
    Compose functional evidence recorder over an explicit persistence port.

    When ``persistence`` is omitted, built-in providers are selected by composition
    inputs (``document_store`` → durable adapter, else in-memory for tests/dev).
    """
    resolved_persistence = persistence
    if resolved_persistence is None:
        if cursor_secret is None:
            raise ValueError("wire_functional_evidence_runtime requires cursor_secret or persistence")
        if document_store is not None:
            resolved_persistence = build_document_store_functional_evidence_persistence(
                document_store=document_store,
                cursor_secret=cursor_secret,
            )
        else:
            resolved_persistence = build_in_memory_functional_evidence_persistence(
                cursor_secret=cursor_secret,
            )
    recorder = FunctionalEvidenceRecorder(
        resolved_persistence,
        producer_component=producer_component,
    )
    return FunctionalEvidenceRuntimeWiring(persistence=resolved_persistence, recorder=recorder)


def wire_in_memory_functional_evidence_runtime(
    *,
    cursor_secret: str | bytes,
    producer_component: str = "agents.local_search",
) -> FunctionalEvidenceRuntimeWiring:
    """Explicit in-memory functional evidence wiring for tests and conformance."""
    return wire_functional_evidence_runtime(
        cursor_secret=cursor_secret,
        producer_component=producer_component,
    )


def functional_evidence_wiring_extra_key() -> str:
    return "functional_evidence_wiring"


def attach_functional_evidence_recorder_from_tool_wiring(
    exec_ctx: RuntimeExecutionContext,
    tool_wiring_context: ToolWiringContext | None,
) -> None:
    """Attach recorder using explicitly composed tool wiring (no runtime object graph discovery)."""
    if tool_wiring_context is None:
        return
    wiring = tool_wiring_context.extras.get(functional_evidence_wiring_extra_key())
    if not isinstance(wiring, FunctionalEvidenceRuntimeWiring):
        return
    attach_functional_evidence_recorder(exec_ctx, wiring.recorder)


__all__ = [
    "FunctionalEvidenceRuntimeWiring",
    "attach_functional_evidence_recorder_from_tool_wiring",
    "functional_evidence_wiring_extra_key",
    "wire_functional_evidence_runtime",
    "wire_in_memory_functional_evidence_runtime",
]
