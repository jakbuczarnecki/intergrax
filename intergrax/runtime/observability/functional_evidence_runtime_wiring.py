# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Harness wiring for functional evidence persistence (DIAG-DURABILITY-D1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    wire_functional_evidence_persistence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import FunctionalEvidencePersistence
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.functional_evidence_recorder import FunctionalEvidenceRecorder

if TYPE_CHECKING:
    from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext

_FUNCTIONAL_EVIDENCE_WIRING_EXTRA_KEY = "functional_evidence_wiring"


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceRuntimeWiring:
    persistence: FunctionalEvidencePersistence
    recorder: FunctionalEvidenceRecorder


def wire_functional_evidence_runtime(
    *,
    cursor_secret: str | bytes,
    document_store: DocumentStore | None = None,
    producer_component: str = "agents.local_search",
) -> FunctionalEvidenceRuntimeWiring:
    """
    Compose functional evidence recorder over explicit persistence backend.

    ``document_store`` selects durable ConditionalDocumentStore persistence.
    When omitted, in-memory persistence is used (tests/dev only).
    """
    secret_bytes = (
        cursor_secret
        if isinstance(cursor_secret, bytes)
        else cursor_secret.encode("utf-8")
    )
    if document_store is not None:
        persistence = wire_functional_evidence_persistence(
            document_store=document_store,
            cursor_secret=secret_bytes,
        )
    else:
        persistence = InMemoryFunctionalEvidencePersistence(cursor_secret=secret_bytes)
    recorder = FunctionalEvidenceRecorder(
        persistence,
        producer_component=producer_component,
    )
    return FunctionalEvidenceRuntimeWiring(persistence=persistence, recorder=recorder)


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
    return _FUNCTIONAL_EVIDENCE_WIRING_EXTRA_KEY


def attach_functional_evidence_recorder_from_runtime_state(
    exec_ctx: RuntimeExecutionContext,
) -> None:
    from intergrax.runtime.observability.functional_evidence_recorder import (
        attach_functional_evidence_recorder,
    )

    runtime_state = exec_ctx.metadata.get("runtime_state")
    if runtime_state is None:
        return
    context = getattr(runtime_state, "context", None)
    config = getattr(context, "config", None) if context is not None else None
    wiring_ctx = getattr(config, "tool_wiring_context", None) if config is not None else None
    if wiring_ctx is None:
        return
    wiring = wiring_ctx.extras.get(functional_evidence_wiring_extra_key())
    if not isinstance(wiring, FunctionalEvidenceRuntimeWiring):
        return
    attach_functional_evidence_recorder(exec_ctx, wiring.recorder)


__all__ = [
    "FunctionalEvidenceRuntimeWiring",
    "attach_functional_evidence_recorder_from_runtime_state",
    "functional_evidence_wiring_extra_key",
    "wire_functional_evidence_runtime",
    "wire_in_memory_functional_evidence_runtime",
]
