# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project read-only runtime facts into immutable intelligence context (W6-C)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_intelligence.context import (
    RuntimeIntelligenceContext,
    RuntimeIntelligenceContextMetadata,
    RuntimeIntelligenceFactKind,
    RuntimeIntelligenceFactReference,
    validate_runtime_intelligence_context,
)
from intergrax.contracts.runtime_intelligence.errors import InvalidIntelligenceContextError
from intergrax.runtime.runtime_intelligence.runtime_facts import (
    RuntimeIntelligenceFacts,
    RuntimeIntelligenceObservedSignal,
    RuntimeIntelligenceSignalKind,
)

_INTELLIGENCE_SIGNAL_REF_PREFIX = "intelligence_signal:"


def _signal_to_fact_reference(signal: RuntimeIntelligenceObservedSignal) -> RuntimeIntelligenceFactReference:
    intensity_code = f"{int(round(signal.intensity * 100)):03d}"
    ref = f"{_INTELLIGENCE_SIGNAL_REF_PREFIX}{signal.kind.value}:{intensity_code}"
    return RuntimeIntelligenceFactReference(
        fact_kind=RuntimeIntelligenceFactKind.RUNTIME_EVENT,
        fact_ref=ref,
    )


def _merged_fact_references(
    facts: RuntimeIntelligenceFacts,
) -> tuple[RuntimeIntelligenceFactReference, ...]:
    if not facts.observed_signals:
        return facts.fact_references
    projected = tuple(_signal_to_fact_reference(signal) for signal in facts.observed_signals)
    return facts.fact_references + projected


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceContextBuilder:
    """Single responsibility: validate facts and materialize immutable context."""

    def build(self, facts: RuntimeIntelligenceFacts) -> RuntimeIntelligenceContext:
        fact_references = _merged_fact_references(facts)
        if not fact_references:
            raise InvalidIntelligenceContextError(
                "fact_references must be non-empty for intelligence context projection"
            )
        context = RuntimeIntelligenceContext(
            tenant_id=facts.tenant_id,
            task_id=facts.task_id,
            run_id=facts.run_id,
            attempt_id=facts.attempt_id,
            execution_id=facts.execution_id,
            fact_references=fact_references,
            metadata=RuntimeIntelligenceContextMetadata(
                collected_at=facts.collected_at,
                correlation_id=facts.correlation_id,
                context_label=facts.context_label,
            ),
        )
        validate_runtime_intelligence_context(context)
        return context


def intelligence_signal_from_fact_ref(
    fact_ref: str,
) -> tuple[RuntimeIntelligenceSignalKind, float] | None:
    """Parse projected signal pointers — shared with deterministic analyzer."""
    if not fact_ref.startswith(_INTELLIGENCE_SIGNAL_REF_PREFIX):
        return None
    payload = fact_ref.removeprefix(_INTELLIGENCE_SIGNAL_REF_PREFIX)
    kind_str, intensity_code = payload.rsplit(":", maxsplit=1)
    try:
        kind = RuntimeIntelligenceSignalKind(kind_str)
    except ValueError:
        return None
    if len(intensity_code) != 3 or not intensity_code.isdigit():
        return None
    return kind, int(intensity_code) / 100.0


__all__ = [
    "RuntimeIntelligenceContextBuilder",
    "intelligence_signal_from_fact_ref",
]
