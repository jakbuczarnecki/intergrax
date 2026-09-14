# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed observability attributes for ERL reliability diagnostic handoff (ERL-DIAG-001B)."""

from __future__ import annotations

from pydantic import Field

from intergrax.runtime.observability.export_attributes import ApplicationObservabilityAttributes

EXTERNAL_EFFECT_RELIABILITY_OBSERVABILITY_NAMESPACE = "enterprise_reliability.diagnostics"


class ExternalEffectReliabilityObservabilityAttributes(ApplicationObservabilityAttributes):
    """Structured ERL reliability facts carried on PlatformProblemSignal — not an opaque blob."""

    namespace: str = EXTERNAL_EFFECT_RELIABILITY_OBSERVABILITY_NAMESPACE
    observation_id: str = Field(min_length=1)
    signal_kind: str = Field(min_length=1)
    reliability_case_id: str = Field(min_length=1)
    lifecycle_state: str = Field(min_length=1)
    automation_safety_hint: str = Field(min_length=1)
    external_effect_contract_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    execution_id: str | None = None
    attempt_id: str | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None
    source_transition_id: str | None = None
    trace_refs: tuple[str, ...] = ()


__all__ = [
    "EXTERNAL_EFFECT_RELIABILITY_OBSERVABILITY_NAMESPACE",
    "ExternalEffectReliabilityObservabilityAttributes",
]
