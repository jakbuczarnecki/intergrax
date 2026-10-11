# © Artur Czarnecki. All rights reserved.

"""Synthetic persisted-schema classification probes (R2-P0 adversarial gates)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.compat_x._compat_x_persisted_types import (
    PersistedVersionDisposition,
)


@dataclass(frozen=True, slots=True)
class SyntheticPersistedClassification:
    payload_type: str
    orchestrator_type: str
    payload_disposition: PersistedVersionDisposition
    orchestrator_disposition: PersistedVersionDisposition


def synthetic_service_false_positive_classification() -> SyntheticPersistedClassification:
    return SyntheticPersistedClassification(
        payload_type="ModelV1",
        orchestrator_type="PersistOrchestratorService",
        payload_disposition=PersistedVersionDisposition.EXPLICIT_PAYLOAD_VERSION,
        orchestrator_disposition=PersistedVersionDisposition.NOT_A_PERSISTED_SCHEMA,
    )


def synthetic_implicit_fallback_allowed() -> bool:
    return False


def synthetic_envelope_inheritance_proven() -> bool:
    return True
