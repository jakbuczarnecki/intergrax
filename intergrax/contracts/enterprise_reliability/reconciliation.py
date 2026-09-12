# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconciliation orchestration contracts (ERL Phase 3 — foundation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    UnknownUncertaintyPosture,
    contract_declares_reconciliation,
)

SCHEMA_RECONCILIATION_BOUNDS_V1: Final = "reconciliation_bounds.v1"
SCHEMA_RECONCILIATION_PLAN_V1: Final = "reconciliation_plan.v1"

_DEFAULT_MAX_ATTEMPTS: Final = 5
_DEFAULT_MIN_INTERVAL_SECONDS: Final = 30


class ReconciliationDisposition(StrEnum):
    """Platform decision before any provider read — no I/O here."""

    SCHEDULE_PROBE = "schedule_probe"
    ESCALATE_REQUIRED = "escalate_required"
    NO_DECLARED_PROBE = "no_declared_probe"


class ReconciliationPlanningError(ValueError):
    """Reconciliation plan cannot be derived from contract and posture."""


class ReconciliationBounds(BaseModel):
    """Bounded reconcile attempts — orchestration enforces in later phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RECONCILIATION_BOUNDS_V1
    max_attempts: int = Field(default=_DEFAULT_MAX_ATTEMPTS, ge=1, le=256)
    min_interval_seconds: int = Field(
        default=_DEFAULT_MIN_INTERVAL_SECONDS,
        ge=0,
        le=86_400,
    )


class ReconciliationPlan(BaseModel):
    """Non-executing reconcile intent — probe selection and bounds only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RECONCILIATION_PLAN_V1
    disposition: ReconciliationDisposition
    probe_ref: str | None = None
    plugin_id: str | None = None
    bounds: ReconciliationBounds | None = None
    rationale: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _validate_probe_fields(self) -> ReconciliationPlan:
        if self.disposition is ReconciliationDisposition.SCHEDULE_PROBE:
            if not self.probe_ref or not self.probe_ref.strip():
                raise ValueError("probe_ref required when disposition is schedule_probe")
            if self.plugin_id is None or not self.plugin_id.strip():
                raise ValueError("plugin_id required when disposition is schedule_probe")
            if self.bounds is None:
                raise ValueError("bounds required when disposition is schedule_probe")
        elif self.probe_ref is not None or self.plugin_id is not None or self.bounds is not None:
            raise ValueError(
                "probe_ref, plugin_id, and bounds forbidden unless disposition is schedule_probe",
            )
        return self


def default_reconciliation_bounds() -> ReconciliationBounds:
    return ReconciliationBounds()


def assert_reconciliation_probe_ref(
    contract: ExternalEffectContract,
    probe_ref: str,
) -> None:
    """Fail closed when a strategy names a probe not declared on the contract."""
    normalized = probe_ref.strip()
    if not normalized:
        raise ReconciliationPlanningError("probe_ref required")
    if normalized not in contract.reconciliation_probe_refs:
        raise ReconciliationPlanningError(
            f"probe_ref not declared on contract: {normalized!r}",
        )


def default_contract_probe_ref(contract: ExternalEffectContract) -> str:
    """Platform default probe — first declared read-only reconcile target."""
    if not contract.reconciliation_probe_refs:
        raise ReconciliationPlanningError(
            "contract declares no reconciliation_probe_refs",
        )
    return contract.reconciliation_probe_refs[0]


def evaluate_reconciliation_disposition(
    *,
    unknown_posture: UnknownUncertaintyPosture,
    contract: ExternalEffectContract,
) -> ReconciliationDisposition:
    """
    Classify whether reconcile planning may schedule a declared probe.

    Does not invoke plugins or perform reads.
    """
    if unknown_posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED:
        return ReconciliationDisposition.ESCALATE_REQUIRED
    if not contract_declares_reconciliation(contract):
        return ReconciliationDisposition.NO_DECLARED_PROBE
    return ReconciliationDisposition.SCHEDULE_PROBE


def build_reconciliation_plan(
    *,
    unknown_posture: UnknownUncertaintyPosture,
    contract: ExternalEffectContract,
    plugin_id: str,
    probe_ref: str,
    bounds: ReconciliationBounds | None = None,
    rationale: str = "",
) -> ReconciliationPlan:
    """Materialize a probe plan after strategy selection and contract validation."""
    disposition = evaluate_reconciliation_disposition(
        unknown_posture=unknown_posture,
        contract=contract,
    )
    if disposition is not ReconciliationDisposition.SCHEDULE_PROBE:
        return ReconciliationPlan(
            disposition=disposition,
            rationale=rationale or disposition.value,
        )
    assert_reconciliation_probe_ref(contract, probe_ref)
    normalized_plugin = plugin_id.strip()
    if not normalized_plugin:
        raise ReconciliationPlanningError("plugin_id required")
    return ReconciliationPlan(
        disposition=ReconciliationDisposition.SCHEDULE_PROBE,
        probe_ref=probe_ref.strip(),
        plugin_id=normalized_plugin,
        bounds=bounds or default_reconciliation_bounds(),
        rationale=rationale,
    )


__all__ = [
    "ReconciliationBounds",
    "ReconciliationDisposition",
    "ReconciliationPlan",
    "ReconciliationPlanningError",
    "SCHEMA_RECONCILIATION_BOUNDS_V1",
    "SCHEMA_RECONCILIATION_PLAN_V1",
    "assert_reconciliation_probe_ref",
    "build_reconciliation_plan",
    "default_reconciliation_bounds",
    "default_contract_probe_ref",
    "evaluate_reconciliation_disposition",
]
