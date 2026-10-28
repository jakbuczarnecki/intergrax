# © Artur Czarnecki. All rights reserved.

"""ALLOWED_FIXED_SUBSTRATE_DECISIONS SSOT (EBH5-G06-02)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification


@dataclass(frozen=True, slots=True)
class FixedSubstrateDecisionRecord:
    substrate_id: str
    technology: str
    scope_paths: str
    contract_if_any: str
    ratification_authority: str
    adr_ref: str
    multi_instance_constraints: str
    review_date: str
    inventory_concern_ids: tuple[str, ...]


_FIXED_SUBSTRATE_DECISIONS: Final[tuple[FixedSubstrateDecisionRecord, ...]] = (
    FixedSubstrateDecisionRecord(
        substrate_id="runtime-sqlite-persistence-bundle",
        technology="SQLite file-backed runtime stores",
        scope_paths="intergrax/runtime/**/sqlite*.py; intergrax/agents/persistence/*sqlite*",
        contract_if_any="Per-store port contracts (checkpoint, human, idempotency, trace, agent, tool-life)",
        ratification_authority="Platform maintainer EBH-5 dossier + OD-EBH5-04 default FIXED disposition",
        adr_ref="STATE-X fixed-substrate inventory SX-F* (historical evidence); EBH-5 SQLite annex",
        multi_instance_constraints="Single-node / profile-scoped; not horizontally scalable substrate",
        review_date="2026-10-28 — EBH-5 certification review cycle (operator re-review required)",
        inventory_concern_ids=(
            "runtime.checkpoint.task",
            "runtime.human.decision",
            "runtime.execution.idempotency",
            "runtime.execution.decision_finalization",
            "runtime.events.runtime_store",
            "runtime.trace.run",
            "runtime.agent.checkpoint",
            "runtime.agent.compensation_queue",
            "runtime.tool_life.durable",
            "runtime.enterprise_reliability.provider_invocation",
        ),
    ),
    FixedSubstrateDecisionRecord(
        substrate_id="single-node-profile-v1-sqlite-triad",
        technology="SQLite (GCA, PIS, agent distribution)",
        scope_paths=(
            "intergrax/applications/_shared/single_node_production_profile_v1.py; "
            "intergrax/agent_distribution/sqlite_stores.py"
        ),
        contract_if_any="SingleNodeProductionProfileV1Config / agent distribution registry",
        ratification_authority="PROD-Q single-node profile qualification chain (operator scoped)",
        adr_ref="PROD-Q ADR chain @ scoped acceptance; DIST-SCALE-X scoped topology evidence",
        multi_instance_constraints="Strict single-node topology; advisory cross-process lock only",
        review_date="2026-10-28 — EBH-5 certification review cycle",
        inventory_concern_ids=("platform.single_node_profile_v1.sqlite_triad",),
    ),
    FixedSubstrateDecisionRecord(
        substrate_id="lab-adaptive-sqlite",
        technology="SQLite adaptive signal/proposal stores",
        scope_paths="applications/lab_application/host/factory.py; intergrax/runtime/adaptive/*_store.py",
        contract_if_any="Lab factory adaptive store paths (GAP-01 remediated)",
        ratification_authority="Lab harness composition owner — H_REFERENCE scope",
        adr_ref="EBH-5 GAP-01 remediation record; OD-EBH5-04 lab-only",
        multi_instance_constraints="Lab harness only; not production extensibility claim",
        review_date="2026-10-28 — EBH-5 certification review cycle",
        inventory_concern_ids=(),
    ),
)


def fixed_substrate_decisions() -> tuple[FixedSubstrateDecisionRecord, ...]:
    return _FIXED_SUBSTRATE_DECISIONS


def fixed_substrate_validation_errors() -> tuple[str, ...]:
    errors: list[str] = []
    inventory_fixed = {
        row.concern_id
        for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY
        if row.extensibility_classification == ExtensibilityClassification.FIXED_SUBSTRATE
    }
    registered: set[str] = set()
    for row in _FIXED_SUBSTRATE_DECISIONS:
        if not row.substrate_id.strip():
            errors.append("fixed substrate: empty substrate_id")
        for req in (
            row.technology,
            row.scope_paths,
            row.ratification_authority,
            row.adr_ref,
            row.multi_instance_constraints,
            row.review_date,
        ):
            if not req.strip():
                errors.append(f"{row.substrate_id}: missing mandatory fixed substrate field")
        registered.update(row.inventory_concern_ids)

    missing = inventory_fixed - registered
    extra = registered - inventory_fixed
    if missing:
        errors.append(f"inventory FIXED_SUBSTRATE concerns without substrate decision: {sorted(missing)}")
    if extra:
        errors.append(f"substrate registry references unknown concern ids: {sorted(extra)}")

    return tuple(errors)


def assert_fixed_substrate_closed_world() -> None:
    errors = fixed_substrate_validation_errors()
    assert not errors, errors
