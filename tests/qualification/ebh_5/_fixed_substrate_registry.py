# © Artur Czarnecki. All rights reserved.

"""ALLOWED_FIXED_SUBSTRATE_DECISIONS SSOT (EBH5-G06-02)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification


class FixedSubstrateDecisionStatus(StrEnum):
    RATIFIED = "RATIFIED — SCOPED QUALIFICATION EVIDENCE"
    PROPOSED_OPERATOR_PENDING = "PROPOSED — OPERATOR PENDING"
    NOT_RATIFIED = "NOT RATIFIED"


@dataclass(frozen=True, slots=True)
class FixedSubstrateDecisionRecord:
    substrate_id: str
    technology: str
    scope_paths: str
    canonical_contract: str
    ratification_authority: str
    adr_ref: str
    multi_instance_constraints: str
    review_date: str
    inventory_concern_ids: tuple[str, ...]
    decision_status: FixedSubstrateDecisionStatus
    decision_evidence_sha: str


_FIXED_SUBSTRATE_DECISIONS: Final[tuple[FixedSubstrateDecisionRecord, ...]] = (
    FixedSubstrateDecisionRecord(
        substrate_id="runtime-sqlite-persistence-bundle",
        technology="SQLite file-backed runtime stores",
        scope_paths="intergrax/runtime/**/sqlite*.py; intergrax/agents/persistence/*sqlite*",
        canonical_contract="Per-store port contracts (checkpoint, human, idempotency, trace, agent, tool-life)",
        ratification_authority=(
            "Operator 2026-10-28 — OD-EBH5-04 RUNTIME SQLITE FIXED SUBSTRATE (SCOPED); "
            "STATE-X SX-F* + EBH_5_SQLITE_ANNEX"
        ),
        adr_ref="STATE-X fixed-substrate inventory SX-F*; EBH_5_SQLITE_ANNEX.md",
        multi_instance_constraints="Single-node / profile-scoped; not horizontally scalable substrate",
        review_date="2026-10-28 — operator ratification (scoped FIXED; not distributed scale)",
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
        decision_status=FixedSubstrateDecisionStatus.RATIFIED,
        decision_evidence_sha=(
            "OD-EBH5-04-A @ 2026-10-28; STATE-X/SQLite annex; wave 83a376b3e1b312e2e4f88da548c99c328f3589f7"
        ),
    ),
    FixedSubstrateDecisionRecord(
        substrate_id="single-node-profile-v1-sqlite-triad",
        technology="SQLite (GCA, PIS, agent distribution)",
        scope_paths=(
            "intergrax/applications/_shared/single_node_production_profile_v1.py; "
            "intergrax/agent_distribution/sqlite_stores.py"
        ),
        canonical_contract="SingleNodeProductionProfileV1Config / agent distribution registry",
        ratification_authority="PROD-Q single-node profile qualification chain (scoped acceptance)",
        adr_ref="PROD-Q ADR chain; DIST-SCALE-X scoped topology evidence",
        multi_instance_constraints="Strict single-node topology; advisory cross-process lock only",
        review_date="2026-10-28 — scoped PROD-Q acceptance (single-node triad)",
        inventory_concern_ids=("platform.single_node_profile_v1.sqlite_triad",),
        decision_status=FixedSubstrateDecisionStatus.RATIFIED,
        decision_evidence_sha="83a376b3e1b312e2e4f88da548c99c328f3589f7 (PROD-Q scoped wave baseline)",
    ),
    FixedSubstrateDecisionRecord(
        substrate_id="lab-adaptive-sqlite",
        technology="SQLite adaptive signal/proposal stores",
        scope_paths="applications/lab_application/host/factory.py; intergrax/runtime/adaptive/*_store.py",
        canonical_contract="Lab factory adaptive store paths (GAP-01 remediated)",
        ratification_authority=("Operator 2026-10-28 — OD-EBH5-04 LAB ADAPTIVE SQLITE FIXED SUBSTRATE (LAB ONLY)"),
        adr_ref="EBH-5 GAP-01 remediation record; OD-EBH5-04-B lab-only",
        multi_instance_constraints="Lab harness only; not production extensibility claim",
        review_date="2026-10-28 — operator ratification (lab-only FIXED)",
        inventory_concern_ids=(),
        decision_status=FixedSubstrateDecisionStatus.RATIFIED,
        decision_evidence_sha="OD-EBH5-04-B @ 2026-10-28; GAP-01 @ ffe0aae8d2b64a558a18226cc1435f0058ff561b",
    ),
    FixedSubstrateDecisionRecord(
        substrate_id="ebh7-conversational-memory-sqlite-lab",
        technology="SQLite conversational memory store (lab scope)",
        scope_paths="intergrax/memory/stores/sqlite_conversational_memory_store.py",
        canonical_contract="Conversational memory store port — sqlite lab substrate only",
        ratification_authority="Operator 2026-10-31 — EBH-7 OD-RPL-G2 scoped conversational SQLite lab",
        adr_ref="EBH_7_FIXED_SUBSTRATE_RATIFICATIONS.md; EBH_5_SQLITE_ANNEX scoped slice",
        multi_instance_constraints="Lab/single-node conversational persistence — not full memory abstraction FIXED",
        review_date="2026-10-31 — EBH-7 terminal closure",
        inventory_concern_ids=("memory.sqlite.conversational",),
        decision_status=FixedSubstrateDecisionStatus.RATIFIED,
        decision_evidence_sha="fd3bc0104c9273f30262946afb16c15c04c8493a operator authorization",
    ),
    FixedSubstrateDecisionRecord(
        substrate_id="ebh7-integration-single-slug-vendor-slice",
        technology="Single catalog slug per integration category (Path B boundary)",
        scope_paths="intergrax/integrations/providers/** — one production slug per EBH-7 blocked category",
        canonical_contract="IntegrationProfile + resolve_from_profile — no in-repo fake second vendor",
        ratification_authority="Operator 2026-10-31 — EBH-7 OD-RPL-G2 FIXED vendor slice",
        adr_ref="EBH_7_FIXED_SUBSTRATE_RATIFICATIONS.md; VBN § Path B",
        multi_instance_constraints="Not vendor-equivalent; live sandbox replay required for Path B evidence",
        review_date="2026-10-31 — EBH-7 terminal closure",
        inventory_concern_ids=(
            "integration.billing_meter",
            "integration.managed_retrieval",
            "integration.ml_inference_host",
            "integration.model_serving_runtime",
            "integration.vision_serving",
        ),
        decision_status=FixedSubstrateDecisionStatus.RATIFIED,
        decision_evidence_sha="fd3bc0104c9273f30262946afb16c15c04c8493a operator authorization",
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
            row.canonical_contract,
            row.decision_evidence_sha,
        ):
            if not req.strip():
                errors.append(f"{row.substrate_id}: missing mandatory fixed substrate field")
        if row.decision_status == FixedSubstrateDecisionStatus.NOT_RATIFIED:
            errors.append(f"{row.substrate_id}: NOT_RATIFIED substrate blocks EBH-5 closure")
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
