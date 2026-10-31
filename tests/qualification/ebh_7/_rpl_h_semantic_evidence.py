# © Artur Czarnecki. All rights reserved.

"""EBH-7 RPL-H semantic evidence — verdicts derived from inspected proof bindings, not name heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX

EBH7_RPL_BATCH_C_EVIDENCE_TEST_NODE: Final[str] = "tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py"


class RplSemanticVerdict(StrEnum):
    QUALIFIED = "QUALIFIED"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class RplConcernSemanticEvidence:
    concern_id: str
    verdict: RplSemanticVerdict
    positive_test: str
    negative_test: str
    production_selection_path: str
    alternate_backends: str
    tenant_gov_assertions: str
    lifecycle: str
    notes: str


LEGACY_BEHAVIORAL_QUALIFIED: Final[frozenset[str]] = frozenset(
    {
        "integration.key_value_cache",
        "persistence.relational_sqlite_lab",
        "int_config.configure_existing",
        "observability.otlp_transport",
        "tokenizer.selection",
    }
)

BLOCKED_CONCERN_ROOT_CAUSE: Final[dict[str, str]] = {
    "integration.external_work": "No providers/external_work category — cannot register alternate backend",
    "integration.billing_meter": "Single catalog provider (stripe) — alternate vendor sandbox required",
    "integration.managed_retrieval": "Single catalog provider (openai) — alternate vendor sandbox required",
    "integration.ml_inference_host": "Single catalog provider (replicate) — alternate vendor sandbox required",
    "integration.model_serving_runtime": "Single catalog provider (ollama) — alternate vendor sandbox required",
    "integration.vision_serving": "Single catalog provider (triton) — alternate vendor sandbox required",
}


def _legacy_evidence(concern_id: str) -> RplConcernSemanticEvidence:
    node = QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE[concern_id]
    return RplConcernSemanticEvidence(
        concern_id=concern_id,
        verdict=RplSemanticVerdict.QUALIFIED,
        positive_test=node,
        negative_test=node,
        production_selection_path="Sanctioned composition @ resolve_from_profile / bootstrap",
        alternate_backends="Legacy EBH-5 behavioral alternate exercised",
        tenant_gov_assertions="Inspected in legacy behavioral proof",
        lifecycle="Inspected in legacy behavioral proof",
        notes="Legacy EBH-5 QUALIFIED — preserved",
    )


def _integration_dual_slug_evidence(entry_concern: str) -> RplConcernSemanticEvidence:
    entry = catalog_by_concern_id()[entry_concern]
    positive = f"{EBH7_RPL_BATCH_C_EVIDENCE_TEST_NODE}::test_ebh5_pkgc_v2_dual_slug_integration[{entry.concern_id}]"
    negative = f"{EBH7_RPL_BATCH_C_EVIDENCE_TEST_NODE}::test_ebh5_pkgc_v2_dual_slug_unregistered[{entry.concern_id}]"
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_row

    verdict = normative_rpl_row(entry_concern).semantic_verdict
    lifecycle = "Dual slug resolve_from_profile + unregistered fail-closed @ Package C v2"
    notes = (
        "EBH-7 gap-closure: sanctioned registry selection + alternate slug proven (vendor sandbox out of dev-ci scope)"
    )
    return RplConcernSemanticEvidence(
        concern_id=entry.concern_id,
        verdict=verdict,
        positive_test=positive,
        negative_test=negative,
        production_selection_path="IntegrationProfile + resolve_from_profile",
        alternate_backends=f"{entry.slug_a} | {entry.slug_b}",
        tenant_gov_assertions="N/A unless category contract enforces tenant scope in proof",
        lifecycle=lifecycle,
        notes=notes,
    )


def _package_c_factory_evidence(concern_id: str) -> RplConcernSemanticEvidence:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import normative_rpl_row

    norm = normative_rpl_row(concern_id)
    node = QUALIFIED_CONCERN_BEHAVIORAL_PROOF_NODE.get(concern_id, "")
    if not node:
        return RplConcernSemanticEvidence(
            concern_id=concern_id,
            verdict=norm.semantic_verdict,
            positive_test="tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py",
            negative_test="tests/qualification/ebh_5/test_ebh_5_behavioral_package_c.py",
            production_selection_path="Factory ownership — behavioral node missing",
            alternate_backends="Not demonstrated",
            tenant_gov_assertions="Not inspected",
            lifecycle="Not inspected",
            notes=norm.defect,
        )
    return RplConcernSemanticEvidence(
        concern_id=concern_id,
        verdict=norm.semantic_verdict,
        positive_test=node,
        negative_test=node,
        production_selection_path="Sanctioned factory / composition @ Package C",
        alternate_backends="Distinct alternate or OD fixed-substrate required for QUALIFIED",
        tenant_gov_assertions="Inspected where proof node enforces tenant scope",
        lifecycle="Package C behavioral proof @ HEAD",
        notes=norm.defect,
    )


def _blocked(concern_id: str) -> RplConcernSemanticEvidence:
    return RplConcernSemanticEvidence(
        concern_id=concern_id,
        verdict=RplSemanticVerdict.BLOCKED,
        positive_test="NONE",
        negative_test="NONE",
        production_selection_path="BLOCKED",
        alternate_backends="NONE",
        tenant_gov_assertions="N/A",
        lifecycle="N/A",
        notes=BLOCKED_CONCERN_ROOT_CAUSE[concern_id],
    )


def rpl_concern_semantic_evidence(concern_id: str) -> RplConcernSemanticEvidence:
    if concern_id in LEGACY_BEHAVIORAL_QUALIFIED:
        return _legacy_evidence(concern_id)
    if concern_id in BLOCKED_CONCERN_ROOT_CAUSE:
        return _blocked(concern_id)
    if concern_id in catalog_by_concern_id():
        return _integration_dual_slug_evidence(concern_id)
    return _package_c_factory_evidence(concern_id)


def all_rpl_concern_semantic_evidence() -> tuple[RplConcernSemanticEvidence, ...]:
    return tuple(rpl_concern_semantic_evidence(row.concern_id) for row in CROSS_LAYER_REPLACEABILITY_MATRIX)


def rpl_semantic_verdict_counts() -> dict[RplSemanticVerdict, int]:
    counts = {v: 0 for v in RplSemanticVerdict}
    for ev in all_rpl_concern_semantic_evidence():
        counts[ev.verdict] += 1
    return counts
