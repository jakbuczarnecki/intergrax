# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P5 — enterprise freeze-oriented qualification selection manifest (Q12, FRZ-REG-10)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

QUAL_X_P5_START_HEAD: str = "8910f580d8c53a7eb6aac0c61029f9a752c26f3a"

QUAL_X_P5_MANIFEST_OWNER_MODULE: str = (
    "tests/qualification/qual_x/_p5_freeze_selection_catalog.py"
)
QUAL_X_P5_MANIFEST_SEMANTIC_OWNER: str = "QUAL-X (freeze selection manifest)"

_P5_TEST = (
    "tests/qualification/qual_x/test_qual_x_p5_freeze_qualification_suite_assembly.py"
)
_P2_BATCH = "tests/qualification/qual_x/test_qual_x_p2_negative_ingress_matrix.py"
_P3_BATCH = (
    "tests/qualification/qual_x/test_qual_x_p3_inventory_allowlist_recovery_integrity.py"
)
_P4_BATCH = (
    "tests/qualification/qual_x/test_qual_x_p4_determinism_environment_classification_reproducibility.py"
)

_EBH_3 = "tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py"
_EBH_2A = "tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py"
_EBH_2I = "tests/unit/architecture/test_ebh_2i_final_rescan_gate.py"
_EBH_4 = "tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py"
_HARDENING_3 = "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
_HARNESS_W5 = (
    "tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py"
)
_UE_9BR1 = (
    "tests/unit/runtime/architecture/test_ue_9br1_runtime_event_legacy_retirement_gate.py"
)
_EVIDENCE_PERSISTENCE_BOUNDARY = (
    "tests/unit/runtime/events/test_evidence_persistence_boundary.py"
)
_EBH_2F = (
    "tests/unit/runtime/architecture/test_ebh_2f_r2_integration_plugin_contract_typing_gate.py"
)
_HARNESS_GATES = "tests/qualification/harness_01/test_harness_01_gates.py"
_PROD_Q_LIVE_DIM_B = (
    "tests/qualification/prod_q/test_prod_q_r1_r7_q_c_dim8_cross_tenant_integrated_gate.py"
)


def _nid(module: str, test_name: str) -> str:
    return f"{module}::{test_name}"


class QualXFreezeQualificationDimension(StrEnum):
    ARCHITECTURE_DIMENSION_A = "ARCHITECTURE_DIMENSION_A"
    LIVE_PROVIDER_DIMENSION_B = "LIVE_PROVIDER_DIMENSION_B"


class QualXFreezeMandatoryStatus(StrEnum):
    MANDATORY_COLLECTABLE_AT_HEAD = "MANDATORY_COLLECTABLE_AT_HEAD"
    MANDATORY_EXECUTE_AT_QUAL_X_FINAL = "MANDATORY_EXECUTE_AT_QUAL_X_FINAL"
    EXCLUDED_FROM_DIMENSION_A = "EXCLUDED_FROM_DIMENSION_A"


class QualXFreezeEvidenceClass(StrEnum):
    CURRENT_HEAD_EXECUTABLE = "CURRENT_HEAD_EXECUTABLE"
    HISTORICAL_STAGE_LINKAGE = "HISTORICAL_STAGE_LINKAGE"
    MANDATORY_FUTURE_STAGE = "MANDATORY_FUTURE_STAGE"
    UNAVAILABLE_ENVIRONMENTAL = "UNAVAILABLE_ENVIRONMENTAL"
    DIMENSION_B_LIVE_EXCLUDED = "DIMENSION_B_LIVE_EXCLUDED"
    COORDINATION_INTEGRITY_ONLY = "COORDINATION_INTEGRITY_ONLY"


class QualXFreezeEvidenceStatus(StrEnum):
    COLLECTABLE_LINKAGE = "COLLECTABLE_LINKAGE"
    HISTORICAL_ONLY = "HISTORICAL_ONLY"
    PARTIAL_PLATFORM_GAP = "PARTIAL_PLATFORM_GAP"
    DIMENSION_B_OPEN = "DIMENSION_B_OPEN"
    P5_MANIFEST_DEFINED = "P5_MANIFEST_DEFINED"


class QualXFreezeNodeRegistry(StrEnum):
    P2_INGRESS_PROOFS = "P2_INGRESS_PROOFS"
    P3_Q06_RECOVERY_PROOFS = "P3_Q06_RECOVERY_PROOFS"
    P4_DETERMINISM_ENVIRONMENT_GATES = "P4_DETERMINISM_ENVIRONMENT_GATES"


@dataclass(frozen=True, slots=True)
class QualXFreezeSelectionEntry:
    selection_id: str
    freeze_family: str
    qual_x_responsibilities: tuple[str, ...]
    owning_stage: str
    invariant_summary: str
    frz_criteria: tuple[str, ...]
    qualification_dimension: QualXFreezeQualificationDimension
    mandatory_status: QualXFreezeMandatoryStatus
    environment_prerequisites: str
    expected_evidence_class: QualXFreezeEvidenceClass
    evidence_status: QualXFreezeEvidenceStatus
    acceptance_boundary: str
    depends_on: tuple[str, ...]
    pytest_modules: tuple[str, ...] = ()
    delegated_node_registry: QualXFreezeNodeRegistry | None = None
    coordination_batch_module: str | None = None


@dataclass(frozen=True, slots=True)
class QualXFrzRegEvidenceLedgerRow:
    criterion_id: str
    disposition_at_p5: str
    p5_visibility: str
    owning_selection_ids: tuple[str, ...]


QUAL_X_P5_FREEZE_SELECTION_MANIFEST: tuple[QualXFreezeSelectionEntry, ...] = (
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F01-ARCH-BOUNDARIES",
        freeze_family="F01",
        qual_x_responsibilities=("Q01",),
        owning_stage="QUAL-X-P1 / EBH-3/4",
        invariant_summary=(
            "Layer boundaries, dependency ownership, public contract seams; "
            "Execution composition uses EvidencePersistencePort (FRZ-BND-03); "
            "typed observability transport (HARNESS-W5) does not own runtime semantics"
        ),
        frz_criteria=(
            "FRZ-BND-01",
            "FRZ-BND-03",
            "FRZ-BND-06",
            "FRZ-REG-02",
            "FRZ-REG-09",
        ),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
        environment_prerequisites="Standard uv/pytest; no live providers",
        expected_evidence_class=QualXFreezeEvidenceClass.CURRENT_HEAD_EXECUTABLE,
        evidence_status=QualXFreezeEvidenceStatus.COLLECTABLE_LINKAGE,
        acceptance_boundary="AST/closed-world gates @ exact SHA; historical EBH waves not auto-PASS",
        depends_on=(),
        pytest_modules=(
            _EBH_3,
            _EBH_2A,
            _EBH_2I,
            _EBH_4,
            _HARDENING_3,
            _HARNESS_W5,
            _UE_9BR1,
            _EVIDENCE_PERSISTENCE_BOUNDARY,
        ),
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F02-OWNERSHIP-DUPLICATION",
        freeze_family="F02",
        qual_x_responsibilities=("Q02",),
        owning_stage="QUAL-X-P1 / EBH-3",
        invariant_summary="Dependency ownership; duplication prevention until DUP-X",
        frz_criteria=("FRZ-OWN-01", "FRZ-REG-02", "FRZ-REG-05"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.CURRENT_HEAD_EXECUTABLE,
        evidence_status=QualXFreezeEvidenceStatus.PARTIAL_PLATFORM_GAP,
        acceptance_boundary="EBH-3 gate green; platform DUP-X certification not claimed",
        depends_on=("QUAL-X-P5-F01-ARCH-BOUNDARIES",),
        pytest_modules=(_EBH_3,),
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F03-CONTRACTS-TYPING",
        freeze_family="F03",
        qual_x_responsibilities=("Q03",),
        owning_stage="QUAL-X-P1 / EBH-2F",
        invariant_summary="Integration plugin typing and replaceability",
        frz_criteria=("FRZ-TYP-01", "FRZ-CTR-01", "FRZ-PLG-01", "FRZ-REG-02"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.CURRENT_HEAD_EXECUTABLE,
        evidence_status=QualXFreezeEvidenceStatus.COLLECTABLE_LINKAGE,
        acceptance_boundary="EBH-2F/R2 + EBH-2I rescan @ exact SHA",
        depends_on=("QUAL-X-P5-F01-ARCH-BOUNDARIES",),
        pytest_modules=(_EBH_2F, _EBH_2I),
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F04-GOV-SEC-TENANT",
        freeze_family="F04",
        qual_x_responsibilities=("Q04",),
        owning_stage="QUAL-X-P2",
        invariant_summary="Governance/security/tenant negative ingress linkage (G1–G8)",
        frz_criteria=("FRZ-SEC-01", "FRZ-TEN-01", "FRZ-GOV-01", "FRZ-REG-03"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.HISTORICAL_STAGE_LINKAGE,
        evidence_status=QualXFreezeEvidenceStatus.HISTORICAL_ONLY,
        acceptance_boundary=(
            "P2 proof registry collectable; TENANT-X-FINAL production delta = historical corrective range "
            "@ 5ccb72fc…; current-HEAD tenant replay = FINAL fast + test_tenant_x_final_current_head"
        ),
        depends_on=("QUAL-X-P5-F03-CONTRACTS-TYPING",),
        delegated_node_registry=QualXFreezeNodeRegistry.P2_INGRESS_PROOFS,
        coordination_batch_module=_P2_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F05-RUNTIME-LIFECYCLE",
        freeze_family="F05",
        qual_x_responsibilities=("Q05",),
        owning_stage="HARNESS-01 / QUAL-X-P2",
        invariant_summary="Harness tool invocation regression; governance deny → zero intake",
        frz_criteria=("FRZ-EXE-01", "FRZ-REL-01", "FRZ-REG-03"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.CURRENT_HEAD_EXECUTABLE,
        evidence_status=QualXFreezeEvidenceStatus.COLLECTABLE_LINKAGE,
        acceptance_boundary="Harness gates + P2 Q05 linkage; TOOL-LIFE-X not substituted",
        depends_on=("QUAL-X-P5-F04-GOV-SEC-TENANT",),
        pytest_modules=(_HARNESS_GATES,),
        coordination_batch_module=_P2_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F06-PERSISTENCE-RECOVERY",
        freeze_family="F06",
        qual_x_responsibilities=("Q06",),
        owning_stage="QUAL-X-P3 / STATE-X / TRACE-X",
        invariant_summary="Recovery/evidence proof linkage matrix",
        frz_criteria=("FRZ-REC-01", "FRZ-STA-01", "FRZ-TRC-02", "FRZ-REG-05"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.HISTORICAL_STAGE_LINKAGE,
        evidence_status=QualXFreezeEvidenceStatus.HISTORICAL_ONLY,
        acceptance_boundary="Q06 nodes collectable; behavioral proofs HISTORICAL until QUAL-X-FINAL",
        depends_on=("QUAL-X-P5-F05-RUNTIME-LIFECYCLE",),
        delegated_node_registry=QualXFreezeNodeRegistry.P3_Q06_RECOVERY_PROOFS,
        coordination_batch_module=_P3_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F07-GATE-CREDIBILITY",
        freeze_family="F07",
        qual_x_responsibilities=("Q07",),
        owning_stage="QUAL-X-P1",
        invariant_summary="AST/reflection adversarial gate credibility (no proxy PASS)",
        frz_criteria=("FRZ-REG-01", "FRZ-REG-09"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_EXECUTE_AT_QUAL_X_FINAL,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.CURRENT_HEAD_EXECUTABLE,
        evidence_status=QualXFreezeEvidenceStatus.COLLECTABLE_LINKAGE,
        acceptance_boundary="P1 strengthened gates @ exact SHA",
        depends_on=("QUAL-X-P5-F01-ARCH-BOUNDARIES",),
        pytest_modules=(_EBH_3, _EBH_2F, _HARDENING_3),
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F08-ADVERSARIAL-NEGATIVES",
        freeze_family="F08",
        qual_x_responsibilities=("Q08",),
        owning_stage="QUAL-X-P2",
        invariant_summary="Cross-ingress adversarial negative catalog integrity",
        frz_criteria=("FRZ-REG-03",),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.HISTORICAL_STAGE_LINKAGE,
        evidence_status=QualXFreezeEvidenceStatus.HISTORICAL_ONLY,
        acceptance_boundary="P2 catalog + batch; global adversarial rollup still OPEN",
        depends_on=("QUAL-X-P5-F04-GOV-SEC-TENANT",),
        delegated_node_registry=QualXFreezeNodeRegistry.P2_INGRESS_PROOFS,
        coordination_batch_module=_P2_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F09-DETERMINISM",
        freeze_family="F09",
        qual_x_responsibilities=("Q09",),
        owning_stage="QUAL-X-P4",
        invariant_summary="Determinism controls, W2-C debt visibility, mandatory outcome vocabulary",
        frz_criteria=("FRZ-REG-06", "FRZ-REG-07"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites=(
            "Standard uv/pytest; W2-C storm module excluded from Dimension A module list; "
            "hygiene proof @ 75b8f264… (R3) reused"
        ),
        expected_evidence_class=QualXFreezeEvidenceClass.COORDINATION_INTEGRITY_ONLY,
        evidence_status=QualXFreezeEvidenceStatus.PARTIAL_PLATFORM_GAP,
        acceptance_boundary=(
            "P4 batch green; W2-C COLLECTABLE_DETERMINISTIC_STATIC @ R3 — not in 535-node list; "
            "FRZ-REG-06 checklist OPEN"
        ),
        depends_on=("QUAL-X-P5-F06-PERSISTENCE-RECOVERY",),
        delegated_node_registry=QualXFreezeNodeRegistry.P4_DETERMINISM_ENVIRONMENT_GATES,
        coordination_batch_module=_P4_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F10-ENV-CLASSIFICATION",
        freeze_family="F10",
        qual_x_responsibilities=("Q10",),
        owning_stage="QUAL-X-P4 / PROD-Q disposition",
        invariant_summary="R1-SQLITE, PHYSQ preflight, pytest summary fail-closed classification",
        frz_criteria=("FRZ-REG-08", "FRZ-PRD-04"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites="INTERGRAX_RELATIONAL_DB when relational bootstrap required",
        expected_evidence_class=QualXFreezeEvidenceClass.UNAVAILABLE_ENVIRONMENTAL,
        evidence_status=QualXFreezeEvidenceStatus.PARTIAL_PLATFORM_GAP,
        acceptance_boundary="Env failures → non-PASS; R1 HTTP reproduction NOT_EXECUTED until PROD-Q",
        depends_on=("QUAL-X-P5-F09-DETERMINISM",),
        delegated_node_registry=QualXFreezeNodeRegistry.P4_DETERMINISM_ENVIRONMENT_GATES,
        coordination_batch_module=_P4_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F11-INVENTORY-ALLOWLIST",
        freeze_family="F11",
        qual_x_responsibilities=("Q11",),
        owning_stage="QUAL-X-P3",
        invariant_summary="HARNESS-QINF, EBH-2I, STATE-X inventory reconciliation",
        frz_criteria=("FRZ-REG-04", "FRZ-REG-05"),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.CURRENT_HEAD_EXECUTABLE,
        evidence_status=QualXFreezeEvidenceStatus.COLLECTABLE_LINKAGE,
        acceptance_boundary="P3 batch @ exact SHA; FRZ-REG-04 platform scope still OPEN",
        depends_on=("QUAL-X-P5-F06-PERSISTENCE-RECOVERY",),
        coordination_batch_module=_P3_BATCH,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-F12-FREEZE-MANIFEST-INTEGRITY",
        freeze_family="F12",
        qual_x_responsibilities=("Q12",),
        owning_stage="QUAL-X-P5",
        invariant_summary="Manifest completeness, Path B separation, selection truthfulness",
        frz_criteria=("FRZ-REG-10",),
        qualification_dimension=QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A,
        mandatory_status=QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD,
        environment_prerequisites="Standard uv/pytest",
        expected_evidence_class=QualXFreezeEvidenceClass.COORDINATION_INTEGRITY_ONLY,
        evidence_status=QualXFreezeEvidenceStatus.P5_MANIFEST_DEFINED,
        acceptance_boundary="P5 integrity batch; does not close QUAL-X or ARCH-FREEZE",
        depends_on=(
            "QUAL-X-P5-F01-ARCH-BOUNDARIES",
            "QUAL-X-P5-F11-INVENTORY-ALLOWLIST",
        ),
        coordination_batch_module=_P5_TEST,
    ),
    QualXFreezeSelectionEntry(
        selection_id="QUAL-X-P5-DIM-B-PROD-Q-LIVE-EXCLUDED",
        freeze_family="DIM-B",
        qual_x_responsibilities=("Q04", "Q10"),
        owning_stage="PROD-Q-R1-R7-Q",
        invariant_summary="Live integrated provider qualification (explicitly excluded from Dimension A PASS)",
        frz_criteria=("FRZ-PRD-01", "FRZ-PRD-02", "FRZ-REG-08"),
        qualification_dimension=QualXFreezeQualificationDimension.LIVE_PROVIDER_DIMENSION_B,
        mandatory_status=QualXFreezeMandatoryStatus.EXCLUDED_FROM_DIMENSION_A,
        environment_prerequisites="Live credentials, immutable external artifacts, or sanctioned sandbox",
        expected_evidence_class=QualXFreezeEvidenceClass.DIMENSION_B_LIVE_EXCLUDED,
        evidence_status=QualXFreezeEvidenceStatus.DIMENSION_B_OPEN,
        acceptance_boundary="Never counts toward Dimension A freeze acceptance",
        depends_on=(),
        pytest_modules=(_PROD_Q_LIVE_DIM_B,),
    ),
)


QUAL_X_P5_FRZ_REG_EVIDENCE_LEDGER: tuple[QualXFrzRegEvidenceLedgerRow, ...] = (
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-01",
        disposition_at_p5="OPEN — P5 links P1/P7 selections; no global PASS",
        p5_visibility="F07 + F01 gate credibility selections",
        owning_selection_ids=("QUAL-X-P5-F07-GATE-CREDIBILITY",),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-02",
        disposition_at_p5="OPEN — architecture ownership gates linked",
        p5_visibility="F01–F03 selections",
        owning_selection_ids=(
            "QUAL-X-P5-F01-ARCH-BOUNDARIES",
            "QUAL-X-P5-F02-OWNERSHIP-DUPLICATION",
            "QUAL-X-P5-F03-CONTRACTS-TYPING",
        ),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-03",
        disposition_at_p5="OPEN — P2 adversarial catalog linked",
        p5_visibility="F04/F05/F08",
        owning_selection_ids=(
            "QUAL-X-P5-F04-GOV-SEC-TENANT",
            "QUAL-X-P5-F05-RUNTIME-LIFECYCLE",
            "QUAL-X-P5-F08-ADVERSARIAL-NEGATIVES",
        ),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-04",
        disposition_at_p5="OPEN — inventory integrity linked, platform gap explicit",
        p5_visibility="F11",
        owning_selection_ids=("QUAL-X-P5-F11-INVENTORY-ALLOWLIST",),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-05",
        disposition_at_p5="OPEN — recovery + allowlist linkage",
        p5_visibility="F02/F06/F11",
        owning_selection_ids=(
            "QUAL-X-P5-F02-OWNERSHIP-DUPLICATION",
            "QUAL-X-P5-F06-PERSISTENCE-RECOVERY",
            "QUAL-X-P5-F11-INVENTORY-ALLOWLIST",
        ),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-06",
        disposition_at_p5=(
            "PARTIAL — P4 W2-C disposition COLLECTABLE_DETERMINISTIC_STATIC; storm @ 75b8f264…; "
            "checklist OPEN"
        ),
        p5_visibility="F09 via P4 catalog",
        owning_selection_ids=("QUAL-X-P5-F09-DETERMINISM",),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-07",
        disposition_at_p5=(
            "FULL CLEAN-CHECKOUT PROOF CANDIDATE @ 9a175177… (535/535); P5 procedure remains PARTIAL"
        ),
        p5_visibility="F09 + QUAL_X_P5_CLEAN_CHECKOUT_PROCEDURE",
        owning_selection_ids=("QUAL-X-P5-F09-DETERMINISM", "QUAL-X-P5-F12-FREEZE-MANIFEST-INTEGRITY"),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-08",
        disposition_at_p5="PARTIAL — env classification via P4; R1-SQLITE NOT_EXECUTED",
        p5_visibility="F10 + DIM-B exclusion row",
        owning_selection_ids=(
            "QUAL-X-P5-F10-ENV-CLASSIFICATION",
            "QUAL-X-P5-DIM-B-PROD-Q-LIVE-EXCLUDED",
        ),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-09",
        disposition_at_p5="OPEN — strengthened gates linked",
        p5_visibility="F01/F07",
        owning_selection_ids=(
            "QUAL-X-P5-F01-ARCH-BOUNDARIES",
            "QUAL-X-P5-F07-GATE-CREDIBILITY",
        ),
    ),
    QualXFrzRegEvidenceLedgerRow(
        criterion_id="FRZ-REG-10",
        disposition_at_p5="DEFINED @ P5 — manifest SSOT; independent audit pending",
        p5_visibility="F12 + full manifest",
        owning_selection_ids=("QUAL-X-P5-F12-FREEZE-MANIFEST-INTEGRITY",),
    ),
)


def qual_x_p5_delegated_node_ids(
    registry: QualXFreezeNodeRegistry,
) -> tuple[str, ...]:
    if registry is QualXFreezeNodeRegistry.P2_INGRESS_PROOFS:
        from tests.qualification.qual_x._p2_ingress_catalog import (
            qual_x_p2_all_proof_pytest_node_ids,
        )

        return qual_x_p2_all_proof_pytest_node_ids()
    if registry is QualXFreezeNodeRegistry.P3_Q06_RECOVERY_PROOFS:
        from tests.qualification.qual_x._p3_inventory_integrity_catalog import (
            qual_x_p3_q06_all_proof_pytest_node_ids,
        )

        return qual_x_p3_q06_all_proof_pytest_node_ids()
    if registry is QualXFreezeNodeRegistry.P4_DETERMINISM_ENVIRONMENT_GATES:
        from tests.qualification.qual_x._p4_determinism_environment_catalog import (
            qual_x_p4_all_gate_pytest_node_ids,
        )

        return qual_x_p4_all_gate_pytest_node_ids()
    raise ValueError(f"unknown registry: {registry}")


def qual_x_p5_entry_pytest_node_ids(
    entry: QualXFreezeSelectionEntry,
) -> tuple[str, ...]:
    if entry.delegated_node_registry is not None:
        return qual_x_p5_delegated_node_ids(entry.delegated_node_registry)
    return ()


def qual_x_p5_dimension_a_pytest_module_targets() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []

    def _add_module(module: str) -> None:
        if module not in seen:
            seen.add(module)
            ordered.append(module)

    for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
        if (
            entry.qualification_dimension
            is not QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A
        ):
            continue
        for module in entry.pytest_modules:
            _add_module(module)
        if entry.coordination_batch_module:
            _add_module(entry.coordination_batch_module)
        for node_id in qual_x_p5_entry_pytest_node_ids(entry):
            _add_module(node_id.split("::", 1)[0])
    return tuple(ordered)


def qual_x_p5_mandatory_collectable_node_ids() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST:
        if entry.mandatory_status is not QualXFreezeMandatoryStatus.MANDATORY_COLLECTABLE_AT_HEAD:
            continue
        if (
            entry.qualification_dimension
            is not QualXFreezeQualificationDimension.ARCHITECTURE_DIMENSION_A
        ):
            continue
        for node_id in qual_x_p5_entry_pytest_node_ids(entry):
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return tuple(ordered)


def qual_x_p5_all_p5_integrity_pytest_node_ids() -> tuple[str, ...]:
    return (
        _nid(_P5_TEST, "test_qual_x_p5_start_head_documented"),
        _nid(_P5_TEST, "test_qual_x_p5_manifest_selection_ids_unique"),
        _nid(_P5_TEST, "test_qual_x_p5_covers_q01_through_q12"),
        _nid(_P5_TEST, "test_qual_x_p5_frz_reg_ledger_maps_all_reg_criteria"),
        _nid(_P5_TEST, "test_qual_x_p5_manifest_entries_have_frz_criteria"),
        _nid(_P5_TEST, "test_qual_x_p5_known_owning_stages_only"),
        _nid(_P5_TEST, "test_qual_x_p5_dependency_chain_resolvable"),
        _nid(_P5_TEST, "test_qual_x_p5_dimension_b_excluded_from_dimension_a_nodes"),
        _nid(_P5_TEST, "test_qual_x_p5_historical_linkage_cannot_mandatory_pass_alone"),
        _nid(_P5_TEST, "test_qual_x_p5_mandatory_collectable_nodes_exist"),
        _nid(_P5_TEST, "test_qual_x_p5_p5_integrity_nodes_are_collectable"),
        _nid(_P5_TEST, "test_qual_x_p5_synthetic_duplicate_selection_id_would_fail"),
        _nid(_P5_TEST, "test_qual_x_p5_synthetic_missing_frz_mapping_would_fail"),
        _nid(_P5_TEST, "test_qual_x_p5_synthetic_unknown_owner_would_fail"),
        _nid(_P5_TEST, "test_qual_x_p5_synthetic_stale_mandatory_node_would_fail_collectability"),
        _nid(_P5_TEST, "test_qual_x_p5_mandatory_skip_forbidden_by_outcome_contract"),
        _nid(_P5_TEST, "test_qual_x_p5_partial_execution_not_pass"),
        _nid(_P5_TEST, "test_qual_x_p5_mismatched_sha_not_pass"),
        _nid(_P5_TEST, "test_qual_x_p5_clean_checkout_procedure_declares_partial"),
        _nid(_P5_TEST, "test_qual_x_p5_w2_c_storm_module_excluded_from_dimension_a_module_targets"),
        _nid(_P5_TEST, "test_qual_x_p5_w2_c_p4_catalog_disposition_aligned_with_f09_metadata"),
        _nid(_P5_TEST, "test_qual_x_p5_new_architecture_surface_reconciled_by_harness_qinf"),
        _nid(_P5_TEST, "test_qual_x_p5_dimension_a_execution_targets_modules_exist"),
    )


@dataclass(frozen=True, slots=True)
class QualXFreezeCleanCheckoutStep:
    order: int
    instruction: str


@dataclass(frozen=True, slots=True)
class QualXFreezeCleanCheckoutProcedure:
    code_revision_command: str
    collection_command: str
    p5_integrity_execution_command: str
    dimension_a_execution_command: str
    verification_status: str
    mandatory_qual_x_final_note: str
    steps: tuple[QualXFreezeCleanCheckoutStep, ...]


def _qual_x_p5_dimension_a_execution_command() -> str:
    return "uv run pytest -p no:xdist " + " ".join(
        qual_x_p5_dimension_a_pytest_module_targets()
    )


QUAL_X_P5_CLEAN_CHECKOUT_PROCEDURE: QualXFreezeCleanCheckoutProcedure = (
    QualXFreezeCleanCheckoutProcedure(
        code_revision_command="git rev-parse HEAD",
        collection_command=(
            "uv run pytest -p no:xdist --collect-only -q "
            f"{_P5_TEST}"
        ),
        p5_integrity_execution_command=(
            "uv run pytest -p no:xdist "
            f"{_P5_TEST}"
        ),
        dimension_a_execution_command=_qual_x_p5_dimension_a_execution_command(),
        verification_status=(
            "PARTIAL — P5 integrity batch demonstrated on implementation worktree; "
            "full Dimension A replay belongs to QUAL-X-FINAL with explicit env readiness"
        ),
        mandatory_qual_x_final_note=(
            "QUAL-X-FINAL must execute dimension_a_execution_command @ pinned SHA "
            "with INTERGRAX_RELATIONAL_DB set when relational tests are in scope; "
            "no skip/xfail on mandatory nodes; Dimension B rows excluded from A acceptance"
        ),
        steps=(
            QualXFreezeCleanCheckoutStep(1, "git fetch origin development && git rev-parse HEAD"),
            QualXFreezeCleanCheckoutStep(2, "uv sync (existing lockfile; no Docker)"),
            QualXFreezeCleanCheckoutStep(3, "Run collection_command for P5 integrity nodes"),
            QualXFreezeCleanCheckoutStep(4, "Run p5_integrity_execution_command; expect exit 0"),
            QualXFreezeCleanCheckoutStep(
                5,
                "QUAL-X-FINAL: run dimension_a_execution_command @ exact SHA (not claimed by P5)",
            ),
            QualXFreezeCleanCheckoutStep(
                6,
                "Archive logs under .tmp/session/QUAL-X-P5-<HEAD>/run.log",
            ),
        ),
    )
)
