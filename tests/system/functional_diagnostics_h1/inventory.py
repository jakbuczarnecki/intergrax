# © Artur Czarnecki. All rights reserved.

"""Static diagnostic test inventory and invariant ownership matrix for H1."""

from __future__ import annotations

from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    DeterminismClass,
    DiagnosticTestDescriptor,
    DiagnosticTestLayer,
    ExpectedOutcome,
    InvariantOwner,
    QualificationFamily,
    QualificationRunnerDescriptor,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_INVENTORY_ROOTS: tuple[tuple[Path, DiagnosticTestLayer, QualificationFamily, str], ...] = (
    (Path("tests/unit/runtime/diagnostics"), DiagnosticTestLayer.UNIT, QualificationFamily.CORE, "runtime.diagnostics"),
    (Path("tests/unit/runtime/architecture"), DiagnosticTestLayer.STATIC_ARCHITECTURE, QualificationFamily.CORE, "runtime.architecture"),
    (Path("tests/unit/core/qualification"), DiagnosticTestLayer.UNIT, QualificationFamily.CORE, "core.qualification"),
    (Path("tests/integration/runtime"), DiagnosticTestLayer.INTEGRATION, QualificationFamily.CORE, "integration.runtime"),
    (Path("tests/integration/applications"), DiagnosticTestLayer.INTEGRATION, QualificationFamily.CORE, "integration.applications"),
    (Path("tests/system/functional_diagnostics_q1"), DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION, QualificationFamily.Q1, "functional.q1"),
    (Path("tests/system/functional_diagnostics_q2"), DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION, QualificationFamily.Q2, "functional.q2"),
    (Path("tests/system/functional_diagnostics_q3"), DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION, QualificationFamily.Q3, "functional.q3"),
    (Path("tests/system/functional_diagnostics_q4"), DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION, QualificationFamily.Q4, "functional.q4"),
    (Path("tests/system/functional_diagnostics_q5"), DiagnosticTestLayer.SYSTEM, QualificationFamily.Q5, "functional.q5"),
    (Path("tests/system/functional_diagnostics_durability"), DiagnosticTestLayer.RECOVERY, QualificationFamily.D1, "functional.durability"),
    (Path("tests/system/functional_diagnostics_scale"), DiagnosticTestLayer.PERFORMANCE_STRUCTURAL, QualificationFamily.S1, "functional.scale"),
    (Path("tests/system/functional_diagnostics_read_r1"), DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION, QualificationFamily.R1, "functional.read_r1"),
    (Path("tests/system/functional_diagnostics_read_r1_r1"), DiagnosticTestLayer.RECOVERY, QualificationFamily.R1_R1, "functional.read_r1_r1"),
    (Path("tests/system/functional_diagnostics_read_r1_r2"), DiagnosticTestLayer.RECOVERY, QualificationFamily.R1_R2, "functional.read_r1_r2"),
    (Path("tests/system/functional_diagnostics_read_r1_r3"), DiagnosticTestLayer.RECOVERY, QualificationFamily.R1_R3, "functional.read_r1_r3"),
    (Path("tests/system/functional_diagnostics_h1"), DiagnosticTestLayer.SYSTEM, QualificationFamily.H1, "functional.h1"),
    (Path("tests/system/unified_execution"), DiagnosticTestLayer.SYSTEM, QualificationFamily.Q1, "unified_execution"),
)

_EXTERNAL_SERVICES_BY_FAMILY: dict[QualificationFamily, tuple[str, ...]] = {
    QualificationFamily.Q1: ("lkw", "qdrant", "ollama", "mongodb"),
    QualificationFamily.Q2: ("lkw", "mongodb"),
    QualificationFamily.Q3: ("tavily", "web_search"),
    QualificationFamily.Q4: ("model_routing"),
    QualificationFamily.Q5: ("in_process_plugins",),
    QualificationFamily.D1: ("mongodb"),
    QualificationFamily.S1: ("mongodb"),
    QualificationFamily.R1: ("mongodb"),
    QualificationFamily.R1_R1: ("mongodb"),
    QualificationFamily.R1_R2: ("mongodb"),
    QualificationFamily.R1_R3: ("mongodb"),
}

_DIAGNOSTIC_KEYWORDS = (
    "diagnostic",
    "functional_evidence",
    "functional_diagnostic",
    "problem_persistence",
    "problem_lifecycle",
    "execution_reconstruction",
    "diag_",
    "functional_diagnostics",
    "ue_11g",
    "ue_11f",
    "harden_4",
    "terminal_diagnostic",
)

_CONFORMANCE_FILES = frozenset(
    {
        "test_problem_persistence_conformance.py",
        "test_problem_occurrence_persistence_conformance.py",
    }
)


def _is_relevant_test_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if not path.name.startswith("test_"):
        return False
    if path.name in _CONFORMANCE_FILES:
        return True
    text = path.as_posix().lower()
    return any(keyword in text for keyword in _DIAGNOSTIC_KEYWORDS)


def _classify_layer(path: Path, default_layer: DiagnosticTestLayer) -> DiagnosticTestLayer:
    if path.name in _CONFORMANCE_FILES:
        return DiagnosticTestLayer.CONFORMANCE
    if default_layer is DiagnosticTestLayer.STATIC_ARCHITECTURE:
        if "test_diag_" not in path.name:
            return DiagnosticTestLayer.STATIC_ARCHITECTURE
    return default_layer


def _determinism_for_layer(layer: DiagnosticTestLayer) -> DeterminismClass:
    if layer in {
        DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION,
        DiagnosticTestLayer.RECOVERY,
        DiagnosticTestLayer.PERFORMANCE_STRUCTURAL,
    }:
        return DeterminismClass.EXTERNAL_DEPENDENT
    if layer is DiagnosticTestLayer.PERFORMANCE_STRUCTURAL:
        return DeterminismClass.STRUCTURAL_PROBE
    return DeterminismClass.DETERMINISTIC


def _expected_outcome(layer: DiagnosticTestLayer) -> ExpectedOutcome:
    if layer in {DiagnosticTestLayer.REAL_SERVICE_QUALIFICATION, DiagnosticTestLayer.RECOVERY}:
        return ExpectedOutcome.BLOCKED_WHEN_UNAVAILABLE
    return ExpectedOutcome.PASS


def build_diagnostic_test_inventory() -> tuple[DiagnosticTestDescriptor, ...]:
    descriptors: list[DiagnosticTestDescriptor] = []
    seen_paths: set[str] = set()
    counter = 0
    for root_rel, default_layer, family, domain in _INVENTORY_ROOTS:
        root = _REPO_ROOT / root_rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("test_*.py")):
            if not _is_relevant_test_file(path):
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            layer = _classify_layer(path, default_layer)
            services = _EXTERNAL_SERVICES_BY_FAMILY.get(family, ())
            counter += 1
            descriptors.append(
                DiagnosticTestDescriptor(
                    id=f"diag-h1-{counter:04d}",
                    path=rel,
                    layer=layer,
                    domain=domain,
                    requires_external_service=bool(services),
                    required_services=services,
                    qualification_family=family,
                    determinism_class=_determinism_for_layer(layer),
                    expected_outcome=_expected_outcome(layer),
                )
            )
    return tuple(descriptors)


def inventory_counts_by_layer(
    inventory: tuple[DiagnosticTestDescriptor, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in inventory:
        key = item.layer.value
        counts[key] = counts.get(key, 0) + 1
    return counts


REPEATABILITY_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
    "tests/unit/runtime/diagnostics/test_diagnostic_assessment.py",
    "tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
    "tests/unit/runtime/diagnostics/test_diag_functional_read_r1_bounded_reads.py",
    "tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r1_projection_recovery.py",
    "tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r2_append_crash_safety.py",
    "tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r3_active_writer_safety.py",
    "tests/unit/runtime/architecture/test_diag_production_import_hygiene_gate.py",
    "tests/system/functional_diagnostics_q5/test_q5_static_architecture.py",
    "tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py",
    "tests/unit/system/functional_diagnostics_h1",
)

CORE_DETERMINISTIC_PYTEST_TARGETS: tuple[str, ...] = REPEATABILITY_PYTEST_TARGETS + (
    "tests/unit/core/qualification",
    "tests/unit/runtime/architecture/test_diag_enterprise_1_r1_read_gate.py",
    "tests/unit/runtime/architecture/test_diag_foundation_5_destructive_proof.py",
    "tests/system/functional_diagnostics_q5/test_cross_domain_identity_isolation.py",
)

SLOW_ARCHITECTURE_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py",
)

COLLECTION_PYTEST_TARGETS: tuple[str, ...] = CORE_DETERMINISTIC_PYTEST_TARGETS + (
    "tests/unit/runtime/diagnostics",
    "tests/system/functional_diagnostics_q1",
    "tests/system/functional_diagnostics_q2",
    "tests/system/functional_diagnostics_q3",
    "tests/system/functional_diagnostics_q4",
    "tests/system/functional_diagnostics_q5",
    "tests/system/functional_diagnostics_durability",
    "tests/system/functional_diagnostics_scale",
    "tests/system/functional_diagnostics_read_r1",
    "tests/system/functional_diagnostics_read_r1_r1",
    "tests/system/functional_diagnostics_read_r1_r2",
    "tests/system/functional_diagnostics_read_r1_r3",
)

LOCAL_INTEGRATION_TARGETS: tuple[str, ...] = (
    "tests/integration/runtime/test_harden_4b_tenant_diagnostic_isolation_e2e.py",
    "tests/integration/runtime/test_harden_4c_clean_diagnostic_host_e2e.py",
    "tests/integration/runtime/test_harden_4e_diagnostic_read_truth_e2e.py",
    "tests/integration/runtime/test_terminal_diagnostic_production_e2e.py",
)

QUALIFICATION_RUNNERS: tuple[QualificationRunnerDescriptor, ...] = (
    QualificationRunnerDescriptor(
        family=QualificationFamily.Q1,
        runner_path="tests/system/functional_diagnostics_q1/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_Q1_REAL_RAG_QUALIFICATION.md",
        powershell_path="tests/system/functional_diagnostics_q1/run_q1_qualification.ps1",
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.Q2,
        runner_path="tests/system/functional_diagnostics_q2/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_Q2_REAL_TOOL_SELECTION_QUALIFICATION.md",
        powershell_path="tests/system/functional_diagnostics_q2/run_q2_qualification.ps1",
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.Q3,
        runner_path="tests/system/functional_diagnostics_q3/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_Q3_REAL_WEB_SEARCH_QUALIFICATION.md",
        powershell_path="tests/system/functional_diagnostics_q3/run_q3_qualification.ps1",
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.Q4,
        runner_path="tests/system/functional_diagnostics_q4/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_Q4_REAL_MODEL_ROUTING_QUALIFICATION.md",
        powershell_path="tests/system/functional_diagnostics_q4/run_q4_qualification.ps1",
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.Q5,
        runner_path="tests/system/functional_diagnostics_q5/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_Q5_CROSS_DOMAIN_QUALIFICATION.md",
        powershell_path=None,
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.D1,
        runner_path="tests/system/functional_diagnostics_durability/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_DURABILITY_D1_QUALIFICATION.md",
        powershell_path=None,
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.S1,
        runner_path="tests/system/functional_diagnostics_scale/runner.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_SCALE_S1_QUALIFICATION.md",
        powershell_path=None,
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.R1,
        runner_path="tests/system/functional_diagnostics_read_r1/mongo_qualification.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_QUALIFICATION.md",
        powershell_path=None,
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.R1_R1,
        runner_path="tests/system/functional_diagnostics_read_r1_r1/mongo_recovery_qualification.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_R1_QUALIFICATION.md",
        powershell_path=None,
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.R1_R2,
        runner_path="tests/system/functional_diagnostics_read_r1_r2/mongo_recovery_qualification.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_R2_QUALIFICATION.md",
        powershell_path=None,
    ),
    QualificationRunnerDescriptor(
        family=QualificationFamily.R1_R3,
        runner_path="tests/system/functional_diagnostics_read_r1_r3/mongo_active_writer_qualification.py",
        doc_path="docs/project/maintainers/qualification/DIAG_FUNCTIONAL_READ_R1_R3_QUALIFICATION.md",
        powershell_path=None,
    ),
)


def build_invariant_ownership_matrix() -> tuple[InvariantOwner, ...]:
    return (
        InvariantOwner(
            invariant_id="runtime_event_execution_truth_separation",
            description="RuntimeEvent execution truth separated from functional evidence",
            unit_owner="tests/unit/runtime/diagnostics/test_execution_reconstruction.py",
            conformance_owner=None,
            system_real_owner="tests/system/unified_execution/test_ue_11g_c1_r4_functional_diagnosis_gate.py",
            normative_owner="tests/unit/runtime/diagnostics/test_execution_reconstruction.py",
        ),
        InvariantOwner(
            invariant_id="functional_failure_not_execution_failure",
            description="Functional failure distinct from execution failure",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_q1/test_q1_real_rag_qualification.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
        ),
        InvariantOwner(
            invariant_id="canonical_evidence_sole_truth",
            description="Canonical evidence is sole diagnostic truth",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_evidence.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_q1/test_q1_evidence_fidelity.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_evidence.py",
        ),
        InvariantOwner(
            invariant_id="deterministic_analyzer",
            description="FunctionalDiagnosticAnalyzer is deterministic",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_q5/runner.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
        ),
        InvariantOwner(
            invariant_id="proven_pass_fail_semantics",
            description="PROVEN_PASS/FAIL semantics",
            unit_owner="tests/unit/runtime/diagnostics/test_diagnostic_assessment.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_diagnostic_assessment.py",
        ),
        InvariantOwner(
            invariant_id="insufficient_evidence",
            description="INSUFFICIENT_EVIDENCE operator semantics",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_4_operator_projection.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_4_operator_projection.py",
        ),
        InvariantOwner(
            invariant_id="blocked_by_upstream",
            description="BLOCKED_BY_UPSTREAM semantics",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_2_analysis.py",
        ),
        InvariantOwner(
            invariant_id="tenant_isolation",
            description="Tenant isolation for functional evidence",
            unit_owner="tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
            conformance_owner="tests/unit/runtime/diagnostics/test_problem_persistence_conformance.py",
            system_real_owner="tests/integration/runtime/test_harden_4b_tenant_diagnostic_isolation_e2e.py",
            normative_owner="tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
        ),
        InvariantOwner(
            invariant_id="idempotent_evidence_append",
            description="Idempotent functional evidence append",
            unit_owner="tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_durability/runner.py",
            normative_owner="tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
        ),
        InvariantOwner(
            invariant_id="conflict_detection",
            description="Conflicting evidence append fails closed",
            unit_owner="tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_durable_functional_evidence_persistence.py",
        ),
        InvariantOwner(
            invariant_id="durability",
            description="Cross-process durable evidence fidelity (D1)",
            unit_owner="tests/system/functional_diagnostics_durability/test_backend_probe_unit.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_durability/runner.py",
            normative_owner="tests/system/functional_diagnostics_durability/runner.py",
        ),
        InvariantOwner(
            invariant_id="process_restart",
            description="Process boundary restart without in-memory handoff",
            unit_owner="tests/system/functional_diagnostics_durability/durability_orchestrator.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_durability/runner.py",
            normative_owner="tests/system/functional_diagnostics_durability/durability_orchestrator.py",
        ),
        InvariantOwner(
            invariant_id="bounded_pagination",
            description="Bounded functional evidence pagination (R1)",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_bounded_reads.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_read_r1/mongo_qualification.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_bounded_reads.py",
        ),
        InvariantOwner(
            invariant_id="cursor_authentication",
            description="Cursor authentication and tamper resistance",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_1_r1_hardening.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_1_r1_hardening.py",
        ),
        InvariantOwner(
            invariant_id="projection_rebuild_recovery",
            description="Projection rebuild recovery (R1-R1)",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r1_projection_recovery.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_read_r1_r1/mongo_recovery_qualification.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r1_projection_recovery.py",
        ),
        InvariantOwner(
            invariant_id="append_crash_recovery",
            description="Append crash recovery (R1-R2)",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r2_append_crash_safety.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_read_r1_r2/mongo_recovery_qualification.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r2_append_crash_safety.py",
        ),
        InvariantOwner(
            invariant_id="active_intent_fail_closed",
            description="Active pending append intent fail-closed (R1-R3)",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r3_active_writer_safety.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_read_r1_r3/mongo_active_writer_qualification.py",
            normative_owner="tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r3_active_writer_safety.py",
        ),
        InvariantOwner(
            invariant_id="no_silent_canonical_omission",
            description="No silent canonical omission on read path",
            unit_owner="tests/unit/runtime/diagnostics/test_diag_enterprise_1_r1_read_index_races.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_diag_enterprise_1_r1_read_index_races.py",
        ),
        InvariantOwner(
            invariant_id="provider_neutrality",
            description="Provider-neutral persistence conformance",
            unit_owner="tests/unit/runtime/diagnostics/test_problem_persistence_conformance.py",
            conformance_owner="tests/unit/runtime/diagnostics/test_problem_persistence_conformance.py",
            system_real_owner=None,
            normative_owner="tests/unit/runtime/diagnostics/test_problem_persistence_conformance.py",
        ),
        InvariantOwner(
            invariant_id="same_analyzer_q1_q4",
            description="Same FunctionalDiagnosticAnalyzer across Q1-Q4",
            unit_owner="tests/system/functional_diagnostics_q5/test_q5_static_architecture.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_q5/runner.py",
            normative_owner="tests/system/functional_diagnostics_q5/runner.py",
        ),
        InvariantOwner(
            invariant_id="cross_domain_q5_isolation",
            description="Cross-domain Q5 plugin isolation",
            unit_owner="tests/system/functional_diagnostics_q5/test_cross_domain_identity_isolation.py",
            conformance_owner=None,
            system_real_owner="tests/system/functional_diagnostics_q5/runner.py",
            normative_owner="tests/system/functional_diagnostics_q5/test_cross_domain_identity_isolation.py",
        ),
        InvariantOwner(
            invariant_id="telemetry_not_diagnostic_truth",
            description="Telemetry is not diagnostic truth",
            unit_owner="tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/system/functional_diagnostics_h1/test_h1_architecture_gates.py",
        ),
        InvariantOwner(
            invariant_id="diagnostics_execution_boundary",
            description="Diagnostics consumes RuntimeEvent/contracts only",
            unit_owner="tests/unit/runtime/architecture/test_diag_production_import_hygiene_gate.py",
            conformance_owner=None,
            system_real_owner=None,
            normative_owner="tests/unit/runtime/architecture/test_diag_production_import_hygiene_gate.py",
        ),
    )


def verify_invariant_owners(repo_root: Path | None = None) -> tuple[str, ...]:
    root = repo_root or _REPO_ROOT
    missing: list[str] = []
    for owner in build_invariant_ownership_matrix():
        owner_path = root / owner.normative_owner
        if not owner_path.exists():
            missing.append(f"{owner.invariant_id}: missing owner {owner.normative_owner}")
    return tuple(missing)
