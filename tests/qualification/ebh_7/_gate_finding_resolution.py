# © Artur Czarnecki. All rights reserved.

"""G09 — verifiable corrective proof / supersession per gate finding (no silent allowlist closure)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from tests.qualification.ebh_5._proof_validation import pytest_test_function_exists
from tests.qualification.ebh_7._gate_findings_register import EBH7_CORRECTIVE_GATE_FINDINGS

_REPO = Path(__file__).resolve().parents[3]


class FindingResolutionKind(StrEnum):
    CORRECTIVE_PYTEST = "CORRECTIVE_PYTEST"
    CORRECTIVE_ASSERT = "CORRECTIVE_ASSERT"
    HISTORICAL_SUPERSEDED = "HISTORICAL_SUPERSEDED"


@dataclass(frozen=True, slots=True)
class FindingResolution:
    gate_id: str
    kind: FindingResolutionKind
    evidence_ref: str
    notes: str = ""


_FINDING_RESOLUTIONS: Final[dict[str, FindingResolution]] = {
    "W6-ID-REGISTRY-ONLY": FindingResolution(
        "W6-ID-REGISTRY-ONLY",
        FindingResolutionKind.CORRECTIVE_PYTEST,
        "tests/qualification/ebh_7/test_ebh_7_semantic_proof_binding_gates.py::test_ebh7_semantic_exact_blocker_counts_registered",
        "TEN/XDM EXACT counts + disposition registry — not scenario_id tuple alone",
    ),
    "W6-TENANT-X-FINAL-READY": FindingResolution(
        "W6-TENANT-X-FINAL-READY",
        FindingResolutionKind.HISTORICAL_SUPERSEDED,
        "docs/project/maintainers/qualification/EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md@TENANT-X",
        "Removed as platform E2E substitute; mapped proofs only where semantically aligned",
    ),
    "W7-HASATTR-DUP-X": FindingResolution(
        "W7-HASATTR-DUP-X",
        FindingResolutionKind.CORRECTIVE_PYTEST,
        "tests/qualification/ebh_7/test_ebh_7_semantic_proof_binding_gates.py::test_ebh7_b6_readiness_mandatory_exact_coverage",
        "Executable proof bindings — not hasattr(dup_exit)",
    ),
    "W5-RPL-ROW-COUNT": FindingResolution(
        "W5-RPL-ROW-COUNT",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._rpl_h_semantic_evidence:rpl_semantic_verdict_counts",
        "Agent vs normative semantic counts separated from row inventory",
    ),
    "W5-VBN-DOC-PRESENCE": FindingResolution(
        "W5-VBN-DOC-PRESENCE",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._traceability_matrix:assert_ebh7_traceability_integrity",
        "Traceability matrix — materialization vs qualification columns",
    ),
    "TRACE-466-COUNT-ONLY": FindingResolution(
        "TRACE-466-COUNT-ONLY",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._traceability_matrix:assert_ebh7_traceability_behavioral_integrity",
        "Extended rows with owner, evidence, acceptance fields",
    ),
    "READINESS-TABLE-PASS": FindingResolution(
        "READINESS-TABLE-PASS",
        FindingResolutionKind.HISTORICAL_SUPERSEDED,
        "docs/project/maintainers/qualification/EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md@readiness-reconcile",
        "Readiness reconciled to materialization vs qualification columns",
    ),
    "GAP-TEN-UNK-05-CLOSED": FindingResolution(
        "GAP-TEN-UNK-05-CLOSED",
        FindingResolutionKind.CORRECTIVE_PYTEST,
        "tests/qualification/ebh_7/test_ebh_7_semantic_proof_binding_gates.py::test_ebh7_semantic_exact_blocker_counts_registered",
        "Structural N/A evidence only — not platform E2E proof",
    ),
    "AUD-01-RPL-AUTO-QUALIFIED": FindingResolution(
        "AUD-01-RPL-AUTO-QUALIFIED",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._rpl_h_normative_evidence:assert_ebh7_rpl_h_normative_evidence_integrity",
        "Normative RPL layer + appendix — not registry membership alone",
    ),
    "B6-NORMATIVE-OPEN-ROWS": FindingResolution(
        "B6-NORMATIVE-OPEN-ROWS",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._rpl_h_normative_evidence:assert_ebh7_rpl_h_normative_for_b6_readiness",
        "Normative RPL readiness gate — L3 green ≠ B6 PASS",
    ),
    "B6-F01-EXTERNAL-WORK-DUAL-INSTANCE": FindingResolution(
        "B6-F01-EXTERNAL-WORK-DUAL-INSTANCE",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_5._external_work_dual_provider_proof:prove_external_work_dual_profile_binding",
        "Qual alpha/beta + IntegrationProfile selection + contract ops",
    ),
    "B6-F02-PRODQ-TEXT-PREFIX-NEGATIVE": FindingResolution(
        "B6-F02-PRODQ-TEXT-PREFIX-NEGATIVE",
        FindingResolutionKind.CORRECTIVE_PYTEST,
        "tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py::test_ebh7_internal_rpl_negative",
        "prove_prodq_colocated_adapter_internal_negative @ terminal INTERNAL gate",
    ),
    "B6-F03-TEN-XDM-DOC-CONFLICT": FindingResolution(
        "B6-F03-TEN-XDM-DOC-CONFLICT",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._ebh7_certification_ssot:assert_ebh7_ten_xdm_evidence_ssot",
        "SSOT 26/0 TEN and 34/0 XDM @ registry + consistency matrix",
    ),
    "B6-F04-P03-TEST-CLOSURE-SHA": FindingResolution(
        "B6-F04-P03-TEST-CLOSURE-SHA",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._ebh7_p03_evidence_integrity:assert_ebh7_p03_manifest_role_integrity",
        "TEST_EXECUTION_SHA descends from B6 corrective — not closure-only alias",
    ),
    "B6-F07-RPL-EF-PROOF-STRENGTH": FindingResolution(
        "B6-F07-RPL-EF-PROOF-STRENGTH",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_5._dual_slug_rpl_ef_contract_evidence:dual_slug_rpl_ef_audit_errors",
        "28 dual-slug RPL-E contract obligations + Package C v2 CDE/unreg pytest",
    ),
    "B6-F08-ECAP-P03-DERIVED-GATES": FindingResolution(
        "B6-F08-ECAP-P03-DERIVED-GATES",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._ebh7_closure_evidence_audit:assert_ebh7_ecap_p03_derived_closure_gates",
        "G07/G08 derive from certification + P03 integrity — not static PASS",
    ),
    "B6-F09-FINDING-RESOLUTION-EVIDENCE": FindingResolution(
        "B6-F09-FINDING-RESOLUTION-EVIDENCE",
        FindingResolutionKind.CORRECTIVE_ASSERT,
        "tests.qualification.ebh_7._gate_finding_resolution:assert_ebh7_gate_findings_resolution_complete",
        "Every finding bound to pytest, assert hook, or documented supersession",
    ),
}


def finding_resolution(gate_id: str) -> FindingResolution | None:
    return _FINDING_RESOLUTIONS.get(gate_id)


def _pytest_node_resolvable(node_id: str) -> bool:
    if pytest_test_function_exists(node_id):
        return True
    if "[" in node_id:
        base = node_id.split("[", 1)[0]
        if pytest_test_function_exists(base):
            return True
    if "::" in node_id and node_id.startswith("tests."):
        mod_part, func = node_id.rsplit("::", 1)
        file_rel = mod_part.replace(".", "/") + ".py"
        return pytest_test_function_exists(f"{file_rel}::{func}")
    return False


def _run_corrective_assert(evidence_ref: str) -> None:
    module_name, func_name = evidence_ref.split(":", 1)
    mod = importlib.import_module(module_name)
    target = getattr(mod, func_name)
    if func_name == "dual_slug_rpl_ef_audit_errors":
        from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import INTEGRATION_DUAL_SLUG_CATALOG

        errors: list[str] = []
        for entry in INTEGRATION_DUAL_SLUG_CATALOG:
            errors.extend(target(entry))
        assert not errors, "; ".join(errors[:6])
        return
    result = target()
    if isinstance(result, tuple) and result:
        raise AssertionError("; ".join(str(x) for x in result[:6]))


def validate_finding_resolution(res: FindingResolution) -> str | None:
    if res.kind is FindingResolutionKind.CORRECTIVE_PYTEST:
        if not _pytest_node_resolvable(res.evidence_ref):
            return f"{res.gate_id}: corrective pytest missing {res.evidence_ref}"
        return None
    if res.kind is FindingResolutionKind.CORRECTIVE_ASSERT:
        try:
            _run_corrective_assert(res.evidence_ref)
        except AssertionError as exc:
            return f"{res.gate_id}: assert hook failed — {exc}"
        except Exception as exc:  # noqa: BLE001 — audit must surface import/runtime faults
            return f"{res.gate_id}: assert hook error — {exc}"
        return None
    if res.kind is FindingResolutionKind.HISTORICAL_SUPERSEDED:
        doc_part = res.evidence_ref.split("@", 1)[0]
        path = _REPO / doc_part
        if not path.is_file():
            return f"{res.gate_id}: supersession doc missing {doc_part}"
        if not res.notes.strip():
            return f"{res.gate_id}: supersession requires notes"
        return None
    return f"{res.gate_id}: unknown resolution kind {res.kind}"


def assert_ebh7_gate_findings_resolution_complete() -> None:
    errors: list[str] = []
    for finding in EBH7_CORRECTIVE_GATE_FINDINGS:
        res = finding_resolution(finding.gate_id)
        if res is None:
            errors.append(f"{finding.gate_id}: no resolution evidence — absence ≠ resolved")
            continue
        if finding.gate_id == "B6-F09-FINDING-RESOLUTION-EVIDENCE":
            continue
        err = validate_finding_resolution(res)
        if err:
            errors.append(err)
    assert not errors, "; ".join(errors[:10]) + (f" (+{len(errors) - 10})" if len(errors) > 10 else "")
