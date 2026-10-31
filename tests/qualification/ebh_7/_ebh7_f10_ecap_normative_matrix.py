# © Artur Czarnecki. All rights reserved.

"""B6-F10 — ECAP C1–C5 certification computed from closed normative sets (no declared verdicts)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

from tests.qualification.ebh_5._proof_validation import pytest_test_function_exists
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification
from tests.qualification.ebh_7._ebh7_f10_canonical_inputs import (
    AGENTS_REL,
    DUP_X_CLASS_MATRIX_REL,
    DUP_X_INDEPENDENT_ACCEPTANCE_SHA,
    EBH7_FAMILY_MATRIX_REL,
    ECAP_PROTOCOL_REL,
    ECAP_REPORT_REL,
    F10_MATRIX_REL,
    FRZ_CHECKLIST_REL,
    ROADMAP_REL,
    F10Inputs,
    cached_canonical_f10_inputs,
)
from tests.qualification.ebh_7._ebh7_f10_types import (
    AttestationMode,
    ChangeClass,
    CoverageVerdict,
    EcapGate,
    EvidenceBinding,
    EvidenceForm,
    ExecutionAttestation,
    GateVerdict,
    NormativeObligation,
    QualificationResult,
    ResultStatus,
)
from tests.qualification.ebh_7._ebh7_l3_profile_coverage import argv_covers_pytest_node

_REPO: Final[Path] = Path(__file__).resolve().parents[3]
_F10_GATE_NODE: Final[str] = (
    "tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py::test_b6_f10_normative_certification_computed"
)
_DUP_X_EVIDENCE: Final[tuple[str, ...]] = (
    "tests/qualification/dup_x/test_dup_x_closed_world_integrity.py",
    "tests/qualification/dup_x/test_dup_x_parent_closure_mechanical_gates.py",
    "tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py",
)
_EBH5_L2: Final[str] = "tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py"
_EBH6_REG: Final[str] = "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py"
_EBH6_CFG: Final[str] = "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py"
_TIER_EVIDENCE: Final[dict[str, tuple[str, ...]]] = {
    "Tier-0": (
        "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py",
        "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py::test_ebh5_rpl_g_tier0_no_agents_import_applications",
        _F10_GATE_NODE,
    ),
    "Tier-1": (
        "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py",
        "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
        _F10_GATE_NODE,
    ),
    "Tier-2": (_F10_GATE_NODE,),
    "Tier-3": (
        "tests/qualification/dup_x/test_dup_x_parent_closure_mechanical_gates.py::test_dup_x_tier3_host_count_matches_disposition_doc",
        _F10_GATE_NODE,
    ),
}
_TIER_IN_PROCESS: Final[dict[str, tuple[str, ...]]] = {"Tier-3": ("dup_x.tier3_host_factory_census",)}
_RESOLVED_DUP_STATUSES: Final[frozenset[str]] = frozenset({"PASS", "RATIFIED", "N/A WITH EVIDENCE"})
_KNOWN_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(c.value for c in ExtensibilityClassification)
_SELF_REFERENTIAL: Final[frozenset[str]] = frozenset({"C5-MATRIX-SYNC"})
REPORT_BLOCK_BEGIN: Final[str] = (
    "<!-- F10-COMPUTED:BEGIN (generated — scripts/qualification/render_ebh7_f10_normative_matrix.py) -->"
)
REPORT_BLOCK_END: Final[str] = "<!-- F10-COMPUTED:END -->"
DISPOSITION_QUALIFIED: Final[str] = "EBH-7 — B6-F10 NORMATIVELY QUALIFIED / READY FOR INDEPENDENT B6 ACCEPTANCE"
DISPOSITION_BLOCKED: Final[str] = "EBH-7 — B6-F10 BLOCKED / EXACT MISSING OBLIGATIONS"
_FALSE_PASS_RE = re.compile(r"ECAP-C[1-5][^\n]*\bPASS\b")


@dataclass(frozen=True, slots=True)
class F10Certification:
    test_execution_sha: str
    results: tuple[QualificationResult, ...]
    attestations: tuple[ExecutionAttestation, ...]
    verdicts: tuple[CoverageVerdict, ...]

    def verdict(self, gate: EcapGate) -> CoverageVerdict:
        return next(v for v in self.verdicts if v.gate is gate)

    @property
    def disposition(self) -> str:
        blocked = any(v.verdict is GateVerdict.BLOCKED for v in self.verdicts)
        return DISPOSITION_BLOCKED if blocked else DISPOSITION_QUALIFIED

    def failed_obligation_ids(self) -> tuple[str, ...]:
        return tuple(r.obligation.obligation_id for r in self.results if r.status is ResultStatus.FAIL)


# --- Execution attestation --------------------------------------------------------------------------------------


@cache
def _evidence_exists(evidence_id: str) -> bool:
    if "::" in evidence_id:
        return pytest_test_function_exists(evidence_id.split("[", 1)[0])
    path = _REPO / evidence_id
    return path.is_file() and path.name.startswith("test_") and "def test_" in path.read_text(encoding="utf-8")


def _attest_node(evidence_id: str, inputs: F10Inputs) -> ExecutionAttestation:
    problems: list[str] = []
    if not _evidence_exists(evidence_id):
        problems.append("evidence node does not resolve")
    if not argv_covers_pytest_node(evidence_id, inputs.profile_argv):
        problems.append("not executed by MANIFEST pytest_profile")
    if inputs.identity_violations:
        problems.append("executed tree differs from TEST_EXECUTION_SHA: " + ", ".join(inputs.identity_violations[:3]))
    return ExecutionAttestation(
        evidence_id, AttestationMode.PROFILE_BOUND, inputs.head_sha, not problems, "; ".join(problems)
    )


def _attest_in_process(check: str, inputs: F10Inputs) -> ExecutionAttestation:
    problems: list[str] = []
    if check in inputs.in_process_failures:
        problems.append(inputs.in_process_failures[check])
    if inputs.identity_violations:
        problems.append("executed tree differs from TEST_EXECUTION_SHA")
    return ExecutionAttestation(check, AttestationMode.IN_PROCESS, inputs.head_sha, not problems, "; ".join(problems))


def _evaluate(
    obligation: NormativeObligation,
    binding: EvidenceBinding | None,
    inputs: F10Inputs,
    sink: list[ExecutionAttestation],
    extra_reasons: Iterable[str] = (),
) -> QualificationResult:
    reasons = list(extra_reasons)
    if binding is None:
        reasons.append("no evidence binding")
        return QualificationResult(obligation, None, ResultStatus.FAIL, tuple(reasons))
    if not binding.owner.strip():
        reasons.append("missing owner")
    if not binding.classification.strip():
        reasons.append("missing classification")
    if binding.form is EvidenceForm.NA_WITH_EVIDENCE:
        if not binding.rationale.strip():
            reasons.append("N/A without rationale")
    else:
        if not binding.evidence_nodes and not binding.in_process_checks:
            reasons.append("no executable evidence")
        if binding.form is EvidenceForm.HISTORICAL_WITH_DELTA:
            if not binding.historical_acceptance_sha:
                reasons.append("historical evidence without accepted SHA")
            if not binding.delta_check:
                reasons.append("historical evidence without current delta check")
    for node in binding.evidence_nodes:
        attestation = _attest_node(node, inputs)
        sink.append(attestation)
        if not attestation.passed:
            reasons.append(f"{node}: {attestation.detail}")
    for check in binding.in_process_checks:
        attestation = _attest_in_process(check, inputs)
        sink.append(attestation)
        if not attestation.passed:
            reasons.append(f"{check}: {attestation.detail}")
    if reasons:
        status = ResultStatus.FAIL
    elif binding.form is EvidenceForm.NA_WITH_EVIDENCE:
        status = ResultStatus.NA_WITH_EVIDENCE
    else:
        status = ResultStatus.PASS
    return QualificationResult(obligation, binding, status, tuple(reasons))


def _ob(obligation_id: str, gate: EcapGate, subject: str, source: str) -> NormativeObligation:
    return NormativeObligation(obligation_id, gate, subject, source)


# --- C1: Tier-0..3 + FRZ architecture families -----------------------------------------------------------------


def _c1_family_na_reasons(family: str, inputs: F10Inputs) -> tuple[list[str], tuple[str, ...]]:
    stages = inputs.ecap_closing_stages.get(family, ())
    reasons: list[str] = []
    if not stages:
        return [f"FRZ-{family} absent from family catalog and has no ECAP closing stage"], stages
    ebh7_index = inputs.roadmap_stage_order.get("EBH-7")
    for token in stages:
        if token.endswith("*"):
            matched = [s for s in inputs.roadmap_stage_order if s.startswith(token[:-1])]
        else:
            matched = [token] if token in inputs.roadmap_stage_order else []
        if token.rstrip("*") == "EBH-7":
            reasons.append("closing stage is EBH-7 — family coverage required")
        elif any(s in inputs.roadmap_closed_stages for s in matched):
            reasons.append(f"closing stage {token} already CLOSED — evidence required, N/A invalid")
        elif not matched or ebh7_index is None:
            reasons.append(f"closing stage {token} not in roadmap stage table")
        elif any(inputs.roadmap_stage_order[s] <= ebh7_index for s in matched):
            reasons.append(f"closing stage {token} is not downstream of EBH-7")
    return reasons, stages


def _tier_violations(tier_id: str, root: str, inputs: F10Inputs) -> tuple[list[str], list[str]]:
    runtime_root = next((t.root for t in inputs.tiers if t.tier_id == "Tier-1"), "intergrax/runtime/")
    scoped = []
    for path, module in inputs.tier_import_violations:
        if tier_id == "Tier-1" and not path.startswith(runtime_root):
            continue
        if tier_id == "Tier-0" and path.startswith(runtime_root):
            continue
        if path.startswith(root):
            scoped.append((path, module))
    unratified: list[str] = []
    ratified: list[str] = []
    for path, module in scoped:
        exc = next(
            (e for e in inputs.tier_import_exceptions if e.path == path and module.startswith(e.module_prefix)),
            None,
        )
        doc = _REPO / exc.evidence_doc if exc else None
        if exc and doc and doc.is_file() and exc.path in doc.read_text(encoding="utf-8"):
            ratified.append(f"{path} → {module} (ratified: {exc.evidence_doc})")
        else:
            unratified.append(f"tier import violation {path} → {module}")
    return unratified, ratified


def evaluate_c1(inputs: F10Inputs, sink: list[ExecutionAttestation]) -> list[QualificationResult]:
    results: list[QualificationResult] = []
    frz = set(inputs.frz_families)
    ebh6 = {f.family_id: f for f in inputs.ebh6_families}
    matrix_ids = inputs.ebh7_family_matrix_ids

    identity_reasons: list[str] = []
    if len(matrix_ids) != len(set(matrix_ids)):
        identity_reasons.append("duplicate family IDs in EBH-7 matrix")
    if set(matrix_ids) != set(ebh6):
        identity_reasons.append(f"EBH-6 catalog vs EBH-7 matrix ID mismatch: {sorted(set(ebh6) ^ set(matrix_ids))}")
    if len(frz) != 26:
        identity_reasons.append(f"FRZ checklist family set size {len(frz)} != 26")
    results.append(
        _evaluate(
            _ob("C1-FAMILY-ID-IDENTITY", EcapGate.C1, "FRZ families ↔ EBH-6 catalog ↔ EBH-7 matrix", FRZ_CHECKLIST_REL),
            EvidenceBinding(
                "C1-FAMILY-ID-IDENTITY",
                owner="EBH-6 family catalog (SSOT) + EBH-7 family matrix",
                classification="ID set identity",
                form=EvidenceForm.EXECUTED,
                evidence_nodes=(
                    "tests/qualification/ebh_6/test_ebh_6_architecture_family_matrix_integrity.py",
                    _F10_GATE_NODE,
                ),
            ),
            inputs,
            sink,
            identity_reasons,
        )
    )
    for family in sorted(frz):
        ob = _ob(f"C1-FAM-{family}", EcapGate.C1, f"FRZ-{family}-*", FRZ_CHECKLIST_REL)
        record = ebh6.get(family)
        if record is not None:
            extra = [] if family in matrix_ids else ["family missing from EBH-7 enterprise family matrix"]
            if not record.gate_nodes:
                extra.append("family has no gate nodes")
            binding = EvidenceBinding(
                ob.obligation_id,
                owner=record.owner,
                classification="EBH-6 architecture family (current-HEAD gates)",
                form=EvidenceForm.EXECUTED,
                evidence_nodes=record.gate_nodes,
            )
            results.append(_evaluate(ob, binding, inputs, sink, extra))
            continue
        reasons, stages = _c1_family_na_reasons(family, inputs)
        binding = EvidenceBinding(
            ob.obligation_id,
            owner=f"ECAP §3 FRZ-{family} closing stage(s): {', '.join(stages) or '—'}",
            classification="DEFERRED-DOWNSTREAM (not an EBH-7 closing family)",
            form=EvidenceForm.NA_WITH_EVIDENCE,
            evidence_nodes=(),
            rationale=f"{ECAP_PROTOCOL_REL} primary closing stage(s) open and downstream of EBH-7 in {ROADMAP_REL}",
        )
        results.append(_evaluate(ob, binding, inputs, sink, reasons))
    for family in sorted(set(ebh6) - frz):
        ob = _ob(f"C1-ROLLUP-{family}", EcapGate.C1, f"non-FRZ catalog row {family}", EBH7_FAMILY_MATRIX_REL)
        target = inputs.rollup_families.get(family)
        extra = [] if target else [f"unknown family {family} not in FRZ checklist and not a declared rollup"]
        binding = EvidenceBinding(
            ob.obligation_id,
            owner=ebh6[family].owner,
            classification=f"rollup → ECAP {target}" if target else "",
            form=EvidenceForm.EXECUTED,
            evidence_nodes=ebh6[family].gate_nodes,
        )
        results.append(_evaluate(ob, binding, inputs, sink, extra))

    tier_ids = tuple(t.tier_id for t in inputs.tiers)
    if tier_ids != ("Tier-0", "Tier-1", "Tier-2", "Tier-3"):
        results.append(
            _evaluate(
                _ob("C1-TIER-SET", EcapGate.C1, "Tier-0..3", AGENTS_REL),
                None,
                inputs,
                sink,
                [f"canonical tier set mismatch: {tier_ids}"],
            )
        )
    stale = [
        e.path
        for e in inputs.tier_import_exceptions
        if not any(p == e.path and m.startswith(e.module_prefix) for p, m in inputs.tier_import_violations)
    ]
    for tier in inputs.tiers:
        ob = _ob(f"C1-{tier.tier_id.upper()}", EcapGate.C1, f"{tier.tier_id} {tier.root}", AGENTS_REL)
        extra: list[str] = []
        if tier.root not in inputs.existing_paths:
            extra.append(f"tier root {tier.root} does not resolve")
        unratified, ratified = _tier_violations(tier.tier_id, tier.root, inputs)
        extra.extend(unratified)
        if tier.tier_id == "Tier-0":
            extra.extend(f"stale tier exception {p}" for p in stale)
        binding = EvidenceBinding(
            ob.obligation_id,
            owner=f"{tier.root} (AGENTS.md hard boundaries)",
            classification="tier import boundary" + (f"; ratified: {'; '.join(ratified)}" if ratified else ""),
            form=EvidenceForm.EXECUTED,
            evidence_nodes=_TIER_EVIDENCE.get(tier.tier_id, ()),
            in_process_checks=_TIER_IN_PROCESS.get(tier.tier_id, ()),
        )
        results.append(_evaluate(ob, binding, inputs, sink, extra))
    return results


# --- C2: duplicate / shadow / bypass ---------------------------------------------------------------------------


def _resolves(path: str, inputs: F10Inputs) -> bool:
    return bool(path) and path in inputs.existing_paths


def evaluate_c2(inputs: F10Inputs, sink: list[ExecutionAttestation]) -> list[QualificationResult]:
    results: list[QualificationResult] = []
    delta_sha_ok = inputs.ancestry.get((DUP_X_INDEPENDENT_ACCEPTANCE_SHA, inputs.test_execution_sha), False)
    doc_classes = set(inputs.dup_class_doc_results)
    identity: list[str] = []
    if set(inputs.dup_classes) != doc_classes:
        identity.append(f"DUP class set mismatch vs class matrix: {sorted(set(inputs.dup_classes) ^ doc_classes)}")
    if len(inputs.dup_classes) != 14:
        identity.append(f"DUP class count {len(inputs.dup_classes)} != 14 (DUP-01..14)")
    if not delta_sha_ok:
        identity.append("DUP-X acceptance SHA is not an ancestor of TEST_EXECUTION_SHA")
    results.append(
        _evaluate(
            _ob("C2-DUP-CLASS-IDENTITY", EcapGate.C2, "DUP-01..14 closed set", DUP_X_CLASS_MATRIX_REL),
            EvidenceBinding(
                "C2-DUP-CLASS-IDENTITY",
                owner="DUP-X findings register",
                classification="class set identity",
                form=EvidenceForm.HISTORICAL_WITH_DELTA,
                evidence_nodes=_DUP_X_EVIDENCE,
                in_process_checks=("dup_x.closed_world_integrity",),
                historical_acceptance_sha=DUP_X_INDEPENDENT_ACCEPTANCE_SHA,
                delta_check="dup_x.closed_world_integrity @ executed HEAD",
            ),
            inputs,
            sink,
            identity,
        )
    )
    for dup_class in sorted(set(inputs.dup_classes) | doc_classes):
        ob = _ob(f"C2-{dup_class}", EcapGate.C2, dup_class, DUP_X_CLASS_MATRIX_REL)
        records = [r for r in inputs.dup_records if dup_class in r.dup_classes]
        doc = inputs.dup_class_doc_results.get(dup_class, "")
        extra = [f"{r.record_id} status {r.status}" for r in records if r.status not in _RESOLVED_DUP_STATUSES]
        if dup_class not in inputs.dup_classes:
            extra.append("class present in matrix but missing from DUP_CLASSES SSOT")
        if not doc:
            extra.append("class missing from DUP-X class coverage matrix")
        elif not records and "N/A WITH EVIDENCE" not in doc:
            extra.append("no finding/candidate record and no N/A WITH EVIDENCE disposition")
        binding = EvidenceBinding(
            ob.obligation_id,
            owner="DUP-X findings register: " + (", ".join(r.record_id for r in records) or "class matrix N/A"),
            classification=doc[:80],
            form=EvidenceForm.HISTORICAL_WITH_DELTA,
            evidence_nodes=_DUP_X_EVIDENCE,
            in_process_checks=("dup_x.closed_world_integrity",),
            historical_acceptance_sha=DUP_X_INDEPENDENT_ACCEPTANCE_SHA if delta_sha_ok else "",
            delta_check="closed-world census re-executed @ HEAD",
        )
        results.append(_evaluate(ob, binding, inputs, sink, extra))

    rf_results: dict[str, ResultStatus] = {}
    for rf in inputs.rf_families:
        ob = _ob(
            f"C2-{rf.family_id}", EcapGate.C2, rf.family_id, "tests/qualification/dup_x/_closed_world_inventory.py"
        )
        delta = inputs.rf_owner_delta.get(rf.family_id)
        extra: list[str] = []
        owner_doc = rf.owner_doc if _resolves(rf.owner_doc, inputs) else (delta.owner_doc if delta else "")
        impl = rf.implementation_path if _resolves(rf.implementation_path, inputs) else ""
        if not impl and delta and delta.implementation_path:
            impl = delta.implementation_path
        checks: tuple[str, ...] = ()
        if delta and delta.discovery_check:
            checks = (delta.discovery_check,)
        if not _resolves(owner_doc, inputs):
            extra.append(f"owner doc unresolved ({rf.owner_doc})")
        if not _resolves(impl, inputs) and not checks:
            extra.append(f"implementation owner unresolved ({rf.implementation_path})")
        composition_ok = _resolves(rf.composition_root, inputs) or rf.composition_root.startswith("N/A") or checks
        if not composition_ok:
            extra.append(f"composition root unresolved ({rf.composition_root})")
        used_delta = owner_doc != rf.owner_doc or impl != rf.implementation_path or bool(checks)
        binding = EvidenceBinding(
            ob.obligation_id,
            owner=f"{owner_doc} · {impl or rf.implementation_path}",
            classification="current-HEAD owner delta binding" if used_delta else "accepted DUP-X SSOT owner",
            form=EvidenceForm.HISTORICAL_WITH_DELTA,
            evidence_nodes=_DUP_X_EVIDENCE[:1],
            in_process_checks=("dup_x.closed_world_integrity", *checks),
            historical_acceptance_sha=DUP_X_INDEPENDENT_ACCEPTANCE_SHA if delta_sha_ok else "",
            delta_check="owner path resolution @ HEAD",
        )
        result = _evaluate(ob, binding, inputs, sink, extra)
        rf_results[rf.family_id] = result.status
        results.append(result)

    referenced: set[str] = set()
    for surface, rf_ids in inputs.c2_surfaces:
        referenced.update(rf_ids)
        ob = _ob(f"C2-SURFACE-{surface}", EcapGate.C2, surface, "ECAP-§2.C C2 / roadmap §3.1 duplication")
        extra = [
            f"{rf} not in DUP-X responsibility families (subset claimed as full)"
            for rf in rf_ids
            if rf not in rf_results
        ]
        extra += [f"{rf} not qualified" for rf in rf_ids if rf_results.get(rf) is ResultStatus.FAIL]
        if not rf_ids:
            extra.append("surface has no responsibility family")
        binding = EvidenceBinding(
            ob.obligation_id,
            owner=", ".join(rf_ids),
            classification="surface → responsibility family coverage",
            form=EvidenceForm.EXECUTED,
            evidence_nodes=_DUP_X_EVIDENCE[:1],
        )
        results.append(_evaluate(ob, binding, inputs, sink, extra))
    orphans = sorted(set(rf_results) - referenced)
    counters = {k: v for k, v in inputs.dup_counters.items() if v}
    results.append(
        _evaluate(
            _ob(
                "C2-COUNTERS",
                EcapGate.C2,
                "duplicate/shadow/bypass/gap/candidate counters = 0",
                "dup_x/_findings_register.py",
            ),
            EvidenceBinding(
                "C2-COUNTERS",
                owner="DUP-X findings register",
                classification="unresolved counters",
                form=EvidenceForm.HISTORICAL_WITH_DELTA,
                evidence_nodes=_DUP_X_EVIDENCE,
                in_process_checks=("dup_x.env_slug_reads_outside_factory", "dup_x.production_tools_base_imports"),
                historical_acceptance_sha=DUP_X_INDEPENDENT_ACCEPTANCE_SHA if delta_sha_ok else "",
                delta_check="bypass/shadow discovery re-executed @ HEAD",
            ),
            inputs,
            sink,
            [f"counter {k}={v}" for k, v in sorted(counters.items())]
            + [f"responsibility family {o} not mapped to any normative surface" for o in orphans],
        )
    )
    return results


# --- C3: VBN domains 1–31 + discovered mechanisms --------------------------------------------------------------


def _domain_number(domain: str) -> int | None:
    head = domain.split("-", 1)[0]
    return int(head) if head.isdigit() else None


def evaluate_c3(inputs: F10Inputs, sink: list[ExecutionAttestation]) -> list[QualificationResult]:
    results: list[QualificationResult] = []
    roadmap_numbers = [n for n, _ in inputs.roadmap_vbn_domains]
    enum_numbers = [_domain_number(d) for d in inputs.vbn_domains]
    identity: list[str] = []
    expected = list(range(1, 32))
    if sorted(roadmap_numbers) != expected:
        identity.append(f"roadmap §3.0.8 domain numbering {roadmap_numbers} != 1..31")
    if sorted(n for n in enum_numbers if n is not None) != expected or None in enum_numbers:
        identity.append(f"RoadmapVendorDomain IDs {inputs.vbn_domains} do not match 1..31")
    results.append(
        _evaluate(
            _ob("C3-TAXONOMY-IDENTITY", EcapGate.C3, "roadmap §3.0.8 domains ↔ RoadmapVendorDomain", ROADMAP_REL),
            EvidenceBinding(
                "C3-TAXONOMY-IDENTITY",
                owner="EBH-5 vendor/backend inventory",
                classification="domain ID identity",
                form=EvidenceForm.EXECUTED,
                evidence_nodes=(f"{_EBH5_L2}::test_ebh5_q04_all_records_have_owner_and_domain", _F10_GATE_NODE),
                in_process_checks=("ebh5.inventory_closed_world",),
            ),
            inputs,
            sink,
            identity,
        )
    )
    domains = set(inputs.vbn_domains)
    for domain in inputs.vbn_domains:
        ob = _ob(f"C3-VBN-{domain}", EcapGate.C3, domain, ROADMAP_REL)
        concerns = [c for c in inputs.vendor_concerns if c.domain == domain]
        na = inputs.domain_na.get(domain)
        if concerns:
            incomplete = [c.concern_id for c in concerns if not (c.semantic_owner and c.classification)]
            binding = EvidenceBinding(
                ob.obligation_id,
                owner="; ".join(sorted({c.semantic_owner for c in concerns}))[:120],
                classification=f"{len(concerns)} concern(s): "
                + ", ".join(sorted(c.concern_id for c in concerns))[:140],
                form=EvidenceForm.EXECUTED,
                evidence_nodes=(_EBH5_L2,),
                in_process_checks=("ebh5.inventory_closed_world",),
            )
            extra = [f"concern {c} lacks owner/classification" for c in incomplete]
        elif na is not None:
            missing = [p for p in na.discovery_paths if p not in inputs.existing_paths]
            extra = [] if na.kind.startswith("N/A — WITH EVIDENCE") else [f"invalid N/A kind {na.kind!r}"]
            extra += [f"N/A discovery path missing {p}" for p in missing]
            if not na.discovery_paths:
                extra.append("N/A without discovery paths")
            binding = EvidenceBinding(
                ob.obligation_id,
                owner="EBH-5 roadmap domain N/A register",
                classification=na.kind,
                form=EvidenceForm.NA_WITH_EVIDENCE,
                evidence_nodes=(),
                rationale=na.rationale,
            )
        else:
            binding = None
            extra = ["domain neither covered by an inventory concern nor N/A WITH EVIDENCE"]
        results.append(_evaluate(ob, binding, inputs, sink, extra))

    ids = [c.concern_id for c in inputs.vendor_concerns]
    for concern in inputs.vendor_concerns:
        ob = _ob(
            f"C3-CONCERN-{concern.concern_id}", EcapGate.C3, concern.concern_id, "ebh_5/_vendor_backend_inventory.py"
        )
        extra: list[str] = []
        if concern.domain not in domains:
            extra.append(f"unknown concern: domain {concern.domain!r} outside roadmap taxonomy")
        if ids.count(concern.concern_id) > 1:
            extra.append("duplicate concern_id")
        if not concern.composition_owner.strip():
            extra.append("missing composition/selection owner")
        if concern.classification not in _KNOWN_CLASSIFICATIONS:
            extra.append(f"unknown classification {concern.classification!r}")
        if not concern.structural_evidence.strip():
            extra.append("missing structural evidence")
        if (
            concern.classification == ExtensibilityClassification.ARCHITECTURE_BLOCKER.value
            and not concern.outstanding_debt
        ):
            extra.append("ARCHITECTURE BLOCKER without tracked debt")
        binding = EvidenceBinding(
            ob.obligation_id,
            owner=concern.semantic_owner,
            classification=concern.classification,
            form=EvidenceForm.EXECUTED,
            evidence_nodes=(f"{_EBH5_L2}::test_ebh5_q04_all_records_have_owner_and_domain",),
        )
        results.append(_evaluate(ob, binding, inputs, sink, extra))

    separate = (
        (
            "C3-INVENTORY-CLOSED-WORLD",
            "EBH-5 closed-world inventory + QUALIFIED proof integrity",
            (_EBH5_L2,),
            ("ebh5.inventory_closed_world", "ebh5.qualified_proof_integrity"),
        ),
        (
            "C3-DISCOVERED-MECHANISMS",
            "domain 31 + EBH-6 universal layer discoveries classified",
            (
                f"{_EBH6_REG}::test_ebh6_findings_unclassified_zero",
                f"{_EBH6_REG}::test_ebh6_layer_coverage_findings_complete",
            ),
            ("ebh6.unclassified_zero",),
        ),
        (
            "C3-PROHIBITED-ZERO",
            "PROHIBITED unresolved = 0 (separate evidence)",
            (f"{_EBH6_REG}::test_ebh6_prohibited_unresolved_zero",),
            ("ebh6.prohibited_unresolved_zero",),
        ),
        ("C3-CONFIG-X-54", "CONFIG-X 54/54 @ HEAD (separate evidence)", (_EBH6_CFG,), ("config_x.head_reconciled_54",)),
    )
    discovered = [c for c in inputs.vendor_concerns if _domain_number(c.domain) == 31]
    for obligation_id, subject, nodes, checks in separate:
        extra = []
        if obligation_id == "C3-DISCOVERED-MECHANISMS" and not discovered:
            extra.append("no discovered-mechanism concerns (domain 31)")
        results.append(
            _evaluate(
                _ob(obligation_id, EcapGate.C3, subject, "VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md"),
                EvidenceBinding(
                    obligation_id,
                    owner="EBH-5 / EBH-6 / CONFIG-X qualification registers",
                    classification="separate VBN evidence (does not substitute domain coverage)",
                    form=EvidenceForm.EXECUTED,
                    evidence_nodes=nodes,
                    in_process_checks=checks,
                ),
                inputs,
                sink,
                extra,
            )
        )
    return results


# --- C4: drift vs accepted baseline ----------------------------------------------------------------------------


def classify_change(path: str, inputs: F10Inputs) -> tuple[ChangeClass, ...]:
    best = None
    for rule in inputs.c4_rules:
        if path == rule.prefix or (rule.prefix.endswith("/") and path.startswith(rule.prefix)):
            if best is None or len(rule.prefix) > len(best.prefix):
                best = rule
    return best.classes if best else (ChangeClass.UNKNOWN,)


def c4_classification(inputs: F10Inputs) -> dict[ChangeClass, tuple[str, ...]]:
    buckets: dict[ChangeClass, list[str]] = {}
    for path in inputs.changed_paths:
        for change_class in classify_change(path, inputs):
            buckets.setdefault(change_class, []).append(path)
    return {k: tuple(v) for k, v in buckets.items()}


def evaluate_c4(inputs: F10Inputs, sink: list[ExecutionAttestation]) -> list[QualificationResult]:
    results: list[QualificationResult] = []
    accepted, authorized, test = inputs.accepted_baseline_sha, inputs.authorized_baseline_sha, inputs.test_execution_sha
    ancestry = []
    for a, b in ((accepted, authorized), (authorized, test), (accepted, test)):
        if not inputs.ancestry.get((a, b), False):
            ancestry.append(f"{a[:12]} is not an ancestor of {b[:12]}")
    results.append(
        _evaluate(
            _ob(
                "C4-BASELINE",
                EcapGate.C4,
                f"accepted {accepted[:12]} → authorized {authorized[:12]} → TEST {test[:12]}",
                ROADMAP_REL,
            ),
            EvidenceBinding(
                "C4-BASELINE",
                owner="ROADMAP-REPLAY-X independent acceptance (last accepted baseline)",
                classification="SHA ancestry",
                form=EvidenceForm.EXECUTED,
                evidence_nodes=(_F10_GATE_NODE,),
            ),
            inputs,
            sink,
            ancestry,
        )
    )
    buckets = c4_classification(inputs)
    unknown = buckets.get(ChangeClass.UNKNOWN, ())
    results.append(
        _evaluate(
            _ob("C4-UNCLASSIFIED", EcapGate.C4, f"{len(inputs.changed_paths)} changed paths classified", ROADMAP_REL),
            EvidenceBinding(
                "C4-UNCLASSIFIED",
                owner="F10 C4 path classifier",
                classification="unknown = BLOCKER",
                form=EvidenceForm.EXECUTED,
                evidence_nodes=(_F10_GATE_NODE,),
            ),
            inputs,
            sink,
            [f"unclassified change {p}" for p in unknown],
        )
    )
    for change_class in sorted(buckets, key=lambda c: c.value):
        if change_class is ChangeClass.UNKNOWN:
            continue
        paths = buckets[change_class]
        ob = _ob(f"C4-CLASS-{change_class.value}", EcapGate.C4, f"{len(paths)} path(s)", ROADMAP_REL)
        regressions = inputs.c4_class_regressions.get(change_class, ())
        extra = [] if regressions else [f"class {change_class.value} has no required regression"]
        binding = EvidenceBinding(
            ob.obligation_id,
            owner="F10 C4 regression map",
            classification=change_class.value,
            form=EvidenceForm.EXECUTED,
            evidence_nodes=regressions,
        )
        results.append(_evaluate(ob, binding, inputs, sink, extra))
    return results


# --- C5: §14 report presents computed results ------------------------------------------------------------------


def _split_report(report_text: str) -> tuple[str, str]:
    if REPORT_BLOCK_BEGIN not in report_text or REPORT_BLOCK_END not in report_text:
        return "", report_text
    head, rest = report_text.split(REPORT_BLOCK_BEGIN, 1)
    block, tail = rest.split(REPORT_BLOCK_END, 1)
    return block.strip("\n"), head + tail


def _verdicts(results: list[QualificationResult], gates: Iterable[EcapGate]) -> tuple[CoverageVerdict, ...]:
    verdicts = []
    for gate in gates:
        rows = [r for r in results if r.obligation.gate is gate]
        failed = tuple(r.obligation.obligation_id for r in rows if r.status is ResultStatus.FAIL)
        verdicts.append(
            CoverageVerdict(
                gate,
                GateVerdict.BLOCKED if failed or not rows else GateVerdict.PASS,
                len(rows),
                sum(1 for r in rows if r.status is ResultStatus.PASS),
                sum(1 for r in rows if r.status is ResultStatus.NA_WITH_EVIDENCE),
                failed,
            )
        )
    return tuple(verdicts)


def _verdict_row(v: CoverageVerdict) -> str:
    counts = f"{v.obligations} | {v.passed} | {v.not_applicable} | {len(v.failed)}"
    return f"| {v.gate.value} | {v.verdict.value} | {counts} |"


def render_report_block(verdicts: tuple[CoverageVerdict, ...], test_execution_sha: str) -> str:
    blocked = [v for v in verdicts if v.verdict is GateVerdict.BLOCKED]
    lines = [
        f"**Computed from closed normative sets @ TEST_EXECUTION_SHA `{test_execution_sha}`** "
        "(C5 verified by read-only gate; no hand-assigned verdicts).",
        "",
        "| ECAP gate | Computed verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in verdicts:
        lines.append(_verdict_row(v))
    lines.append("")
    if blocked:
        lines.append("**C1–C4 computed disposition:** " + DISPOSITION_BLOCKED)
        for v in blocked:
            lines.append(f"- {v.gate.value} missing obligations: " + ", ".join(v.failed))
    else:
        lines.append("**C1–C4 computed disposition:** all obligations PASS or N/A WITH EVIDENCE")
    return "\n".join(lines)


def evaluate_c5(
    inputs: F10Inputs,
    upstream: tuple[CoverageVerdict, ...],
    sink: list[ExecutionAttestation],
    rendered_matrix: str | None,
) -> list[QualificationResult]:
    results: list[QualificationResult] = []
    block, outside = _split_report(inputs.report_text)
    present: list[str] = []
    if not inputs.report_text:
        present.append("ECAP §14 report missing")
    for token in ("Platform scope banner", *(f"## {s}." for s in "ABCDEFGHIJK")):
        if inputs.report_text and token not in inputs.report_text:
            present.append(f"report missing {token}")
    expected_block = render_report_block(upstream, inputs.test_execution_sha)
    computed: list[str] = []
    if not block:
        computed.append("report lacks F10 computed block markers")
    elif block != expected_block:
        computed.append("report computed block differs from C1–C4 results")
    false_pass = [m.group(0)[:80] for m in _FALSE_PASS_RE.finditer(outside)]
    if any(v.verdict is GateVerdict.BLOCKED for v in upstream) and DISPOSITION_QUALIFIED in inputs.report_text:
        false_pass.append("report claims NORMATIVELY QUALIFIED while C1–C4 blocked")
    checks = (
        ("C5-REPORT-PRESENT", "§14 template sections A–K + banner", present),
        ("C5-REPORT-COMPUTED", "report block == computed C1–C4 verdicts", computed),
        (
            "C5-NO-FALSE-PASS",
            "no static PASS claims outside computed block",
            [f"static claim: {f}" for f in false_pass],
        ),
    )
    for obligation_id, subject, extra in checks:
        results.append(
            _evaluate(
                _ob(obligation_id, EcapGate.C5, subject, ECAP_REPORT_REL),
                EvidenceBinding(
                    obligation_id,
                    owner=ECAP_REPORT_REL,
                    classification="ECAP §14 computed presentation",
                    form=EvidenceForm.EXECUTED,
                    evidence_nodes=(_F10_GATE_NODE,),
                ),
                inputs,
                sink,
                extra,
            )
        )
    if rendered_matrix is not None:
        sync = [] if inputs.matrix_disk_text == rendered_matrix else ["F10 matrix on disk differs from generated"]
        results.append(
            _evaluate(
                _ob("C5-MATRIX-SYNC", EcapGate.C5, "matrix on disk == deterministic render", F10_MATRIX_REL),
                EvidenceBinding(
                    "C5-MATRIX-SYNC",
                    owner=F10_MATRIX_REL,
                    classification="read-only comparison",
                    form=EvidenceForm.EXECUTED,
                    evidence_nodes=(
                        "tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py::"
                        "test_b6_f10_matrix_markdown_matches_generated",
                    ),
                ),
                inputs,
                sink,
                sync,
            )
        )
    return results


# --- Certification + rendering ---------------------------------------------------------------------------------


def _certify(inputs: F10Inputs, *, with_matrix_sync: bool) -> F10Certification:
    sink: list[ExecutionAttestation] = []
    results = (
        evaluate_c1(inputs, sink) + evaluate_c2(inputs, sink) + evaluate_c3(inputs, sink) + evaluate_c4(inputs, sink)
    )
    upstream = _verdicts(results, (EcapGate.C1, EcapGate.C2, EcapGate.C3, EcapGate.C4))
    base_c5 = evaluate_c5(inputs, upstream, sink, None)
    if with_matrix_sync:
        partial = F10Certification(
            inputs.test_execution_sha,
            tuple(results + base_c5),
            tuple(sink),
            upstream + _verdicts(base_c5, (EcapGate.C5,)),
        )
        c5 = evaluate_c5(inputs, upstream, [], render_f10_matrix_markdown(partial))
    else:
        c5 = base_c5
    all_results = results + c5
    return F10Certification(
        inputs.test_execution_sha,
        tuple(all_results),
        tuple(sink),
        upstream + _verdicts(c5, (EcapGate.C5,)),
    )


def certify_f10(inputs: F10Inputs | None = None) -> F10Certification:
    return _certify(inputs or cached_canonical_f10_inputs(), with_matrix_sync=True)


def certify_f10_for_render(inputs: F10Inputs | None = None) -> F10Certification:
    return _certify(inputs or cached_canonical_f10_inputs(), with_matrix_sync=False)


def _cell(text: str, limit: int = 160) -> str:
    flat = " ".join(text.replace("|", "/").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def render_f10_matrix_markdown(cert: F10Certification) -> str:
    rendered_results = [r for r in cert.results if r.obligation.obligation_id not in _SELF_REFERENTIAL]
    rendered_verdicts = _verdicts(rendered_results, tuple(EcapGate))
    blocked = any(v.verdict is GateVerdict.BLOCKED for v in rendered_verdicts)
    lines = [
        "# EBH-7 — B6-F10 ECAP normative certification (computed C1–C5)",
        "",
        "Generated by `scripts/qualification/render_ebh7_f10_normative_matrix.py` from closed normative sets; "
        "verified read-only by `test_b6_f10_matrix_markdown_matches_generated`. Not B6 PASS.",
        "",
        f"**TEST_EXECUTION_SHA:** `{cert.test_execution_sha}`",
        "",
        f"**Computed disposition:** {DISPOSITION_BLOCKED if blocked else DISPOSITION_QUALIFIED}",
        "",
        "**Canonical sources:** "
        + " · ".join(
            f"`{s}`" for s in (FRZ_CHECKLIST_REL, ECAP_PROTOCOL_REL, ROADMAP_REL, AGENTS_REL, DUP_X_CLASS_MATRIX_REL)
        ),
        "",
        "**Execution attestation:** PROFILE_BOUND = node resolves, is executed by MANIFEST `pytest_profile`, and "
        "executed tree is identical to TEST_EXECUTION_SHA outside the B5 allowlist; IN_PROCESS = validator executed "
        "inside the F10 gate. Post-run JUnit attestation: `scripts/qualification/attest_ebh7_f10_execution.py`.",
        "",
        "| Gate | Verdict | Obligations | PASS | N/A WITH EVIDENCE | FAIL |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in rendered_verdicts:
        lines.append(_verdict_row(v))
    for gate in EcapGate:
        lines += [
            "",
            f"## {gate.value}",
            "",
            "| Obligation | Subject | Owner | Classification | Form | Evidence | Status | Missing / reasons |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in (r for r in rendered_results if r.obligation.gate is gate):
            b = r.binding
            evidence = "; ".join((*b.evidence_nodes, *b.in_process_checks)) if b else ""
            lines.append(
                "| "
                + " | ".join(
                    (
                        r.obligation.obligation_id,
                        _cell(r.obligation.subject, 80),
                        _cell(b.owner if b else "—", 100),
                        _cell(b.classification if b else "—", 90),
                        b.form.value if b else "—",
                        _cell(evidence, 140) or "—",
                        r.status.value,
                        _cell("; ".join(r.reasons), 200) or "—",
                    )
                )
                + " |"
            )
    lines += [
        "",
        "C5-MATRIX-SYNC (disk == generated) is evaluated at test time and is not rendered (self-referential).",
        "",
        "**Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · EVOL-X entry · global FRZ PASS.",
        "",
    ]
    return "\n".join(lines)


def assert_ebh7_f10_ecap_normative_ready() -> None:
    cert = certify_f10()
    failures = [
        f"{r.obligation.obligation_id}: {'; '.join(r.reasons)}" for r in cert.results if r.status is ResultStatus.FAIL
    ]
    assert not failures, f"{cert.disposition}: " + " | ".join(failures[:12])


# --- Post-run JUnit attestation (GHA) -------------------------------------------------------------------------


def _junit_node(classname: str, name: str) -> str:
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        rel = "/".join(parts[:cut]) + ".py"
        if (_REPO / rel).is_file():
            return f"{rel}::{name.split('[', 1)[0]}"
    return f"{classname}::{name}"


def attest_junit(junit_xml: Path, cert: F10Certification) -> tuple[dict[str, dict[str, int]], tuple[str, ...]]:
    outcomes: dict[str, dict[str, int]] = {}
    for case in ET.parse(junit_xml).getroot().iter("testcase"):
        node = _junit_node(case.get("classname", ""), case.get("name", ""))
        failed = any(child.tag in ("failure", "error") for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        for key in (node, node.split("::", 1)[0]):
            bucket = outcomes.setdefault(key, {"passed": 0, "failed": 0, "skipped": 0})
            bucket["failed" if failed else "skipped" if skipped else "passed"] += 1
    required = sorted({a.evidence_id for a in cert.attestations if a.mode is AttestationMode.PROFILE_BOUND})
    missing: list[str] = []
    report: dict[str, dict[str, int]] = {}
    for evidence_id in required:
        bucket = outcomes.get(evidence_id.split("[", 1)[0], {"passed": 0, "failed": 0, "skipped": 0})
        report[evidence_id] = bucket
        if bucket["passed"] == 0 or bucket["failed"] or bucket["skipped"]:
            missing.append(evidence_id)
    return report, tuple(missing)
