# © Artur Czarnecki. All rights reserved.

"""B6-F10 C1/C4 closure gates (T-IGX-036-Y) — closed-stage family evidence, C4 path classification, profile coverage.

All checks run on identity-neutral canonical inputs of the pinned range: ``identity_violations`` is emptied because the
executable-tree identity gate is, by design, broken between this batch and the final re-pin (wave-2 decision 6). The
evaluation code and every other canonical input stay the real ones.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.qualification.ebh_7._ebh7_f10_canonical_inputs import (
    C4_RULES,
    F10Inputs,
    cached_canonical_f10_inputs,
)
from tests.qualification.ebh_7._ebh7_f10_ecap_normative_matrix import (
    certify_f10,
    classify_change,
)
from tests.qualification.ebh_7._ebh7_f10_types import (
    ChangeClass,
    EcapGate,
    EvidenceForm,
    GateVerdict,
    QualificationResult,
    ResultStatus,
)
from tests.qualification.ebh_7._ebh7_l3_profile_coverage import argv_covers_pytest_node

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_AUD = "tests/qualification/ent_audit_x/unit"
_EVOL = "tests/qualification/evol_x/frz"
_EXPECTED_EVIDENCE: dict[str, tuple[str, ...]] = {
    "AUD": (
        f"{_AUD}/test_frz_coverage_matrix_complete.py",
        f"{_AUD}/test_frz_aud_03_unmapped_neg.py",
        f"{_AUD}/test_frz_aud_w402_neg.py",
    ),
    "EVOL": (
        f"{_EVOL}/test_evol_x_frz_01_semantic_impact.py",
        f"{_EVOL}/test_evol_x_frz_02_graph_coverage.py",
        f"{_EVOL}/test_evol_x_frz_03_blast_radius.py",
        f"{_EVOL}/test_evol_x_frz_04_contract_consumers.py",
        f"{_EVOL}/test_evol_x_frz_05_workflow_evolution.py",
        f"{_EVOL}/test_evol_x_frz_06_regression_selection.py",
        f"{_EVOL}/test_evol_x_frz_07_deploy_rollback.py",
        f"{_EVOL}/test_evol_x_frz_08_change_diagnostics.py",
        f"{_EVOL}/test_evol_x_frz_09_extensibility.py",
        f"{_EVOL}/test_evol_x_frz_10_requalification.py",
    ),
}
_CONTRACT_REGRESSION = "tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py"
_CATCH_ALL_PREFIXES = frozenset({"", "/", "intergrax/", "agents/", "applications/", "scripts/"})
_SYNTHETIC_UNKNOWN_PATHS = (
    "intergrax/zz_unclassified/x.py",
    "agents/zz_unclassified/x.py",
    "scripts/zz_unclassified/x.py",
    "governance_zz/x.md",
)


def _neutral() -> F10Inputs:
    """Canonical inputs of the pinned range with the (intentionally broken until re-pin) tree identity neutralised."""
    return replace(cached_canonical_f10_inputs(), identity_violations=())


def _result(inputs: F10Inputs, obligation_id: str) -> QualificationResult:
    cert = certify_f10(inputs)
    return next(r for r in cert.results if r.obligation.obligation_id == obligation_id)


def _evidence_files() -> tuple[str, ...]:
    return tuple(node for nodes in _EXPECTED_EVIDENCE.values() for node in nodes)


# --- C1: closed-stage families -----------------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(_EXPECTED_EVIDENCE))
def test_f10_c1_closed_stage_family_is_executed_with_bound_evidence(family: str) -> None:
    result = _result(_neutral(), f"C1-FAM-{family}")
    assert result.status is ResultStatus.PASS, result.reasons
    assert result.binding is not None
    assert result.binding.form is EvidenceForm.EXECUTED
    assert result.binding.evidence_nodes == _EXPECTED_EVIDENCE[family]


@pytest.mark.parametrize("family", sorted(_EXPECTED_EVIDENCE))
def test_f10_c1_closed_stage_family_without_evidence_mapping_fails(family: str) -> None:
    inputs = replace(_neutral(), c1_closed_stage_family_evidence={})
    result = _result(inputs, f"C1-FAM-{family}")
    assert result.status is ResultStatus.FAIL
    assert any("already CLOSED — evidence required" in reason for reason in result.reasons), result.reasons
    assert any("no executable evidence" in reason for reason in result.reasons), result.reasons
    assert result.binding is not None and result.binding.form is not EvidenceForm.NA_WITH_EVIDENCE


@pytest.mark.parametrize("family", sorted(_EXPECTED_EVIDENCE))
def test_f10_c1_closed_stage_family_with_profile_stripped_of_evidence_fails(family: str) -> None:
    base = _neutral()
    stripped = tuple(arg for arg in base.profile_argv if arg not in _evidence_files())
    result = _result(replace(base, profile_argv=stripped), f"C1-FAM-{family}")
    assert result.status is ResultStatus.FAIL
    assert any("not executed by MANIFEST pytest_profile" in reason for reason in result.reasons), result.reasons


@pytest.mark.parametrize("family", sorted(_EXPECTED_EVIDENCE))
def test_f10_c1_closed_stage_family_with_unresolvable_evidence_node_fails(family: str) -> None:
    bogus = {family: ("tests/qualification/evol_x/frz/test_zz_missing_evidence_node.py",)}
    result = _result(replace(_neutral(), c1_closed_stage_family_evidence=bogus), f"C1-FAM-{family}")
    assert result.status is ResultStatus.FAIL
    assert any("evidence node does not resolve" in reason for reason in result.reasons), result.reasons


def test_f10_c1_closed_stage_family_is_never_not_applicable() -> None:
    base = _neutral()
    variants = (
        base,
        replace(base, c1_closed_stage_family_evidence={}),
        replace(base, profile_argv=()),
        replace(base, c1_closed_stage_family_evidence={"AUD": (), "EVOL": ()}),
    )
    for inputs in variants:
        for family in _EXPECTED_EVIDENCE:
            result = _result(inputs, f"C1-FAM-{family}")
            assert result.status is not ResultStatus.NA_WITH_EVIDENCE, (family, result.reasons)
            assert result.binding is not None and result.binding.form is EvidenceForm.EXECUTED


def test_f10_c1_open_stage_families_keep_the_not_applicable_path() -> None:
    cert = certify_f10(_neutral())
    not_applicable = [
        r for r in cert.results if r.obligation.gate is EcapGate.C1 and r.status is ResultStatus.NA_WITH_EVIDENCE
    ]
    assert len(not_applicable) == 2, [r.obligation.obligation_id for r in not_applicable]
    assert not {f"C1-FAM-{family}" for family in _EXPECTED_EVIDENCE} & {
        r.obligation.obligation_id for r in not_applicable
    }
    for result in not_applicable:
        assert result.binding is not None and result.binding.form is EvidenceForm.NA_WITH_EVIDENCE


def test_f10_c1_family_with_non_closed_reason_still_fails_despite_evidence() -> None:
    base = _neutral()
    broken = replace(base, roadmap_stage_order={k: v for k, v in base.roadmap_stage_order.items() if k != "EVOL-X"})
    result = _result(broken, "C1-FAM-EVOL")
    assert result.status is ResultStatus.FAIL
    assert any("not in roadmap stage table" in reason for reason in result.reasons), result.reasons


# --- C4: classification ------------------------------------------------------------------------------------------


def test_f10_c4_no_path_of_pinned_range_is_unknown() -> None:
    inputs = _neutral()
    assert inputs.changed_paths, "pinned range produced no changed paths"
    unknown = [p for p in inputs.changed_paths if classify_change(p, inputs) == (ChangeClass.UNKNOWN,)]
    assert unknown == [], unknown[:20]


def test_f10_c4_every_rule_has_a_stated_reason() -> None:
    missing = [rule.prefix for rule in C4_RULES if not rule.reason.strip()]
    assert missing == []


def test_f10_c4_no_catch_all_rule() -> None:
    offenders = [rule.prefix for rule in C4_RULES if rule.prefix in _CATCH_ALL_PREFIXES]
    assert offenders == []
    prefixes = [rule.prefix for rule in C4_RULES]
    assert len(prefixes) == len(set(prefixes)), "duplicate C4 rule prefixes"
    assert all(rule.classes and ChangeClass.UNKNOWN not in rule.classes for rule in C4_RULES)


@pytest.mark.parametrize("path", _SYNTHETIC_UNKNOWN_PATHS)
def test_f10_c4_synthetic_path_in_new_directory_stays_unknown_and_blocks(path: str) -> None:
    base = _neutral()
    assert classify_change(path, base) == (ChangeClass.UNKNOWN,)
    inputs = replace(base, changed_paths=(*base.changed_paths, path))
    result = _result(inputs, "C4-UNCLASSIFIED")
    assert result.status is ResultStatus.FAIL
    assert any(f"unclassified change {path}" in reason for reason in result.reasons), result.reasons
    assert certify_f10(inputs).verdict(EcapGate.C4).verdict is GateVerdict.BLOCKED


def test_f10_c4_pinned_range_has_no_unclassified_reason() -> None:
    result = _result(_neutral(), "C4-UNCLASSIFIED")
    assert result.status is ResultStatus.PASS, result.reasons[:10]
    assert result.reasons == ()


def test_f10_c4_contract_regression_is_bound_to_the_replay_profile() -> None:
    inputs = _neutral()
    assert _CONTRACT_REGRESSION in inputs.c4_class_regressions[ChangeClass.CONTRACT]
    assert argv_covers_pytest_node(_CONTRACT_REGRESSION, inputs.profile_argv)


def test_f10_c4_every_used_class_has_regression_nodes_bound_to_the_replay_profile() -> None:
    inputs = _neutral()
    used = {change_class for rule in C4_RULES for change_class in rule.classes}
    unbound: list[str] = []
    for change_class in sorted(used, key=lambda c: c.value):
        nodes = inputs.c4_class_regressions.get(change_class, ())
        assert nodes, f"class {change_class.value} has no regression nodes"
        unbound += [f"{change_class.value}: {n}" for n in nodes if not argv_covers_pytest_node(n, inputs.profile_argv)]
    assert unbound == []


def test_f10_c1_and_c4_verdicts_are_pass_on_neutral_canonical_inputs() -> None:
    cert = certify_f10(_neutral())
    for gate in (EcapGate.C1, EcapGate.C4):
        verdict = cert.verdict(gate)
        assert verdict.verdict is GateVerdict.PASS, (gate, verdict.failed)
        assert verdict.failed == ()
