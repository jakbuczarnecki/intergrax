# © Artur Czarnecki. All rights reserved.

"""B6-F10 C4 classifier entries for three paths that the P8 repair loop made UNKNOWN (T-IGX-041-O).

PR 584 (T-IGX-041-M) and PR 583 (T-IGX-041-N) changed three paths that no rule classified. Each now has one exact-path
rule. The default classification is unchanged: a path under no rule is still ``UNKNOWN`` and still blocks
``C4-UNCLASSIFIED``. All checks run on identity-neutral canonical inputs, as in the T-IGX-041-K gates.
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
    GateVerdict,
    QualificationResult,
    ResultStatus,
)
from tests.qualification.ebh_7._ebh7_l3_profile_coverage import argv_covers_pytest_node

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_CLIENTS = "intergrax/integrations/_shared/p3/clients.py"
_COMPOSITION = "platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py"
_DESCRIPTOR = "platform_proofs/scenarios/indirect_prompt_injection/proof.json"

_EXPECTED: dict[str, tuple[ChangeClass, ...]] = {
    _CLIENTS: (ChangeClass.VENDOR_BACKEND, ChangeClass.SECURITY_GOVERNANCE_TENANT),
    _COMPOSITION: (ChangeClass.QUALIFICATION_INFRASTRUCTURE,),
    _DESCRIPTOR: (ChangeClass.QUALIFICATION_INFRASTRUCTURE,),
}
_SIBLING_PATHS = (
    "intergrax/integrations/_shared/p3/zz_unclassified.py",
    "intergrax/integrations/_shared/p2/clients.py",
    "platform_proofs/scenarios/ai_incident_investigation/application/zz_unclassified.py",
    "platform_proofs/scenarios/indirect_prompt_injection/zz_unclassified.json",
    "platform_proofs/scenarios/zz_unclassified/proof.json",
)


def _neutral() -> F10Inputs:
    """Canonical inputs of the pinned range with the (intentionally broken until re-pin) tree identity neutralised."""
    return replace(cached_canonical_f10_inputs(), identity_violations=())


def _result(inputs: F10Inputs, obligation_id: str) -> QualificationResult:
    cert = certify_f10(inputs)
    return next(r for r in cert.results if r.obligation.obligation_id == obligation_id)


def _without_entries(inputs: F10Inputs) -> F10Inputs:
    return replace(inputs, c4_rules=tuple(rule for rule in inputs.c4_rules if rule.prefix not in _EXPECTED))


# --- negatives: the default is unchanged ------------------------------------------------------------------------


@pytest.mark.parametrize("path", _SIBLING_PATHS)
def test_f10_c4_041o_sibling_of_an_entry_stays_unknown_and_blocks(path: str) -> None:
    base = _neutral()
    assert classify_change(path, base) == (ChangeClass.UNKNOWN,)
    inputs = replace(base, changed_paths=(*base.changed_paths, path))
    result = _result(inputs, "C4-UNCLASSIFIED")
    assert result.status is ResultStatus.FAIL
    assert any(f"unclassified change {path}" in reason for reason in result.reasons), result.reasons
    assert certify_f10(inputs).verdict(EcapGate.C4).verdict is GateVerdict.BLOCKED


def test_f10_c4_041o_default_is_unknown_and_no_rule_carries_unknown() -> None:
    inputs = _neutral()
    assert classify_change("zz_041o_no_rule/x.py", inputs) == (ChangeClass.UNKNOWN,)
    assert all(ChangeClass.UNKNOWN not in rule.classes for rule in inputs.c4_rules)


def test_f10_c4_041o_each_entry_occurs_exactly_once() -> None:
    prefixes = [rule.prefix for rule in C4_RULES]
    for path in _EXPECTED:
        assert prefixes.count(path) == 1, path


def test_f10_c4_041o_removing_the_entries_reopens_exactly_the_three_paths() -> None:
    base = _neutral()
    paths = tuple(_EXPECTED)
    closed = replace(base, changed_paths=(*base.changed_paths, *paths))
    assert _result(closed, "C4-UNCLASSIFIED").status is ResultStatus.PASS
    assert certify_f10(closed).verdict(EcapGate.C4).verdict is GateVerdict.PASS

    reopened = _without_entries(closed)
    result = _result(reopened, "C4-UNCLASSIFIED")
    assert result.status is ResultStatus.FAIL
    for path in paths:
        assert sum(1 for reason in result.reasons if reason == f"unclassified change {path}") == 1, result.reasons
    assert len(result.reasons) == len(paths), result.reasons
    assert certify_f10(reopened).verdict(EcapGate.C4).verdict is GateVerdict.BLOCKED


# --- positives: the three entries -------------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(_EXPECTED))
def test_f10_c4_041o_path_is_classified_as_documented(path: str) -> None:
    assert classify_change(path, _neutral()) == _EXPECTED[path]


@pytest.mark.parametrize("path", sorted(_EXPECTED))
def test_f10_c4_041o_entry_is_an_exact_path_with_a_reason(path: str) -> None:
    rules = [rule for rule in C4_RULES if rule.prefix == path]
    assert len(rules) == 1, path
    assert not path.endswith("/")
    assert rules[0].reason.strip()
    assert rules[0].classes == _EXPECTED[path]


def test_f10_c4_041o_classes_have_regression_nodes_bound_to_the_replay_profile() -> None:
    inputs = _neutral()
    unbound: list[str] = []
    for change_class in {c for classes in _EXPECTED.values() for c in classes}:
        nodes = inputs.c4_class_regressions.get(change_class, ())
        assert nodes, f"class {change_class.value} has no regression nodes"
        unbound += [f"{change_class.value}: {n}" for n in nodes if not argv_covers_pytest_node(n, inputs.profile_argv)]
    assert unbound == []
