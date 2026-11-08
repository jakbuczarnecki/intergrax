"""D-5: OD-ENT-02 verifiers read real evidence (rulesets, sub-manifest); marker text alone never passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import scripts.governance.verify_ent_audit_branch_protection as live
import scripts.governance.verify_od_ent_02_compensating_control as comp

FAST = "Enterprise audit (ENT-AUDIT-X) / ent-audit-fast"
GATE = "Enterprise audit (ENT-AUDIT-X) / ent-audit-pr-gate"


def _ruleset_rule(*contexts: str) -> list[dict[str, Any]]:
    return [
        {"type": "required_status_checks", "parameters": {"required_status_checks": [{"context": c} for c in contexts]}}
    ]


def _api(protection: tuple[int, str], rules: tuple[int, str]):
    def fake(path: str) -> tuple[int, str]:
        return protection if path.endswith("/protection") else rules

    return fake


NOT_PROTECTED = (1, '{"message":"Branch not protected","status":"404"}')
FORK_RULES = (0, json.dumps([{"type": "deletion"}, {"type": "non_fast_forward"}]))


def test_live_fails_when_only_deletion_and_force_push_are_blocked() -> None:
    errors = live.verify_branch_protection(gh_api=_api(NOT_PROTECTED, FORK_RULES))
    assert any("Missing required status check context" in e for e in errors), errors


def test_live_passes_when_a_ruleset_requires_both_checks() -> None:
    rules = (0, json.dumps(_ruleset_rule(FAST, GATE)))
    assert live.verify_branch_protection(gh_api=_api(NOT_PROTECTED, rules)) == []


def test_live_passes_when_classic_protection_requires_both_checks() -> None:
    protection = (0, json.dumps({"required_status_checks": {"contexts": [FAST, GATE]}}))
    assert live.verify_branch_protection(gh_api=_api(protection, (1, "boom"))) == []


def test_live_fails_when_one_check_is_missing() -> None:
    rules = (0, json.dumps(_ruleset_rule(FAST)))
    errors = live.verify_branch_protection(gh_api=_api(NOT_PROTECTED, rules))
    assert any(GATE in e for e in errors), errors


def test_live_fails_closed_when_nothing_is_readable() -> None:
    errors = live.verify_branch_protection(gh_api=_api((1, "x"), (1, "y")))
    assert errors and "no live enforcement source readable" in errors[0]


def test_live_fails_on_nightly_only_gate() -> None:
    rules = (0, json.dumps(_ruleset_rule(FAST, GATE, "ent-audit-nightly")))
    errors = live.verify_branch_protection(gh_api=_api(NOT_PROTECTED, rules))
    assert any("Nightly-only" in e for e in errors), errors


# ---- compensating verifier ----------------------------------------------------------------------------------------


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(root: Path, *, sub: dict[str, Any] | None) -> Path:
    qual = "docs/project/maintainers/qualification/ent_audit_x"
    _write(root, f"{qual}/OD_ENT_02_THREAT_MODEL.md", "bypass residual risk OPEN-BP-01")
    _write(
        root,
        f"{qual}/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md",
        "Option B OPEN-BP-01 no equivalence fail-closed igx_gov ticket",
    )
    _write(root, ".github/workflows/enterprise-audit.yml", "ent-audit-fast ent-audit-pr-gate")
    _write(root, "tests/unit/governance/test_iiac_negative_guards.py", "N-01 N-05 N-08 N-12")
    _write(root, "evidence/rulesets.json", "{}")
    _write(root, "evidence/audit.md", "# audit")
    _write(root, "evidence/manifest.json", "{}")
    if sub is not None:
        _write(root, comp.SUB_MANIFEST_REL, json.dumps(sub))
    return root


def _sub(**override: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": comp.SUB_MANIFEST_SCHEMA,
        "mode": "compensating",
        "no_equivalence_claim": True,
        "open_debt": [comp.OPEN_DEBT_ID],
        "controls": [
            {
                "id": "C1",
                "description": "ruleset blocks deletion",
                "kind": "ruleset",
                "evidence_path": "evidence/rulesets.json",
            }
        ],
        "auditor_acceptance": {
            "verdict": "PASS_SCOPED",
            "record_path": "evidence/audit.md",
            "manifest_path": "evidence/manifest.json",
        },
    }
    data.update(override)
    return data


GOOD_RULES = [{"type": "deletion"}, {"type": "non_fast_forward"}]


def test_draft_mode_still_passes_on_complete_package(tmp_path: Path) -> None:
    assert comp.collect_errors(_repo(tmp_path, sub=None)) == []


def test_draft_text_alone_never_satisfies_published_mode(tmp_path: Path) -> None:
    errors = comp.collect_errors(_repo(tmp_path, sub=None), published=True)
    assert any("missing" in e and "od_ent_02_compensating.v1.json" in e for e in errors), errors


def test_published_mode_passes_with_real_evidence(tmp_path: Path) -> None:
    assert comp.collect_errors(_repo(tmp_path, sub=_sub()), published=True, branch_rules=GOOD_RULES) == []


PUBLISHED_REJECTED = {
    "equivalence_claim_true_is_not_required_false": (_sub(no_equivalence_claim=False), "no_equivalence_claim"),
    "wrong_mode": (_sub(mode="live"), "mode must be"),
    "open_debt_missing": (_sub(open_debt=[]), "OPEN-BP-01"),
    "no_controls": (_sub(controls=[]), "controls must be a non-empty list"),
    "control_without_evidence": (
        _sub(controls=[{"id": "C1", "description": "d", "kind": "ci", "evidence_path": "evidence/nope.json"}]),
        "evidence_path",
    ),
    "control_path_escape": (
        _sub(controls=[{"id": "C1", "description": "d", "kind": "ci", "evidence_path": "../outside.json"}]),
        "evidence_path",
    ),
    "control_claims_equivalence": (
        _sub(
            controls=[
                {
                    "id": "C1",
                    "description": "equivalent to live protection",
                    "kind": "ci",
                    "evidence_path": "evidence/audit.md",
                }
            ]
        ),
        "claims equivalence",
    ),
    "duplicate_control": (
        _sub(controls=[{"id": "C1", "description": "d", "evidence_path": "evidence/audit.md"}] * 2),
        "duplicate control id",
    ),
    "no_auditor_acceptance": (_sub(auditor_acceptance=None), "auditor_acceptance"),
    "auditor_verdict_fail": (
        _sub(
            auditor_acceptance={
                "verdict": "FAIL",
                "record_path": "evidence/audit.md",
                "manifest_path": "evidence/manifest.json",
            }
        ),
        "verdict must be PASS",
    ),
    "auditor_record_missing": (
        _sub(
            auditor_acceptance={
                "verdict": "PASS",
                "record_path": "evidence/none.md",
                "manifest_path": "evidence/manifest.json",
            }
        ),
        "record_path",
    ),
}


@pytest.mark.parametrize(("sub", "match"), PUBLISHED_REJECTED.values(), ids=PUBLISHED_REJECTED.keys())
def test_published_mode_rejects(tmp_path: Path, sub: dict[str, Any], match: str) -> None:
    errors = comp.collect_errors(_repo(tmp_path, sub=sub), published=True, branch_rules=GOOD_RULES)
    assert any(match in e for e in errors), errors


def test_ruleset_control_requires_rules_to_be_read_and_present(tmp_path: Path) -> None:
    root = _repo(tmp_path, sub=_sub())
    assert any("not read" in e for e in comp.collect_errors(root, published=True, branch_rules=None))
    errors = comp.collect_errors(root, published=True, branch_rules=[{"type": "deletion"}])
    assert any("non_fast_forward" in e for e in errors), errors


def test_malformed_sub_manifest_is_rejected_in_draft_mode_too(tmp_path: Path) -> None:
    root = _repo(tmp_path, sub=None)
    _write(root, comp.SUB_MANIFEST_REL, "{not json")
    assert any("invalid" in e for e in comp.collect_errors(root))
