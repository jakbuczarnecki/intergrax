# © Artur Czarnecki. All rights reserved.

"""Unit tests for scripts/governance/verify_ent_audit_branch_protection.py (T-IGX-022-B).

Offline only: ``_gh_api`` is monkeypatched and ``subprocess.run`` raises if ever reached.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.governance import verify_ent_audit_branch_protection as v

ROOT = Path(__file__).resolve().parents[3]
FAST = "ent-audit-fast"
GATE = "ent-audit-pr-gate"
LEGACY_FAST = "Enterprise audit (ENT-AUDIT-X) / ent-audit-fast"
LEGACY_GATE = "Enterprise audit (ENT-AUDIT-X) / ent-audit-pr-gate"

NOT_PROTECTED = (1, '{"message":"Branch not protected","status":"404"}\ngh: Branch not protected (HTTP 404)')

# Recorded verbatim from the live read-only GET rules/branches/development (ruleset 20844442).
_SRC = {"ruleset_source_type": "Repository", "ruleset_source": "jakbuczarnecki/intergrax", "ruleset_id": 20844442}
REAL_RULES_TODAY = [
    {"type": "deletion", **_SRC},
    {"type": "non_fast_forward", **_SRC},
]


def _classic(*contexts: str) -> tuple[int, str]:
    return 0, json.dumps({"required_status_checks": {"strict": True, "contexts": list(contexts)}})


def _classic_checks(*contexts: str) -> tuple[int, str]:
    return 0, json.dumps({"required_status_checks": {"checks": [{"context": c, "app_id": 15368} for c in contexts]}})


def _rules(*contexts: str) -> tuple[int, str]:
    rules: list[dict] = [{"type": "deletion", "ruleset_id": 1}]
    if contexts:
        rules.append(
            {
                "type": "required_status_checks",
                "ruleset_id": 1,
                "parameters": {"required_status_checks": [{"context": c} for c in contexts]},
            }
        )
    return 0, json.dumps(rules)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("subprocess.run reached in unit test (network/gh forbidden)")

    monkeypatch.setattr(subprocess, "run", _boom)


def _patch(monkeypatch: pytest.MonkeyPatch, classic: tuple[int, str], ruleset: tuple[int, str]) -> list[str]:
    calls: list[str] = []

    def fake(path: str) -> tuple[int, str]:
        calls.append(path)
        if "/protection" in path:
            return classic
        if "/rules/branches/" in path:
            return ruleset
        raise AssertionError(f"unexpected gh path {path}")

    monkeypatch.setattr(v, "_gh_api", fake)
    return calls


def test_required_contexts_are_bare_names() -> None:
    assert v._REQUIRED_CONTEXTS == (FAST, GATE)


# ---------------------------------------------------------------- positives
def test_pass_classic_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, _classic(FAST, GATE), (0, "[]"))
    assert v.verify_branch_protection() == []
    assert all(p.startswith("repos/") for p in calls)


def test_pass_classic_checks_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic_checks(FAST, GATE), (0, "[]"))
    assert v.verify_branch_protection() == []


def test_pass_ruleset_only_with_classic_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, NOT_PROTECTED, _rules(FAST, GATE))
    assert v.verify_branch_protection() == []


def test_pass_names_split_across_classic_and_ruleset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic(FAST), _rules(GATE))
    sources: dict[str, set[str]] = {}
    assert v.verify_branch_protection(sources_out=sources) == []
    assert sources["classic"] == {FAST} and sources["ruleset"] == {GATE}


# ---------------------------------------------------------------- negatives
def test_fail_todays_real_state_names_both_missing_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, NOT_PROTECTED, (0, json.dumps(REAL_RULES_TODAY)))
    errors = v.verify_branch_protection()
    joined = "\n".join(errors)
    assert len(errors) == 2
    assert repr(FAST) in joined and repr(GATE) in joined
    assert "classic: none" in joined
    assert "ruleset: 2 rules (deletion, non_fast_forward)" in joined


def test_fail_only_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, NOT_PROTECTED, _rules(FAST))
    errors = v.verify_branch_protection()
    assert len(errors) == 1 and repr(GATE) in errors[0]


def test_fail_legacy_prefixed_contexts_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic(LEGACY_FAST, LEGACY_GATE), (0, "[]"))
    errors = v.verify_branch_protection()
    joined = "\n".join(errors)
    assert joined.count("never reports") == 2
    assert "would block every PR" in joined
    assert f"Missing required status check context: {FAST!r}" in joined


def test_fail_legacy_prefixed_alongside_bare_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic(FAST, GATE, LEGACY_FAST), (0, "[]"))
    errors = v.verify_branch_protection()
    assert len(errors) == 1 and "never reports" in errors[0]


def test_fail_nightly_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, NOT_PROTECTED, _rules("ent-audit-nightly"))
    errors = v.verify_branch_protection()
    assert any("Nightly-only" in e for e in errors)
    assert any(repr(FAST) in e for e in errors) and any(repr(GATE) in e for e in errors)


@pytest.mark.parametrize("code,body", [(1, "gh: Forbidden (HTTP 403)"), (1, "gh: Server Error (HTTP 500)")])
def test_fail_classic_http_error_even_if_ruleset_correct(monkeypatch: pytest.MonkeyPatch, code: int, body: str) -> None:
    _patch(monkeypatch, (code, body), _rules(FAST, GATE))
    errors = v.verify_branch_protection()
    assert errors and "classic branch protection unavailable" in errors[0]


def test_fail_ruleset_endpoint_error_even_if_classic_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic(FAST, GATE), (1, "gh: Forbidden (HTTP 403)"))
    errors = v.verify_branch_protection()
    assert errors and "rulesets unavailable" in errors[0]


def test_fail_non_json_classic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, (0, "<html>"), _rules(FAST, GATE))
    assert any("non-JSON classic" in e for e in v.verify_branch_protection())


def test_fail_non_object_classic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, (0, "[]"), _rules(FAST, GATE))
    assert any("non-object classic" in e for e in v.verify_branch_protection())


def test_fail_non_json_ruleset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic(FAST, GATE), (0, "not json"))
    assert any("non-JSON rulesets" in e for e in v.verify_branch_protection())


def test_fail_non_list_ruleset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _classic(FAST, GATE), (0, json.dumps({"rules": []})))
    assert any("non-list rulesets" in e for e in v.verify_branch_protection())


def test_fail_empty_ruleset_list_and_no_classic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, NOT_PROTECTED, (0, "[]"))
    errors = v.verify_branch_protection()
    assert len(errors) == 2 and "ruleset: 0 rules" in errors[0]


def test_ruleset_rule_without_parameters_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    rules = json.dumps(
        [{"type": "required_status_checks"}, "junk", {"type": "required_status_checks", "parameters": []}]
    )
    _patch(monkeypatch, NOT_PROTECTED, (0, rules))
    assert len(v.verify_branch_protection()) == 2


# ---------------------------------------------------------------- drift guard
def test_workflow_job_ids_are_check_names_without_name_override() -> None:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "enterprise-audit.yml").read_text(encoding="utf-8"))
    jobs = wf["jobs"]
    for ctx in v._REQUIRED_CONTEXTS:
        assert ctx in jobs, f"workflow has no job id {ctx!r}; verifier and workflow diverged"
        assert "name" not in jobs[ctx], f"job {ctx!r} has a name: override; check-run name != job id"


# ---------------------------------------------------------------- main()
def test_main_pass_exit_0(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch(monkeypatch, NOT_PROTECTED, _rules(FAST, GATE))
    assert v.main([]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("ENT-AUDIT-X branch protection verify: PASS")
    assert f"{FAST} (source: ruleset)" in out.out and out.err == ""


def test_main_fail_exit_1_header_on_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch(monkeypatch, NOT_PROTECTED, (0, json.dumps(REAL_RULES_TODAY)))
    assert v.main(["--branch", "development"]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("ENT-AUDIT-X branch protection verify: FAIL")


def test_gh_api_wrapper_is_only_subprocess_user() -> None:
    # The autouse guard makes subprocess.run raise; the real wrapper must hit it (proves tests above never used it).
    with pytest.raises(AssertionError, match="network/gh forbidden"):
        v._gh_api("repos/x/y")
