#!/usr/bin/env python3
"""Verify the OD-ENT-02 Option B compensating-control package.

Default (draft) mode checks the package text, the threat model, the CI gate list and the negative tests.
``--published`` additionally requires the published sub-manifest ``od_ent_02_compensating.v1.json``: scoped
controls with existing evidence, an accepting independent-audit record, the open debt ``OPEN-BP-01`` and an
explicit "no equivalence" statement. If a control is declared as a ruleset control, ``--live-rulesets`` reads
the active branch rules from GitHub and requires ``deletion`` and ``non_fast_forward`` to be present.

No mode ever treats compensating controls as equivalent to live branch protection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.governance.iiac_common import (
    NEGATIVE_TEST_MODULE,
    OD_ENT_02_PACKAGE,
    OD_ENT_02_THREAT,
    REPO,
)

_REQUIRED_PACKAGE_MARKERS = (
    "Option B",
    "OPEN-BP-01",
    "no equivalence",
    "fail-closed",
    "igx_gov ticket",
)

SUB_MANIFEST_REL = "docs/project/maintainers/qualification/ent_audit_x/od_ent_02_compensating.v1.json"
SUB_MANIFEST_SCHEMA = "od_ent_02_compensating.v1"
OPEN_DEBT_ID = "OPEN-BP-01"
_ACCEPTING_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
_RULESET_REQUIRED_RULES = frozenset({"deletion", "non_fast_forward"})
_REPO_SLUG = "jakbuczarnecki/intergrax"
_DEFAULT_BRANCH = "development"


def _paths(repo: Path | None) -> tuple[Path, Path, Path, Path]:
    if repo is None:
        return REPO, OD_ENT_02_THREAT, OD_ENT_02_PACKAGE, NEGATIVE_TEST_MODULE
    return (
        repo,
        repo / OD_ENT_02_THREAT.relative_to(REPO),
        repo / OD_ENT_02_PACKAGE.relative_to(REPO),
        repo / NEGATIVE_TEST_MODULE.relative_to(REPO),
    )


def _existing_repo_file(root: Path, rel: object) -> bool:
    if not isinstance(rel, str) or not rel.strip():
        return False
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (root / path).is_file()


def _draft_errors(root: Path, threat: Path, package: Path, negative_tests: Path) -> list[str]:
    errors: list[str] = []

    if not threat.is_file():
        errors.append(f"missing threat model: {threat.name} (N-05)")
    else:
        text = threat.read_text(encoding="utf-8")
        for token in ("bypass", "residual risk", OPEN_DEBT_ID):
            if token.lower() not in text.lower():
                errors.append(f"threat model missing marker: {token}")

    if not package.is_file():
        errors.append(f"missing compensating package: {package.name}")
    else:
        pkg = package.read_text(encoding="utf-8")
        for marker in _REQUIRED_PACKAGE_MARKERS:
            if marker.lower() not in pkg.lower():
                errors.append(f"compensating package missing marker: {marker}")
        if "equivalent to live branch protection" in pkg.lower():
            errors.append("forbidden equivalence claim (N-06)")

    workflow = root / ".github/workflows/enterprise-audit.yml"
    if not workflow.is_file():
        errors.append("missing enterprise-audit workflow for CI gate list")
    else:
        wf = workflow.read_text(encoding="utf-8")
        if "ent-audit-fast" not in wf or "ent-audit-pr-gate" not in wf:
            errors.append("enterprise-audit.yml missing required ENT job ids")

    if not negative_tests.is_file():
        errors.append("missing negative test module test_iiac_negative_guards.py")
    else:
        text = negative_tests.read_text(encoding="utf-8")
        for nid in ("N-01", "N-05", "N-08", "N-12"):
            if nid not in text:
                errors.append(f"negative tests missing {nid}")

    return errors


def _published_errors(root: Path, branch_rules: list[dict[str, Any]] | None) -> list[str]:
    path = root / SUB_MANIFEST_REL
    if not path.is_file():
        return [f"published mode: missing {SUB_MANIFEST_REL}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return [f"od_ent_02_compensating.v1.json invalid: {exc}"]
    if not isinstance(data, dict):
        return ["od_ent_02_compensating.v1.json must be a JSON object"]

    errors: list[str] = []
    if data.get("schema_version") != SUB_MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {SUB_MANIFEST_SCHEMA}")
    if data.get("mode") != "compensating":
        errors.append("mode must be 'compensating'")
    if data.get("no_equivalence_claim") is not True:
        errors.append("no_equivalence_claim must be true (compensating is never equivalent to live protection)")
    debt = data.get("open_debt")
    if not isinstance(debt, list) or OPEN_DEBT_ID not in debt:
        errors.append(f"open_debt must list {OPEN_DEBT_ID}")

    controls = data.get("controls")
    has_ruleset_control = False
    if not isinstance(controls, list) or not controls:
        errors.append("controls must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, control in enumerate(controls):
            if not isinstance(control, dict):
                errors.append(f"controls[{index}] must be an object")
                continue
            cid = str(control.get("id", "")).strip()
            if not cid:
                errors.append(f"controls[{index}] has no id")
            elif cid in seen:
                errors.append(f"duplicate control id {cid}")
            seen.add(cid)
            if not str(control.get("description", "")).strip():
                errors.append(f"control {cid or index} has no description")
            if "equivalent to live" in str(control.get("description", "")).lower():
                errors.append(f"control {cid or index} claims equivalence with live protection")
            if not _existing_repo_file(root, control.get("evidence_path")):
                errors.append(f"control {cid or index}: evidence_path missing or not an existing repo file")
            has_ruleset_control = has_ruleset_control or control.get("kind") == "ruleset"

    acceptance = data.get("auditor_acceptance")
    if not isinstance(acceptance, dict):
        errors.append("auditor_acceptance missing (independent audit ACCEPT is required)")
    else:
        if str(acceptance.get("verdict", "")) not in _ACCEPTING_VERDICTS:
            errors.append("auditor_acceptance.verdict must be PASS or PASS_SCOPED")
        for key in ("record_path", "manifest_path"):
            if not _existing_repo_file(root, acceptance.get(key)):
                errors.append(f"auditor_acceptance.{key} missing or not an existing repo file")

    if has_ruleset_control:
        if branch_rules is None:
            errors.append("a ruleset control is declared but active branch rules were not read (use --live-rulesets)")
        else:
            present = {str(rule.get("type", "")) for rule in branch_rules if isinstance(rule, dict)}
            missing = sorted(_RULESET_REQUIRED_RULES - present)
            if missing:
                errors.append(f"active branch rules lack: {', '.join(missing)}")

    return errors


def collect_errors(
    repo: Path | None = None,
    *,
    published: bool = False,
    branch_rules: list[dict[str, Any]] | None = None,
) -> list[str]:
    root, threat, package, negative_tests = _paths(repo)
    errors = _draft_errors(root, threat, package, negative_tests)
    if published:
        errors.extend(_published_errors(root, branch_rules))
    else:
        # Draft mode still rejects a malformed sub-manifest if one exists (as before).
        sub_manifest = root / SUB_MANIFEST_REL
        if sub_manifest.is_file():
            try:
                json.loads(sub_manifest.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                errors.append(f"od_ent_02_compensating.v1.json invalid: {exc}")
    return errors


def _read_branch_rules(slug: str, branch: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    proc = subprocess.run(
        ["gh", "api", f"repos/{slug}/rules/branches/{branch}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None, f"could not read active branch rules (gh exit {proc.returncode}): {proc.stderr.strip()[:200]}"
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, "branch rules payload is not JSON"
    if not isinstance(parsed, list):
        return None, "branch rules payload is not a list"
    return [rule for rule in parsed if isinstance(rule, dict)], None


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", action="store_true", help="require the published sub-manifest and acceptance")
    parser.add_argument("--live-rulesets", action="store_true", help="read active branch rules from GitHub")
    parser.add_argument("--branch", default=_DEFAULT_BRANCH)
    parser.add_argument("--repo-slug", default=_REPO_SLUG)
    args = parser.parse_args(list(argv))

    branch_rules: list[dict[str, Any]] | None = None
    errors: list[str] = []
    if args.live_rulesets:
        branch_rules, problem = _read_branch_rules(args.repo_slug, args.branch)
        if problem:
            errors.append(problem)
    errors.extend(collect_errors(published=args.published, branch_rules=branch_rules))

    if errors:
        print("verify_od_ent_02_compensating_control: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.published:
        print(
            "verify_od_ent_02_compensating_control: PASS (published compensating scope; "
            f"NOT equivalent to live branch protection; {OPEN_DEBT_ID} remains open)"
        )
    else:
        print("verify_od_ent_02_compensating_control: PASS (package draft)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
