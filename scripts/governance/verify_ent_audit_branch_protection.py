#!/usr/bin/env python3
"""Fail-closed verifier for OD-ENT-02 LIVE enforcement (read-only gh API).

Live enforcement means the ENT checks are REQUIRED on the branch. They may be required by classic branch
protection (``/branches/<b>/protection``) or by an active ruleset (``/rules/branches/<b>``, which lists the
active rules that apply to the branch). A ruleset that only blocks deletion or force-push does not satisfy
this verifier: that is a compensating control, never live enforcement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from typing import Any

_REPO_SLUG = "jakbuczarnecki/intergrax"
_DEFAULT_BRANCH = "development"

# From .github/workflows/enterprise-audit.yml - workflow `name` + job id (not nightly-only jobs).
_REQUIRED_CONTEXTS = (
    "Enterprise audit (ENT-AUDIT-X) / ent-audit-fast",
    "Enterprise audit (ENT-AUDIT-X) / ent-audit-pr-gate",
)

_FORBIDDEN_NIGHTLY_ONLY = (
    "ent-audit-nightly",
    "nightly",
)

GhApi = Callable[[str], tuple[int, str]]


def _gh_api(path: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout if proc.returncode == 0 else proc.stderr or proc.stdout


def _collect_required_contexts(payload: dict[str, Any] | list[Any]) -> set[str]:
    """Required status-check contexts from a protection payload or from a list of active branch rules."""
    contexts: set[str] = set()
    if isinstance(payload, list):
        payload = {"rules": payload}
    for key in ("required_status_checks", "required_pull_request_checks"):
        block = payload.get(key)
        if isinstance(block, dict):
            for ctx in block.get("contexts") or block.get("checks") or []:
                if isinstance(ctx, str):
                    contexts.add(ctx)
                elif isinstance(ctx, dict) and isinstance(ctx.get("context"), str):
                    contexts.add(ctx["context"])
    rules = payload.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            params = rule.get("parameters") or {}
            for ctx in params.get("required_status_checks") or []:
                if isinstance(ctx, dict) and isinstance(ctx.get("context"), str):
                    contexts.add(ctx["context"])
                elif isinstance(ctx, str):
                    contexts.add(ctx)
    return contexts


def verify_branch_protection(
    *,
    branch: str = _DEFAULT_BRANCH,
    repo: str = _REPO_SLUG,
    required: tuple[str, ...] = _REQUIRED_CONTEXTS,
    gh_api: GhApi = _gh_api,
) -> list[str]:
    errors: list[str] = []
    contexts: set[str] = set()
    sources = 0
    problems: list[str] = []

    code, body = gh_api(f"repos/{repo}/branches/{branch}/protection")
    if code == 0:
        try:
            contexts |= _collect_required_contexts(json.loads(body))
            sources += 1
        except json.JSONDecodeError:
            problems.append("GitHub API returned a non-JSON protection payload")
    else:
        problems.append(f"branch protection unavailable for {branch} (gh exit {code}): {body.strip()[:200]}")

    code, body = gh_api(f"repos/{repo}/rules/branches/{branch}")
    if code == 0:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            problems.append("GitHub API returned a non-JSON branch-rules payload")
        else:
            if isinstance(parsed, list):
                contexts |= _collect_required_contexts(parsed)
                sources += 1
            else:
                problems.append("GitHub branch-rules payload is not a list")
    else:
        problems.append(f"branch rules unavailable for {branch} (gh exit {code}): {body.strip()[:200]}")

    if sources == 0:
        errors.append("no live enforcement source readable: " + "; ".join(problems))
        return errors

    for req in required:
        if req not in contexts:
            errors.append(f"Missing required status check context: {req!r}")

    for ctx in contexts:
        low = ctx.lower()
        if any(f in low for f in _FORBIDDEN_NIGHTLY_ONLY) and "pr-gate" not in low:
            errors.append(f"Nightly-only check must not be sole PR gate: {ctx!r}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default=_DEFAULT_BRANCH)
    parser.add_argument("--repo", default=_REPO_SLUG)
    args = parser.parse_args(argv)

    errors = verify_branch_protection(branch=args.branch, repo=args.repo)
    if errors:
        print("ENT-AUDIT-X branch protection verify: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("ENT-AUDIT-X branch protection verify: PASS")
    for ctx in _REQUIRED_CONTEXTS:
        print(f"  - {ctx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
