#!/usr/bin/env python3
"""Fail-closed verifier for OD-ENT-02 GitHub branch protection (read-only gh API).

Required status-check contexts are the bare job names of ``.github/workflows/enterprise-audit.yml``
(the jobs carry no ``name:`` override, so check-run name == job id; the PR UI renders ``Workflow / job``
but that is a display label, not the status-check context).

Evidence = union of classic branch protection and repository/organization rulesets
(``rules/branches/{branch}``, which returns a top-level JSON list of *active* rules only).
Any endpoint error or malformed payload fails closed. This script never writes to GitHub.
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

# Check-run / status-check context == job id in .github/workflows/enterprise-audit.yml (no job `name:`).
_REQUIRED_CONTEXTS = (
    "ent-audit-fast",
    "ent-audit-pr-gate",
)

# Legacy "Workflow / job" UI labels: never reported as a status-check context. Requiring them would
# block every PR forever, so they are rejected loudly instead of being treated as a plain miss.
_LEGACY_PREFIXED = (
    "Enterprise audit (ENT-AUDIT-X) / ent-audit-fast",
    "Enterprise audit (ENT-AUDIT-X) / ent-audit-pr-gate",
)

_FORBIDDEN_NIGHTLY_ONLY = (
    "ent-audit-nightly",
    "nightly",
)

_NOT_PROTECTED_MARKER = "branch not protected"

_GH_API_DEFAULT = object()
GhApi = Callable[[str], tuple[int, str]]


def _gh_api(path: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout if proc.returncode == 0 else proc.stderr or proc.stdout


def _requirement_satisfied(req: str, contexts: set[str]) -> bool:
    """Bare job name must appear as a required context (legacy UI labels do not satisfy)."""
    return req in contexts


def _add_context(contexts: set[str], ctx: Any) -> None:
    if isinstance(ctx, str):
        contexts.add(ctx)
    elif isinstance(ctx, dict) and isinstance(ctx.get("context"), str):
        contexts.add(ctx["context"])


def _collect_classic_contexts(payload: dict[str, Any]) -> set[str]:
    """Contexts from the classic ``branches/{b}/protection`` object."""
    contexts: set[str] = set()
    for key in ("required_status_checks", "required_pull_request_checks"):
        block = payload.get(key)
        if isinstance(block, dict):
            for ctx in block.get("contexts") or []:
                _add_context(contexts, ctx)
            for ctx in block.get("checks") or []:
                _add_context(contexts, ctx)
    return contexts


def _collect_ruleset_contexts(rules: list[Any]) -> set[str]:
    """Contexts from the ``rules/branches/{b}`` top-level list (``required_status_checks`` rules)."""
    contexts: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("type") != "required_status_checks":
            continue
        params = rule.get("parameters")
        if not isinstance(params, dict):
            continue
        for ctx in params.get("required_status_checks") or []:
            _add_context(contexts, ctx)
    return contexts


def verify_branch_protection(
    *,
    branch: str = _DEFAULT_BRANCH,
    repo: str = _REPO_SLUG,
    required: tuple[str, ...] = _REQUIRED_CONTEXTS,
    gh_api: GhApi | Any = _GH_API_DEFAULT,
    sources_out: dict[str, set[str]] | None = None,
) -> list[str]:
    api: GhApi = _gh_api if gh_api is _GH_API_DEFAULT else gh_api

    errors: list[str] = []
    classic: set[str] = set()
    ruleset: set[str] = set()
    classic_note = "classic: unavailable"
    ruleset_note = "ruleset: unavailable"
    classic_readable = False
    ruleset_readable = False

    # --- classic branch protection: 404 "Branch not protected" tolerated, everything else fails closed
    code, body = api(f"repos/{repo}/branches/{branch}/protection")
    if code == 0:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            errors.append("GitHub API returned non-JSON classic protection payload")
        else:
            if isinstance(payload, dict):
                classic = _collect_classic_contexts(payload)
                classic_note = f"classic: {len(classic)} contexts"
                classic_readable = True
            else:
                errors.append("GitHub API returned non-object classic protection payload")
    elif _NOT_PROTECTED_MARKER in body.lower():
        classic_note = "classic: none"
        classic_readable = True
    else:
        errors.append(
            f"GitHub classic branch protection unavailable for {branch} (gh exit {code}): {body.strip()[:500]}"
        )

    # --- rulesets: top-level JSON list of active rules; any error or non-list fails closed
    rcode, rbody = api(f"repos/{repo}/rules/branches/{branch}")
    if rcode == 0:
        try:
            rules = json.loads(rbody)
        except json.JSONDecodeError:
            errors.append("GitHub API returned non-JSON rulesets payload")
        else:
            if isinstance(rules, list):
                ruleset = _collect_ruleset_contexts(rules)
                types = sorted({str(r.get("type")) for r in rules if isinstance(r, dict)})
                ruleset_note = f"ruleset: {len(rules)} rules ({', '.join(types) or 'none'})"
                ruleset_readable = True
            else:
                errors.append("GitHub API returned non-list rulesets payload")
    else:
        errors.append(f"GitHub rulesets unavailable for {branch} (gh exit {rcode}): {rbody.strip()[:500]}")

    if not classic_readable and not ruleset_readable:
        return errors

    if sources_out is not None:
        sources_out["classic"] = set(classic)
        sources_out["ruleset"] = set(ruleset)

    contexts = classic | ruleset
    for req in required:
        if not _requirement_satisfied(req, contexts):
            errors.append(
                f"Missing required status check context: {req!r} (sources read: {classic_note}; {ruleset_note})"
            )

    for ctx in sorted(contexts):
        if ctx in _LEGACY_PREFIXED:
            bare = ctx.rsplit(" / ", 1)[-1]
            errors.append(f"context {ctx!r} never reports (check-run name is {bare!r}); it would block every PR")
        low = ctx.lower()
        if any(f in low for f in _FORBIDDEN_NIGHTLY_ONLY) and "pr-gate" not in low:
            errors.append(f"Nightly-only check must not be sole PR gate: {ctx!r}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default=_DEFAULT_BRANCH)
    parser.add_argument("--repo", default=_REPO_SLUG)
    args = parser.parse_args(argv)

    sources: dict[str, set[str]] = {}
    errors = verify_branch_protection(branch=args.branch, repo=args.repo, sources_out=sources)
    if errors:
        print("ENT-AUDIT-X branch protection verify: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("ENT-AUDIT-X branch protection verify: PASS")
    for ctx in _REQUIRED_CONTEXTS:
        supplied = [name for name in ("classic", "ruleset") if ctx in sources.get(name, set())]
        print(f"  - {ctx} (source: {', '.join(supplied)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
