#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Run maintenance gates only when changed paths match scope rules (TOOL-ROI-1.5 subset)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from git_changed_paths import paths_touch_prefix, push_paths, staged_paths  # noqa: E402


@dataclass(frozen=True)
class DiffScopedRule:
    prefixes: tuple[str, ...]
    script: str
    label: str
    script_args: tuple[str, ...] = ()


TIER_RULES: tuple[DiffScopedRule, ...] = (
    DiffScopedRule(
        ("agents/",),
        "scripts/maintenance/check_agents_no_tier3_imports.py",
        "agents tier-3 import boundary",
    ),
    DiffScopedRule(
        ("intergrax/", "agents/", "tests/", "scripts/", "testing_support/"),
        "scripts/maintenance/check_harness_no_getattr.py",
        "harness getattr/setattr AST gate (smoke scan)",
        ("--smoke",),
    ),
)

EXTRA_RULES: tuple[DiffScopedRule, ...] = (
    DiffScopedRule(
        ("applications/",),
        "scripts/ci/check_workspace_cleanup.py",
        "application workspace cleanup",
    ),
    DiffScopedRule(
        ("docs/project/architecture/", "docs/project/maintainers/plans/"),
        "scripts/docs/check_docs_domain_pairs.py",
        "docs domain pair integrity",
    ),
)

DEFAULT_RULES: tuple[DiffScopedRule, ...] = (
    *TIER_RULES,
    *EXTRA_RULES,
)


def _run_script(script_rel: str, script_args: tuple[str, ...]) -> int:
    script = _REPO / script_rel
    if not script.is_file():
        print(f"[ERROR] missing gate script: {script_rel}", file=sys.stderr)
        return 1
    print(f"[INFO] diff-scoped gate: {script_rel}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(script), *script_args],
        cwd=_REPO,
        check=False,
    )
    return completed.returncode


def run_rules(paths: frozenset[str], rules: Sequence[DiffScopedRule]) -> int:
    if not paths:
        print("[INFO] diff-scoped gates: no changed paths; skip.")
        return 0
    print(f"[INFO] diff-scoped gates: {len(paths)} changed path(s)", flush=True)
    exit_code = 0
    for rule in rules:
        if not paths_touch_prefix(paths, rule.prefixes):
            continue
        print(f"[INFO] scope hit ({rule.label})", flush=True)
        code = _run_script(rule.script, rule.script_args)
        if code != 0:
            exit_code = code
    if exit_code == 0:
        print("[OK] diff-scoped gates passed.", flush=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run path-scoped maintenance gates.")
    parser.add_argument(
        "--mode",
        choices=("push", "staged"),
        default="push",
        help="push=commits since upstream merge-base; staged=index for pre-commit.",
    )
    parser.add_argument(
        "--rules",
        choices=("all", "extras", "tier"),
        default="all",
        help="all=default; extras=apps/docs only (pair with push-fast tier steps); tier=tier scripts only.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = push_paths() if args.mode == "push" else staged_paths()
    if args.rules == "extras":
        rules = EXTRA_RULES
    elif args.rules == "tier":
        rules = TIER_RULES
    else:
        rules = DEFAULT_RULES
    return run_rules(paths, rules)


if __name__ == "__main__":
    raise SystemExit(main())
