#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-7.3 — reasoning failure taxonomy on all planner kinds."""

from __future__ import annotations

import sys

from intergrax.applications._shared.reasoning_failure_wiring import (
    _CLASSIFIER_FAILURE_KINDS,
    _PLANNER_FAILURE_KINDS,
    reasoning_failure_taxonomy_complete,
    resolve_reasoning_failure_taxonomy,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.reasoning_failure import ReasoningFailureKind


def main() -> int:
    env = ApplicationEnvironmentProfile.lab_defaults()
    if not reasoning_failure_taxonomy_complete(env):
        print("reasoning failure taxonomy incomplete for lab defaults", file=sys.stderr)
        return 1
    taxonomy = resolve_reasoning_failure_taxonomy(env)
    covered: set[str] = set()
    for kinds in _PLANNER_FAILURE_KINDS.values():
        covered.update(item.value for item in kinds)
    for kinds in _CLASSIFIER_FAILURE_KINDS.values():
        covered.update(item.value for item in kinds)
    required = {item.value for item in ReasoningFailureKind}
    if not required.issubset(covered):
        missing = sorted(required - covered)
        print(f"missing failure kinds in taxonomy map: {missing}", file=sys.stderr)
        return 1
    if not taxonomy.get("planner") or not taxonomy.get("classifier"):
        print("active environment taxonomy must expose planner and classifier kinds", file=sys.stderr)
        return 1
    print("OK: reasoning failure taxonomy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
