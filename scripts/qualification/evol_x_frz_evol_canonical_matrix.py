#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Canonical FRZ-EVOL-01..10 requirement text (SSOT: freeze acceptance checklist).

Used by audit reconciliation to detect mislabeled FRZ rows in human/agent reports.
Does not promote FRZ checklist rows — maps behavioral evidence only.
"""

from __future__ import annotations

from typing import Final

# Literal requirement column from PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md (EVOL rows).
FRZ_EVOL_CANONICAL_REQUIREMENTS: Final[dict[str, str]] = {
    "FRZ-EVOL-01": "semantic change-impact assessment exists for in-scope platform surfaces",
    "FRZ-EVOL-02": "change-impact graph coverage meets ratified threshold (EVOL-M-02)",
    "FRZ-EVOL-03": "change isolation / blast radius explainable per scenario and application",
    "FRZ-EVOL-04": "contract/schema evolution at change time detects consumers (COMPAT policy applied)",
    "FRZ-EVOL-05": "scenario/workflow definition evolution rules explicit (in-flight instances)",
    "FRZ-EVOL-06": "regression test selection from impact graph with mandatory safety suites",
    "FRZ-EVOL-07": "deployment / rollback / roll-forward qualified or unsafe refusal explicit",
    "FRZ-EVOL-08": "change diagnostics answer what changed / what failed / which deploy introduced regression",
    "FRZ-EVOL-09": "extensibility without unauthorized platform-core modification (EVOL-M-10)",
    "FRZ-EVOL-10": "post-freeze requalification triggers and evidence freshness enforced",
}

FRZ_EVOL_BEHAVIORAL_TEST_PREFIX: Final[dict[str, str]] = {
    f"FRZ-EVOL-{i:02d}": f"tests/qualification/evol_x/frz/test_evol_x_frz_{i:02d}_" for i in range(1, 11)
}


def validate_frz_evol_id_set(ids: set[str]) -> list[str]:
    expected = set(FRZ_EVOL_CANONICAL_REQUIREMENTS)
    problems: list[str] = []
    if ids != expected:
        problems.append(f"FRZ id set mismatch: missing={sorted(expected - ids)} extra={sorted(ids - expected)}")
    return problems


def validate_requirement_labels(labels: dict[str, str]) -> list[str]:
    """``labels`` maps FRZ-EVOL-xx -> requirement summary used in an audit matrix row."""
    problems = validate_frz_evol_id_set(set(labels))
    for frz_id, canonical in FRZ_EVOL_CANONICAL_REQUIREMENTS.items():
        got = (labels.get(frz_id) or "").strip()
        if got != canonical:
            problems.append(f"{frz_id}: requirement text must match checklist literal ({canonical!r} != {got!r})")
    return problems
