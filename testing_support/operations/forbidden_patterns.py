# © Artur Czarnecki. All rights reserved.

"""Forbidden operator bypass patterns for runbook static validation (EE-B4-C)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForbiddenRunbookPattern:
    pattern_id: str
    regex: re.Pattern[str]
    rationale: str


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


FORBIDDEN_RUNBOOK_PATTERNS: tuple[ForbiddenRunbookPattern, ...] = (
    ForbiddenRunbookPattern(
        "governance_disable",
        _compile(r"\bdisable\s+governance\b"),
        "Manual governance bypass",
    ),
    ForbiddenRunbookPattern(
        "skip_policy",
        _compile(r"\bskip\s+policy\b"),
        "Manual governance bypass",
    ),
    ForbiddenRunbookPattern(
        "force_allow",
        _compile(r"\bforce\s+allow\b"),
        "Manual governance bypass",
    ),
    ForbiddenRunbookPattern(
        "allowing_runtime_policy",
        _compile(r"AllowingRuntimeExecutionPolicyAdmission"),
        "Production governance bypass class",
    ),
    ForbiddenRunbookPattern(
        "delete_checkpoint",
        _compile(r"\bdelete\s+checkpoint\b"),
        "Checkpoint mutation",
    ),
    ForbiddenRunbookPattern(
        "edit_checkpoint_manually",
        _compile(r"\bedit\s+checkpoint\s+manually\b|\bmanually\s+edit\s+checkpoint\b"),
        "Checkpoint mutation",
    ),
    ForbiddenRunbookPattern(
        "delete_evidence",
        _compile(r"\bdelete\s+evidence\b"),
        "Evidence destruction",
    ),
    ForbiddenRunbookPattern(
        "clear_persistence",
        _compile(r"\bclear\s+persistence\b"),
        "Persistence bypass",
    ),
    ForbiddenRunbookPattern(
        "mint_run_id",
        _compile(r"\bmanually\s+mint\s+run_id\b|\bmint\s+run_id\s+manually\b"),
        "Identity bypass",
    ),
    ForbiddenRunbookPattern(
        "mint_execution_id",
        _compile(
            r"\bmanually\s+mint\s+execution_id\b|\bmint\s+execution_id\s+manually\b"
        ),
        "Identity bypass",
    ),
    ForbiddenRunbookPattern(
        "invoke_provider_directly",
        _compile(r"\binvoke\s+provider\s+directly\b"),
        "Direct provider execution",
    ),
    ForbiddenRunbookPattern(
        "rerun_tool_directly",
        _compile(r"\brerun\s+tool\s+directly\b"),
        "Manual retry bypass",
    ),
    ForbiddenRunbookPattern(
        "increment_permit",
        _compile(r"\bmanually\s+increment\s+capacity\b|\bclear\s+internal\s+permit\b"),
        "Capacity state manipulation",
    ),
    ForbiddenRunbookPattern(
        "disable_mandatory_persistence",
        _compile(r"\bdisable\s+mandatory\s+persistence\b"),
        "Evidence fail-open",
    ),
    ForbiddenRunbookPattern(
        "switch_to_best_effort_mandatory",
        _compile(r"\bswitch\s+to\s+best-effort\b.*\bmandatory\b"),
        "Evidence fail-open",
    ),
)
