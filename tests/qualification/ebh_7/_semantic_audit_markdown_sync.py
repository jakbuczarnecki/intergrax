# © Artur Czarnecki. All rights reserved.

"""Render EBH_7_SEMANTIC_PROOF_BINDING_AUDIT.md from disposition SSOT."""

from __future__ import annotations

from pathlib import Path

from tests.qualification.ebh_7._semantic_disposition_registry import (
    EBH7_SEMANTIC_DISPOSITION_SHA,
    all_verified_dispositions,
)
from tests.qualification.ebh_7._semantic_proof_binding_audit import semantic_exact_counts

_REPO = Path(__file__).resolve().parents[3]
_MD_PATH = _REPO / "docs/project/maintainers/qualification/EBH_7_SEMANTIC_PROOF_BINDING_AUDIT.md"


def render_semantic_proof_binding_audit_markdown() -> str:
    ten_exact, ten_open, xdm_exact, xdm_open = semantic_exact_counts()
    lines = [
        "# EBH-7 — Semantic proof-binding audit (SSOT mirror)",
        "",
        "**Authoritative module:** `tests/qualification/ebh_7/_semantic_disposition_registry.py`",
        "",
        f"**Disposition SHA:** `{EBH7_SEMANTIC_DISPOSITION_SHA}`",
        "",
        f"**Summary:** TEN-E2E {ten_exact} EXACT / {ten_open} OPEN; XDM {xdm_exact} EXACT / {xdm_open} OPEN.",
        "",
        "| SCENARIO_ID | SEMANTIC_MATCH | ACTUAL_TEST | ASSERTIONS_INSPECTED | TEST_EXECUTION_EVIDENCE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in all_verified_dispositions():
        ai = d.assertions_inspected.replace("|", "\\|")
        if len(ai) > 120:
            ai = ai[:117] + "..."
        lines.append(
            f"| {d.scenario_id} | {d.semantic_match.value} | {d.actual_test} | {ai} | {d.test_execution_evidence} |"
        )
    lines.append("")
    return "\n".join(lines)


def assert_semantic_audit_markdown_matches_ssot() -> None:
    expected = render_semantic_proof_binding_audit_markdown()
    actual = _MD_PATH.read_text(encoding="utf-8")
    assert actual == expected, "EBH_7_SEMANTIC_PROOF_BINDING_AUDIT.md out of sync — regenerate from SSOT"
