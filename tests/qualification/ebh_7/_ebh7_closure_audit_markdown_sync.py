# © Artur Czarnecki. All rights reserved.

"""Render EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md from closure audit SSOT."""

from __future__ import annotations

from pathlib import Path

from tests.qualification.ebh_7._ebh7_closure_evidence_audit import (
    all_closure_audit_rows,
    closure_open_findings,
)
from tests.qualification.ebh_7._ebh7_p03_evidence_integrity import _load_manifest

_REPO = Path(__file__).resolve().parents[3]
_MD_PATH = (
    _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md"
)


def render_closure_evidence_audit_markdown() -> str:
    manifest = _load_manifest()
    test_sha = manifest["TEST_EXECUTION_SHA"]
    open_count = len(closure_open_findings())
    lines = [
        "# EBH-7 — Closure evidence audit matrix (G01–G09)",
        "",
        f"**TEST_EXECUTION_SHA:** `{test_sha}`",
        "",
        f"**OPEN_FINDING count:** {open_count} (must be 0 before independent B6)",
        "",
        "**Authority:** `tests/qualification/ebh_7/_ebh7_closure_evidence_audit.py`",
        "",
        "| G | Subject | Status | Evidence | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in all_closure_audit_rows():
        ev = row.evidence.replace("|", "\\|")
        if len(ev) > 100:
            ev = ev[:97] + "..."
        notes = row.notes.replace("|", "\\|")
        if len(notes) > 80:
            notes = notes[:77] + "..."
        lines.append(f"| {row.gate_id} | {row.subject} | {row.status.value} | {ev} | {notes} |")
    lines.extend(
        [
            "",
            "## G10–G12 (outside agent closure)",
            "",
            "| ID | Status |",
            "| --- | --- |",
            "| G10 Protocol 03 B6 | OPEN — external auditor PASS/FAIL record |",
            "| G11 OD-EBH7-08 | OPEN — operator signature |",
            "| G12 Roadmap EBH-7 | IN PROGRESS until OD-08 |",
            "",
            "**Non-claims:** B6 PASS, OD-EBH7-08 APPROVED, parent CLOSED, global FRZ PASS.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_closure_audit_markdown_matches_ssot() -> None:
    expected = render_closure_evidence_audit_markdown()
    actual = _MD_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md out of sync — run render from SSOT or pytest gate"
    )


def write_closure_evidence_audit_markdown() -> None:
    _MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MD_PATH.write_text(render_closure_evidence_audit_markdown(), encoding="utf-8")


if __name__ == "__main__":
    write_closure_evidence_audit_markdown()
    print(f"Wrote {_MD_PATH.relative_to(_REPO)}")
