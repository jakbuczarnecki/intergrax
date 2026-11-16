"""Tests for scripts/governance/lint_iiac_closure_regex.py (T-IGX-050-J).

Fixtures are inline strings and temporary files. They are not certification evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.governance import iiac_activation_gate
from scripts.governance import lint_iiac_closure_regex as lint


def test_pattern_is_imported_from_the_gate_not_copied() -> None:
    assert lint._CLOSURE is iiac_activation_gate._CLOSURE


def test_negated_text_is_flagged() -> None:
    findings = lint.scan_text("ARCH-FREEZE not CLOSED", "fixture.md")
    assert [f.line for f in findings] == [1]
    assert findings[0].matched == "ARCH-FREEZE not CLOSED"


def test_match_is_case_insensitive_like_the_gate() -> None:
    assert len(lint.scan_text("ARCH-FREEZE is NOT closed yet", "fixture.md")) == 1


def test_negated_text_is_flagged_by_the_gate_itself() -> None:
    errors = iiac_activation_gate.closure_conflicts({}, "ARCH-FREEZE not CLOSED")
    assert errors == ["ACT-N18 activation text promotes ARCH-FREEZE to CLOSED"]


@pytest.mark.parametrize(
    "text",
    [
        "ARCH-FREEZE is still open; EVOL-X is only certified.",
        "ARCH-FREEZE remains open",
        "closed-world audit of the freeze",
    ],
)
def test_clean_text_is_not_flagged(text: str) -> None:
    assert lint.scan_text(text, "fixture.md") == []


def test_closure_wording_for_each_stage_is_flagged_with_line_number() -> None:
    text = "intro\nEVOL-X closure: CLOSED\nok line\nENT-AUDIT-X is CLOSED\n"
    findings = lint.scan_text(text, "fixture.md")
    assert [f.line for f in findings] == [2, 4]


def test_match_does_not_cross_a_line_break() -> None:
    assert lint.scan_text("ARCH-FREEZE\nCLOSED", "fixture.md") == []


def test_json_record_is_scanned_as_one_blob(tmp_path: Path) -> None:
    record = {"status": "NOT_IN_FORCE", "notes": "ARCH-FREEZE is not CLOSED"}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    findings = lint.scan_file(path)
    assert len(findings) == 1
    assert findings[0].line == 0


def test_main_exits_1_and_prints_the_line_on_a_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "note.md"
    path.write_text("fine\nARCH-FREEZE not CLOSED\n", encoding="utf-8")
    assert lint.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert f"{path}:2:" in out
    assert "1 match(es)" in out


def test_main_exits_0_on_clean_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "note.md"
    path.write_text("ARCH-FREEZE remains open\n", encoding="utf-8")
    assert lint.main([str(path)]) == 0
    assert "0 match(es)" in capsys.readouterr().out


def test_main_exits_2_on_unreadable_path(tmp_path: Path) -> None:
    assert lint.main([str(tmp_path / "missing.md")]) == 2


def test_default_targets_name_the_record_and_activation_decisions() -> None:
    names = [t.name for t in lint.default_targets()]
    assert "IIAC_POLICY_ACTIVATION_RECORD.json" in names
    assert any(n.startswith("IIAC_OD_ACTIVATION_") and n.endswith(".md") for n in names)
