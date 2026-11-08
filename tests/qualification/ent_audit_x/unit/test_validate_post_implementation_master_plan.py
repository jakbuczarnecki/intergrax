from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_lib():
    scripts = _repo_root() / "scripts/enterprise_audit"
    sys.path.insert(0, str(scripts))
    import master_plan_ssot_lib as lib  # noqa: PLC0415

    return lib


def _plan_path() -> Path:
    return (
        _repo_root()
        / "docs/project/maintainers/qualification/ent_audit_x"
        / "ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md"
    )


@pytest.fixture
def baseline_plan_text() -> str:
    return _plan_path().read_text(encoding="utf-8")


def test_validator_passes_on_canonical_ssot(baseline_plan_text: str) -> None:
    lib = _load_lib()
    errors = lib.validate_master_plan(baseline_plan_text, _repo_root())
    assert errors == [], errors


def test_validator_fails_backward_wave_edge(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text.replace(
        "| C072 | C012,C024 | W-B |",
        "| C072 | C012,C024 | W-A |",
    )
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("C072" in e for e in errors)


def test_validator_fails_missing_check_in_matrix_e(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = re.sub(
        r"^\| ENT-A-BND-01 \|.*\n",
        "",
        baseline_plan_text,
        count=1,
        flags=re.MULTILINE,
    )
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("Matrix E" in e for e in errors)


def test_validator_fails_generic_oracle(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text.replace(
        "executes manifest entrypoint via sandbox",
        "subprocess gate exit 0 @ TES",
        1,
    )
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("wrapper" in e.lower() or "Matrix E" in e for e in errors)


def test_validator_fails_duplicate_task_id_in_dag(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text.replace(
        "| C001 | — | W-A |",
        "| C001 | — | W-A |\n| C001 | — | W-A |",
        1,
    )
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert errors


def test_validator_fails_c075_w0_batch(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text.replace(
        "| C075 | C041,C048,C055 | W-G |",
        "| C075 | C041,C048,C055 | W-0 |",
    )
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("C075" in e for e in errors)


def _disposition_line(text: str) -> str:
    return next(ln for ln in text.splitlines() if ln.startswith("**Disposition:**"))


def test_validator_accepts_published_internal_closure_disposition(baseline_plan_text: str) -> None:
    lib = _load_lib()
    assert lib.INTERNALLY_CLOSED_DISPOSITION in _disposition_line(baseline_plan_text)
    errors = lib.validate_master_plan(baseline_plan_text, _repo_root())
    assert not any("disposition" in e.lower() for e in errors), errors


def test_validator_fails_unknown_disposition(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text.replace(lib.INTERNALLY_CLOSED_DISPOSITION, "ENT-AUDIT-X — FULLY CERTIFIED", 1)
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("disposition must declare" in e for e in errors), errors


def test_validator_fails_formal_iiac_closure_disposition(baseline_plan_text: str) -> None:
    """This plan only ever records internal closure; the formal IIAC headline must not satisfy it."""
    lib = _load_lib()
    formal = "ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED"
    mutated = baseline_plan_text.replace(lib.INTERNALLY_CLOSED_DISPOSITION, formal, 1)
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("disposition must declare" in e for e in errors), errors


def test_validator_fails_missing_disposition_line(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = re.sub(r"^\*\*Disposition:\*\*.*\n", "", baseline_plan_text, count=1, flags=re.MULTILINE)
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("disposition line" in e for e in errors), errors


def test_validator_disposition_constant_elsewhere_does_not_satisfy(baseline_plan_text: str) -> None:
    """A known constant in the body must not rescue a wrong Disposition line."""
    lib = _load_lib()
    bad = baseline_plan_text.replace(lib.INTERNALLY_CLOSED_DISPOSITION, "ENT-AUDIT-X — FULLY CERTIFIED", 1)
    bad += f"\n\nNote: {lib.READY_DISPOSITION}\n"
    errors = lib.validate_master_plan(bad, _repo_root())
    assert any("disposition must declare" in e for e in errors), errors


def test_validator_cli_passes() -> None:
    root = _repo_root()
    script = root / "scripts/enterprise_audit/validate_post_implementation_master_plan.py"
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(script), "--path", str(_plan_path())],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
