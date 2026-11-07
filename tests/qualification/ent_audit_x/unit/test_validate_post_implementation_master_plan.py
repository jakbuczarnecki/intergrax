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


# --- T-IGX-022-A (CT-ENT-01): accepted disposition set / anti-divergence ---------------------

_DISPOSITION_ERROR_PREFIX = "disposition must declare one accepted constant"


def _disposition_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("**Disposition:**")]
    assert len(lines) == 1, f"expected exactly one **Disposition:** line, found {len(lines)}"
    return lines[0]


def _with_disposition(text: str, disposition_line: str) -> str:
    return text.replace(_disposition_line(text), disposition_line, 1)


def test_plan_disposition_line_declares_an_accepted_constant(baseline_plan_text: str) -> None:
    """Anti-divergence guard: plan text and validator constants must not drift apart again (NF-01)."""
    lib = _load_lib()
    line = _disposition_line(baseline_plan_text)
    found = [c for c in lib.ACCEPTED_DISPOSITIONS if c in line]
    assert len(found) == 1, (
        "plan **Disposition:** line and lib.ACCEPTED_DISPOSITIONS diverged: "
        f"plan line={line!r}; accepted={list(lib.ACCEPTED_DISPOSITIONS)!r}; matched={found!r}"
    )


def test_internally_closed_constant_matches_published_plan_line(baseline_plan_text: str) -> None:
    lib = _load_lib()
    assert lib.INTERNALLY_CLOSED_DISPOSITION in lib.ACCEPTED_DISPOSITIONS
    assert lib.INTERNALLY_CLOSED_DISPOSITION in _disposition_line(baseline_plan_text)


def test_validator_fails_when_no_disposition_constant(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = _with_disposition(baseline_plan_text, "**Disposition:** ENT-AUDIT-X — SOMETHING ELSE")
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any(e.startswith(_DISPOSITION_ERROR_PREFIX) for e in errors), errors
    assert not any("INTERNALLY CLOSED disposition incompatible" in e for e in errors), errors


@pytest.mark.parametrize("index", range(4))
def test_validator_accepts_each_accepted_disposition(baseline_plan_text: str, index: int) -> None:
    lib = _load_lib()
    constants = lib.ACCEPTED_DISPOSITIONS
    assert len(constants) == 4
    mutated = _with_disposition(baseline_plan_text, f"**Disposition:** `{constants[index]}`")
    errors = lib.validate_master_plan(mutated, _repo_root())
    # Other rules (e.g. READY-specific ones) may legitimately fire; only the disposition check is asserted.
    assert not any(e.startswith(_DISPOSITION_ERROR_PREFIX) for e in errors), errors


def test_internally_closed_with_other_errors_is_rejected(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text.replace(
        "| C072 | C012,C024 | W-B |",
        "| C072 | C012,C024 | W-A |",
    )
    assert lib.INTERNALLY_CLOSED_DISPOSITION in mutated
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("C072" in e for e in errors), errors
    assert "INTERNALLY CLOSED disposition incompatible with validation errors" in errors


def test_internally_closed_does_not_whitelist_self_cert_phrases(baseline_plan_text: str) -> None:
    lib = _load_lib()
    mutated = baseline_plan_text + "\nB6 PASS\n"
    assert lib.INTERNALLY_CLOSED_DISPOSITION in mutated
    errors = lib.validate_master_plan(mutated, _repo_root())
    assert any("B6 PASS" in e for e in errors), errors
