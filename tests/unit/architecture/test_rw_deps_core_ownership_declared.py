# © Artur Czarnecki. All rights reserved.

"""NT-RW-DEP-01 — every core requirement has a declared owner in the DEP-4 allowlist.

RW-PLAN rows NT-ROW-003 (EQOA-A4-005), NT-ROW-059 (FRZ-DEBT-02) and NT-ROW-060
(FRZ-DEBT-03); ticket T-IGX-051-H. The checks run in process against the
repository ``pyproject.toml``; the script exit code is not used.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from scripts.maintenance.check_dependency_ownership import (
    DEFAULT_PROJECT,
    check_project,
    normalize_name,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_VIOLATION_PREFIX = "CORE_OWNERSHIP_VIOLATION:"
_SEED_NAME = "rw-dep-seed-undeclared"


def _core_requirement_names(path: Path) -> list[str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = data["project"]["dependencies"]
    return sorted({normalize_name(Requirement(str(item)).name) for item in raw})


@pytest.mark.parametrize("name", _core_requirement_names(DEFAULT_PROJECT))
def test_rw_dep_01a_core_requirement_has_declared_owner(name: str) -> None:
    findings = check_project(DEFAULT_PROJECT)
    assert f"{_VIOLATION_PREFIX} {name}" not in findings


def test_rw_dep_01b_project_reports_no_core_ownership_violation() -> None:
    violations = [finding for finding in check_project(DEFAULT_PROJECT) if finding.startswith(_VIOLATION_PREFIX)]
    assert violations == []


def test_rw_dep_01c_gate_reports_seeded_undeclared_core_requirement(tmp_path: Path) -> None:
    text = DEFAULT_PROJECT.read_text(encoding="utf-8")
    marker = "\ndependencies = [\n"
    assert text.count(marker) == 1
    seeded = text.replace(marker, f'{marker}  "{_SEED_NAME}>=1,<2",\n', 1)
    copy = tmp_path / "pyproject.toml"
    copy.write_text(seeded, encoding="utf-8")

    findings = check_project(copy)

    assert f"{_VIOLATION_PREFIX} {_SEED_NAME}" in findings
