# © Artur Czarnecki. All rights reserved.

"""P0 — execution certification parallelization inventory static gates."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from testing_support.pytest_temp_root import PYTEST_TEMP_NAMESPACE

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_P0_QUAL_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "EXECUTION_CERTIFICATION_ACCELERATION_P0.md"
)
_P0_ARCH_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md"
)
_P0_DOC_INVENTORY = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md"
)

_R3_FINAL_GATE = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py"
)
_R3_IMPLEMENTATION_GATE = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_npsc5e_r3_child_fanout_partial_recovery.py"
)
_R2_FINAL_GATE = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"
)
_ROOT_CONFTEST = _REPO_ROOT / "conftest.py"

_EXPECTED_R3_FINAL_LABELS: frozenset[str] = frozenset(
    {
        "R1 Final",
        "R2 Final",
        "R3 implementation gate",
        "P0A",
        "DG_001",
        "NPSC-5A",
        "NPSC-5B Final",
        "NPSC-5C",
        "NPSC-5D Final",
        "HITL R3",
        "Attempt lifecycle",
        "Child execution",
        "Terminal",
        "Cancellation",
        "Checkpoint store",
        "Long-running",
        "Fan-out",
    },
)


@dataclass(frozen=True, slots=True)
class MandatorySuiteEntry:
    label: str
    targets: tuple[str, ...]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_mandatory_suite_labels(qualification_gate: Path) -> tuple[MandatorySuiteEntry, ...]:
    module = ast.parse(_read_text(qualification_gate), filename=str(qualification_gate))
    for node in module.body:
        assign_value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_MANDATORY_SUITES":
                    assign_value = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_MANDATORY_SUITES":
                if node.value is None:
                    continue
                assign_value = node.value
        if assign_value is None:
            continue
        value = assign_value
        if not isinstance(value, ast.Tuple):
            raise AssertionError("_MANDATORY_SUITES must be a tuple")
        entries: list[MandatorySuiteEntry] = []
        for element in value.elts:
            if not isinstance(element, ast.Tuple) or len(element.elts) != 2:
                raise AssertionError("each mandatory suite entry must be (label, targets)")
            label_node, targets_node = element.elts
            if not isinstance(label_node, ast.Constant) or not isinstance(label_node.value, str):
                raise AssertionError("suite label must be a string constant")
            if not isinstance(targets_node, (ast.List, ast.Tuple)):
                raise AssertionError("suite targets must be a list or tuple")
            targets: list[str] = []
            for item in targets_node.elts:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    raise AssertionError("suite target paths must be string constants")
                targets.append(item.value)
            entries.append(
                MandatorySuiteEntry(label=label_node.value, targets=tuple(targets)),
            )
        return tuple(entries)
    raise AssertionError("_MANDATORY_SUITES not found")


def test_p0_required_documentation_artifacts_exist() -> None:
    assert _P0_QUAL_DOC.is_file()
    assert _P0_ARCH_DOC.is_file()
    assert _P0_DOC_INVENTORY.is_file()


def test_p0_qualification_doc_links_architecture_companion() -> None:
    body = _read_text(_P0_QUAL_DOC)
    assert "EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md" in body
    assert "MISSING REUSABLE ABSTRACTION" in body


def test_p0_architecture_doc_defines_deterministic_aggregation() -> None:
    body = _read_text(_P0_ARCH_DOC)
    assert "Deterministic aggregation" in body
    assert "COLLECT ALL" in body


def test_root_conftest_applies_invocation_basetemp() -> None:
    source = _read_text(_ROOT_CONFTEST)
    assert "apply_invocation_pytest_basetemp" in source
    assert PYTEST_TEMP_NAMESPACE == "build/pytest"


def test_r2_final_gate_exposes_subprocess_run_pytest_helper() -> None:
    source = _read_text(_R2_FINAL_GATE)
    assert "def _run_pytest" in source
    assert '"uv"' in source or "'uv'" in source
    assert "subprocess.run" in source


def test_r3_final_mandatory_suite_inventory_matches_p0_contract() -> None:
    entries = _parse_mandatory_suite_labels(_R3_FINAL_GATE)
    labels = {entry.label for entry in entries}
    assert labels == _EXPECTED_R3_FINAL_LABELS
    by_label = {entry.label: entry for entry in entries}
    r3_impl = by_label["R3 implementation gate"]
    assert len(r3_impl.targets) == 1
    assert r3_impl.targets[0].endswith("test_npsc5e_r3_child_fanout_partial_recovery.py")


def test_r3_implementation_gate_uses_fixed_cross_db_path() -> None:
    source = _read_text(_R3_IMPLEMENTATION_GATE)
    assert "npsc5e-r3" in source
    assert "cross.db" in source
    assert ".tmp" in source


def test_p0_inventory_documents_fixed_cross_db_serial_risk() -> None:
    body = _read_text(_P0_QUAL_DOC)
    assert "npsc5e-r3/cross.db" in body
    assert "SERIAL ONLY" in body


def test_r3_final_gate_imports_shared_run_pytest_helper() -> None:
    source = _read_text(_R3_FINAL_GATE)
    assert "_run_pytest" in source
    assert re.search(r"test_mandatory_frozen_suite_passes", source)


@pytest.mark.gate
def test_p0_doc_inventory_lists_uea_canonical_owner() -> None:
    body = _read_text(_P0_DOC_INVENTORY)
    assert "UNIFIED_EXECUTION_ARCHITECTURE.md" in body
    assert "CANONICAL" in body
