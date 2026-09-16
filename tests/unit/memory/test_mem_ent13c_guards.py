# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13C architecture guards for durable qualification harness."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_HARNESS = Path(__file__).with_name("durable_provider_qualification_harness.py")
_13C_TESTS = Path(__file__).with_name("test_mem_ent13c_durable_provider_qualification.py")

_FORBIDDEN_REFLECTION = frozenset({"getattr", "hasattr", "setattr"})
_FORBIDDEN_PRIVATE_ATTRS = frozenset(
    {
        "_connection",
        "_client",
        "_db",
        "_store",
    }
)


def _iter_attribute_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
    return names


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def test_13c_harness_has_no_reflection() -> None:
    calls = _call_names(_HARNESS)
    for forbidden in _FORBIDDEN_REFLECTION:
        assert forbidden not in calls, f"{_HARNESS.name} uses {forbidden}()"


def test_13c_tests_have_no_reflection() -> None:
    calls = _call_names(_13C_TESTS)
    for forbidden in _FORBIDDEN_REFLECTION:
        assert forbidden not in calls, f"{_13C_TESTS.name} uses {forbidden}()"


def test_13c_harness_has_no_private_provider_member_access() -> None:
    attrs = _iter_attribute_names(_HARNESS)
    for forbidden in _FORBIDDEN_PRIVATE_ATTRS:
        assert forbidden not in attrs, f"{_HARNESS.name} accesses .{forbidden}"


def test_13c_tests_have_no_private_provider_member_access() -> None:
    attrs = _iter_attribute_names(_13C_TESTS)
    for forbidden in _FORBIDDEN_PRIVATE_ATTRS:
        assert forbidden not in attrs, f"{_13C_TESTS.name} accesses .{forbidden}"
