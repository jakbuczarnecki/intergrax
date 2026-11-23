# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) POS-1..POS-4: the shared posture predicate ``wiring_fails_closed`` (DP-24)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.applications._shared import wiring_posture
from intergrax.applications._shared.wiring_posture import wiring_fails_closed
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _with_posture(
    base: ApplicationEnvironmentProfile,
    *,
    profile: ApplicationProfile,
    mode: ExecutionMode,
) -> ApplicationEnvironmentProfile:
    meta = base.meta.model_copy(update={"application_profile": profile, "execution_mode": mode})
    return base.model_copy(update={"meta": meta})


def test_wiring_fails_closed_true_for_product_profile() -> None:
    env = _with_posture(
        ApplicationEnvironmentProfile.product_defaults(),
        profile=ApplicationProfile.PRODUCT,
        mode=ExecutionMode.BALANCED,
    )
    assert env.application_profile is ApplicationProfile.PRODUCT
    assert env.execution_mode is ExecutionMode.BALANCED
    assert wiring_fails_closed(env) is True


def test_wiring_fails_closed_true_for_strict_mode_without_product_profile() -> None:
    env = _with_posture(
        ApplicationEnvironmentProfile.lab_defaults(),
        profile=ApplicationProfile.LAB,
        mode=ExecutionMode.STRICT,
    )
    assert env.application_profile is ApplicationProfile.LAB
    assert wiring_fails_closed(env) is True


def test_wiring_fails_closed_false_for_lab_balanced() -> None:
    env = _with_posture(
        ApplicationEnvironmentProfile.lab_defaults(),
        profile=ApplicationProfile.LAB,
        mode=ExecutionMode.BALANCED,
    )
    assert wiring_fails_closed(env) is False


def _annotation_names(node: ast.expr | None) -> set[str]:
    """Names used inside an annotation, also inside a string annotation."""
    if node is None:
        return set()
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            names |= _annotation_names(ast.parse(child.value, mode="eval").body)
    return names


def _typing_violations(source: str) -> list[str]:
    """Functions with an unannotated parameter or return, or with ``Any`` / ``object`` in an annotation."""
    violations: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg is not None:
            params.append(args.vararg)
        if args.kwarg is not None:
            params.append(args.kwarg)
        for param in params:
            if param.annotation is None:
                violations.append(f"{node.name}: parameter {param.arg} has no annotation")
            elif {"Any", "object"} & _annotation_names(param.annotation):
                violations.append(f"{node.name}: parameter {param.arg} is annotated Any or object")
        if node.returns is None:
            violations.append(f"{node.name}: no return annotation")
        elif {"Any", "object"} & _annotation_names(node.returns):
            violations.append(f"{node.name}: return annotated Any or object")
    return violations


_SHAPES: tuple[tuple[str, int], ...] = (
    ("def f(x: int) -> bool:\n    return True\n", 0),
    ("def f(x) -> bool:\n    return True\n", 1),
    ("def f(x: int):\n    return True\n", 1),
    ("from typing import Any\n\ndef f(x: Any) -> bool:\n    return True\n", 1),
    ("def f(x: int) -> object:\n    return True\n", 1),
    ("def f(x: int) -> 'Any | None':\n    return None\n", 1),
)


def test_wiring_posture_module_is_fully_typed() -> None:
    for source, expected in _SHAPES:
        assert len(_typing_violations(source)) == expected, source
    module_source = Path(wiring_posture.__file__).read_text(encoding="utf-8")
    assert "def wiring_fails_closed(" in module_source
    assert _typing_violations(module_source) == []
