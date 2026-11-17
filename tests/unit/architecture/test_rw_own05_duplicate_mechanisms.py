# © Artur Czarnecki. All rights reserved.

"""RW-plan architecture guard NT-OWN-05-1: one mechanism per named responsibility.

Covered here: the host LLM resolver (decision DP-16, finding A4-016). Exactly one product module
may both define ``resolve_llm_adapter`` and construct adapters; the Tier-3 module of the same name
may only delegate. The other four named responsibilities (plugin discovery, context-window tables,
tool-input guard, evaluator contract) have no step-0 scan in the W-B plan and are not guarded
here. Pure guard: it tightens or loosens no existing gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER_FACTORIES = frozenset({"create_adapter", "create_adapter_with_failover"})
_CANONICAL = "intergrax/runtime/wiring/llm_resolver.py"
_DELEGATING_FACADE = "intergrax/applications/_shared/llm_resolver.py"


def _resolver_definitions() -> dict[str, bool]:
    """Map module path to whether the module also calls an adapter factory."""
    found: dict[str, bool] = {}
    for root in ("intergrax", "agents", "applications"):
        for path in (_REPO_ROOT / root).rglob("*.py"):
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if "/tests/" in relative:
                continue
            text = path.read_text(encoding="utf-8")
            if "def resolve_llm_adapter" not in text:
                continue
            tree = ast.parse(text)
            defines = any(
                isinstance(node, ast.FunctionDef) and node.name == "resolve_llm_adapter" for node in ast.walk(tree)
            )
            if not defines:
                continue
            constructs = any(
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id in _ADAPTER_FACTORIES)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr in _ADAPTER_FACTORIES)
                )
                for node in ast.walk(tree)
            )
            found[relative] = constructs
    return found


def test_nt_own_05_1_one_host_llm_resolver() -> None:
    definitions = _resolver_definitions()
    assert set(definitions) == {_CANONICAL, _DELEGATING_FACADE}
    assert [path for path, constructs in definitions.items() if constructs] == [_CANONICAL]
