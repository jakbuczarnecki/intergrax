# © Artur Czarnecki. All rights reserved.

"""Architecture gate — conformance suite stays provider-neutral."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFORMANCE_ROOT = Path(__file__).resolve().parent

_FORBIDDEN_TRANSPORT_IMPORT_PREFIXES = (
    "intergrax.queueing.providers.kafka",
    "intergrax.queueing.providers.rabbitmq",
    "intergrax.integrations.providers.message_bus.kafka",
)


def _collect_import_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if any(module.startswith(prefix) for prefix in _FORBIDDEN_TRANSPORT_IMPORT_PREFIXES):
                    violations.append(f"{rel}:{node.lineno}: import {module}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if any(module.startswith(prefix) for prefix in _FORBIDDEN_TRANSPORT_IMPORT_PREFIXES):
                violations.append(f"{rel}:{node.lineno}: from {module}")
    return violations


def test_conformance_suite_does_not_import_provider_transport_implementations() -> None:
    violations: list[str] = []
    for path in sorted(_CONFORMANCE_ROOT.glob("test_*.py")):
        violations.extend(_collect_import_violations(path))
    assert violations == [], (
        "P0C-8 conformance tests must remain platform-contract focused: "
        + ", ".join(violations)
    )
