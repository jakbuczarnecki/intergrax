# © Artur Czarnecki. All rights reserved.

"""AW-7B — ephemeral capability execution architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_AW7B_MODULE_PATHS = (
    "ephemeral_capability_execution.py",
    "ephemeral_capability_execution_ports.py",
)

_FORBIDDEN_IMPORTS = (
    "intergrax.runtime",
    "intergrax.codecraft",
    "intergrax.tools.providers.codecraft",
    "sandbox",
    "CodeCraftOrchestrator",
    "ToolRegistry",
)

_FORBIDDEN_TOKENS = (
    "WorkerCodeCraft",
    "WorkerSandbox",
    "WorkerGeneratedToolRegistry",
    "WorkerCapabilityRuntime",
    "AdaptiveCapabilityRuntime",
    "UniversalCapabilityExecutor",
    "WorkerHITLEngine",
    "WorkerPolicyEngine",
    "dict[str, Any]",
)


def _aw7b_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [base / name for name in _AW7B_MODULE_PATHS]


def test_aw7b_modules_forbidden_imports() -> None:
    for path in _aw7b_paths():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        joined = "\n".join(imported)
        for token in _FORBIDDEN_IMPORTS:
            assert token.lower() not in joined.lower(), f"{path.name} imports {token}"


def test_aw7b_modules_forbidden_tokens() -> None:
    for path in _aw7b_paths():
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            assert token not in source, f"{path.name} contains forbidden token {token}"


def test_aw7b_contracts_no_codecraft_import() -> None:
    module = importlib.import_module(
        "intergrax.contracts.autonomous_work.ephemeral_capability_execution",
    )
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported).lower()
    assert "codecraft" not in joined
    assert "runtime" not in joined


def test_aw7b_public_import_smoke() -> None:
    from intergrax.autonomous_work.ephemeral_capability_execution import (
        WorkerEphemeralCapabilityExecutionService,
    )
    from intergrax.contracts.autonomous_work.ephemeral_capability_execution import (
        WorkerEphemeralCapabilityExecutionStatus,
    )

    assert WorkerEphemeralCapabilityExecutionService is not None
    assert WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED.value == "SUCCEEDED"
