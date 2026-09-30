# © Artur Czarnecki. All rights reserved.

"""AW-7C-P4 static architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_P4_AW = Path(__file__).resolve().parents[3] / "intergrax/autonomous_work/scoped_adaptive_integration_execution.py"
_P4_CONTRACTS = (
    Path(__file__).resolve().parents[3]
    / "intergrax/contracts/autonomous_work/scoped_adaptive_integration_execution.py"
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_p4_coordinator_forbidden_nexus_toolruntime_execution_runtime() -> None:
    joined = "\n".join(_imports(_P4_AW)).lower()
    assert "intergrax.runtime.nexus" not in joined
    assert "toolruntime" not in joined
    assert "intergrax.runtime.execution.runtime" not in joined


def test_p4_coordinator_no_parent_execution_authority_mint() -> None:
    text = _P4_AW.read_text(encoding="utf-8")
    assert "ParentExecutionAuthority(" not in text


def test_p4_contracts_no_resolved_credential() -> None:
    text = _P4_CONTRACTS.read_text(encoding="utf-8")
    assert "ResolvedCredential" not in text


def test_p4_coordinator_no_resolved_credential() -> None:
    text = _P4_AW.read_text(encoding="utf-8")
    assert "ResolvedCredential" not in text


def test_request_echo_target_resolver_removed() -> None:
    resolver_path = (
        Path(__file__).resolve().parents[3]
        / "intergrax/integrations/scoped_integration_adaptation_target_resolver.py"
    )
    text = resolver_path.read_text(encoding="utf-8")
    assert "IntegrationIdentityScopedIntegrationAdaptationTargetResolver" not in text
    assert "SourceBackedScopedIntegrationAdaptationTargetResolver" in text
