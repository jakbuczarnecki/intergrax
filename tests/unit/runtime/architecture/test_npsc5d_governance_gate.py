# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R1 — multi-agent governance architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_NPSC5D_PRODUCTION = (
    _REPO_ROOT / "intergrax" / "contracts" / "multi_agent_coordination_governance.py",
    _REPO_ROOT / "intergrax" / "runtime" / "governance" / "multi_agent_coordination_governance.py",
    _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_governance_adapter.py",
    _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py",
)

_FORBIDDEN_ENGINE_NAMES = (
    "MultiAgentPolicyEngine",
    "CoordinationGovernanceEngine",
    "AgentGovernanceRuntime",
    "NpscPolicyEngine",
    "DelegationPolicyEngine",
    "NpscGovernanceRuntime",
)

_FORBIDDEN_GOVERNANCE_IMPORTS = (
    "intergrax.agent_distribution.multi_agent_coordination",
    "intergrax.agent_distribution.bounded_multi_agent_fanout",
    "intergrax.agent_distribution.coordination_intent_executor",
)

_FORBIDDEN_PATTERNS = (
    re.compile(r"\bAny\b"),
    re.compile(r"dict\[str,\s*Any\]"),
    re.compile(r"Dict\[str,\s*Any\]"),
    re.compile(r"\bgetattr\("),
    re.compile(r"\bsetattr\("),
    re.compile(r"\bhasattr\("),
    re.compile(r"except\s+Exception\b"),
)


def _repo_python_files() -> list[Path]:
    return list((_REPO_ROOT / "intergrax").rglob("*.py"))


@pytest.mark.gate
def test_npsc5d_modules_exist() -> None:
    missing = [path for path in _NPSC5D_PRODUCTION if not path.is_file()]
    assert missing == [], f"missing NPSC-5D/R1 modules: {missing}"


@pytest.mark.gate
def test_no_second_governance_engine_concepts() -> None:
    violations: list[str] = []
    for path in _repo_python_files():
        if "build" in path.parts or ".tmp" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for name in _FORBIDDEN_ENGINE_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == [], (
        "Forbidden second governance engine concepts found:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_governance_core_has_no_reverse_agent_distribution_dependency() -> None:
    governance_root = _REPO_ROOT / "intergrax" / "runtime" / "governance"
    violations: list[str] = []
    for path in governance_root.rglob("*.py"):
        if path.name == "multi_agent_coordination_governance.py":
            source = path.read_text(encoding="utf-8-sig")
            for forbidden in _FORBIDDEN_GOVERNANCE_IMPORTS:
                if forbidden in source:
                    violations.append(f"{path.name}: {forbidden}")
    assert violations == [], (
        "Governance core must not import NPSC implementation internals:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_coordination_intent_executor_evaluates_governance_before_routing() -> None:
    executor_path = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    )
    source = executor_path.read_text(encoding="utf-8-sig")
    governance_index = source.index("_enforce_governance")
    coordinate_index = source.index("await self._coordination.coordinate")
    fan_out_index = source.index("await self._fan_out.fan_out")
    assert governance_index < coordinate_index
    assert governance_index < fan_out_index


@pytest.mark.gate
def test_governance_contract_has_no_physical_agent_fields() -> None:
    contract_path = (
        _REPO_ROOT / "intergrax" / "contracts" / "multi_agent_coordination_governance.py"
    )
    source = contract_path.read_text(encoding="utf-8-sig")
    for field in ("agent_id", "agent_instance_id", "lease_id", "OrchestrationSlot"):
        assert f"{field}:" not in source


@pytest.mark.gate
def test_npsc5d_contracts_avoid_prohibited_patterns() -> None:
    violations: list[str] = []
    for path in _NPSC5D_PRODUCTION:
        source = path.read_text(encoding="utf-8-sig")
        for pattern in _FORBIDDEN_PATTERNS:
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{path.relative_to(_REPO_ROOT).as_posix()}:{line}:{match.group()}",
                )
    assert violations == [], (
        "NPSC-5D/R1 authoritative modules contain prohibited patterns:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_coordination_governance_adapter_imports_public_contract_only() -> None:
    adapter_path = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_governance_adapter.py"
    )
    tree = ast.parse(adapter_path.read_text(encoding="utf-8-sig"), filename=str(adapter_path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    forbidden = [
        module
        for module in sorted(modules)
        if module.startswith("intergrax.runtime.governance")
        or module.startswith("intergrax.runtime.policy.runtime_policy_engine")
    ]
    assert forbidden == [], (
        "Caller adapter must depend on public governance contracts, not runtime internals:\n"
        + "\n".join(forbidden)
    )
