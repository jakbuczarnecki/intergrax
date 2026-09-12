# © Artur Czarnecki. All rights reserved.

"""HARDENING-6 — execution authority audit architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_AUTHORITY_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_AUTHORITY_HARDENING.md"
)

# Composition roots — sole runtime modules allowed to reach the external-operation / task spine.
_EXECUTION_COMPOSITION_ALLOWLIST = frozenset(
    {
        "intergrax/runtime/self_healing/orchestrator.py",
        "intergrax/runtime/self_healing/workflow/orchestrator.py",
        "intergrax/runtime/prevention/actions/orchestrator.py",
        "intergrax/runtime/self_healing/autonomy/default_execution_boundary.py",
    }
)

# Diagnostics may correlate evidence with execution identity / decision finalization only.
_DIAGNOSTICS_EXECUTION_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "intergrax/runtime/diagnostics/terminal_execution_diagnostic_bridge.py": frozenset(
        {"intergrax.runtime.execution.boundary"}
    ),
    "intergrax/runtime/diagnostics/decision_lifecycle_projection.py": frozenset(
        {"intergrax.runtime.execution.decision_finalization_persistence"}
    ),
}

_DECISION_LAYER_ROOTS: tuple[tuple[str, Path], ...] = (
    ("diagnostics", _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"),
    ("prediction", _REPO_ROOT / "intergrax" / "runtime" / "prediction"),
    ("preventive_contracts", _REPO_ROOT / "intergrax" / "contracts" / "preventive"),
    ("predictive_contracts", _REPO_ROOT / "intergrax" / "contracts" / "predictive"),
    ("self_healing_contracts", _REPO_ROOT / "intergrax" / "contracts" / "self_healing"),
    ("self_healing_runtime", _REPO_ROOT / "intergrax" / "runtime" / "self_healing"),
    ("prevention_runtime", _REPO_ROOT / "intergrax" / "runtime" / "prevention"),
)

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus.execution",
    "intergrax.runtime.execution.runtime",
    "intergrax.runtime.execution.orchestration",
    "intergrax.runtime.execution.host_task",
    "intergrax.runtime.task.unified_task_runner",
    "intergrax.runtime.task.task_run_bridge",
    "intergrax.runtime.external_operations.admission.execution_gate",
    "intergrax.runtime.external_operations.admission.provider_executor",
)

_AUTONOMY_CONTROL_REL_PATHS = (
    "intergrax/runtime/self_healing/autonomy/service.py",
    "intergrax/runtime/self_healing/autonomy/plugin_control_engine.py",
    "intergrax/runtime/self_healing/autonomy/decision_evaluation_service.py",
    "intergrax/runtime/self_healing/autonomy/default_execution_guard.py",
    "intergrax/runtime/self_healing/autonomy/guard_support.py",
    "intergrax/runtime/self_healing/autonomy/default_risk_evaluator.py",
    "intergrax/runtime/self_healing/autonomy/default_policy.py",
    "intergrax/runtime/self_healing/autonomy/plugin_decision_evaluator.py",
)

_REQUIRED_DOC_SECTIONS = (
    "## Authority model",
    "## Forbidden flows",
    "## Approved flows",
    "## Legacy and exceptions",
)

_STRATEGY_DECISION_REL = (
    "intergrax/runtime/self_healing/decision_engine.py",
    "intergrax/runtime/self_healing/strategy_recommendation/service.py",
    "intergrax/runtime/self_healing/knowledge_evolution/service.py",
    "intergrax/runtime/self_healing/adaptive/engine.py",
    "intergrax/runtime/self_healing/lifecycle/engine.py",
    "intergrax/runtime/prevention/preventive_intelligence_engine.py",
    "intergrax/runtime/prediction/prediction_engine.py",
    "intergrax/runtime/diagnostics/diagnostic_orchestrator.py",
)


def _rel_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def _execution_import_violation(rel: str, module: str) -> str | None:
    if rel in _EXECUTION_COMPOSITION_ALLOWLIST:
        return None
    allowed = _DIAGNOSTICS_EXECUTION_IMPORT_ALLOWLIST.get(rel)
    if allowed is not None:
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in allowed):
            return None
        if module.startswith("intergrax.runtime.execution"):
            return f"{rel}: disallowed execution import {module}"
        return None
    if module.startswith("intergrax.runtime.execution"):
        return f"{rel}: execution import {module}"
    return None


def _prefix_violation(rel: str, module: str) -> str | None:
    if rel in _EXECUTION_COMPOSITION_ALLOWLIST:
        return None
    for prefix in _FORBIDDEN_IMPORT_PREFIXES:
        if module == prefix or module.startswith(f"{prefix}."):
            return f"{rel}: {module}"
    return _execution_import_violation(rel, module)


def _iter_python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_hardening_6_authority_doc_present() -> None:
    assert _AUTHORITY_DOC.is_file(), "EXECUTION_AUTHORITY_HARDENING.md required"
    text = _AUTHORITY_DOC.read_text(encoding="utf-8")
    missing = [section for section in _REQUIRED_DOC_SECTIONS if section not in text]
    assert missing == [], f"missing doc sections: {missing}"


def test_hardening_6_decision_layers_avoid_direct_spine_imports() -> None:
    violations: list[str] = []
    for label, root in _DECISION_LAYER_ROOTS:
        for path in _iter_python_files(root):
            rel = _rel_posix(path)
            if rel.startswith("intergrax/runtime/self_healing/action_providers/"):
                continue
            if rel == "intergrax/runtime/prevention/actions/orchestrator.py":
                continue
            for module in _collect_import_modules(path):
                if module.startswith("intergrax.runtime") and module.startswith("intergrax.contracts"):
                    continue
                if module.startswith("intergrax.contracts") and label.endswith("_contracts"):
                    if module.startswith("intergrax.runtime"):
                        violations.append(f"{rel}: {module}")
                    continue
                hit = _prefix_violation(rel, module)
                if hit:
                    violations.append(f"[{label}] {hit}")
                if label.endswith("_contracts") and module.startswith("intergrax.runtime"):
                    violations.append(f"[{label}] {rel}: {module}")
    assert violations == [], "direct spine / execution coupling:\n" + "\n".join(violations)


def test_hardening_6_autonomy_control_stays_authorization_only() -> None:
    violations: list[str] = []
    forbidden_import_fragments = (
        "intergrax.runtime.external_operations",
        "intergrax.runtime.nexus.execution",
        "intergrax.runtime.execution.runtime",
    )
    forbidden_source_tokens = (
        "SelfHealingActionProvider",
        "attempt_execution",
        "UAEPExecutor",
        "external_gate.admit",
        "graph_executor",
    )
    for rel in _AUTONOMY_CONTROL_REL_PATHS:
        path = _REPO_ROOT / rel
        for module in _collect_import_modules(path):
            for fragment in forbidden_import_fragments:
                if fragment in module:
                    violations.append(f"{rel}: import {module}")
        lowered = path.read_text(encoding="utf-8")
        for token in forbidden_source_tokens:
            if token in lowered:
                violations.append(f"{rel}: contains {token}")
    assert violations == [], "\n".join(violations)


def test_hardening_6_intelligence_modules_do_not_import_composition_roots() -> None:
    forbidden = (
        "intergrax.runtime.self_healing.orchestrator",
        "intergrax.runtime.prevention.actions.orchestrator",
    )
    violations: list[str] = []
    for rel in _STRATEGY_DECISION_REL:
        path = _REPO_ROOT / rel
        for module in _collect_import_modules(path):
            for prefix in forbidden:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{rel}: {module}")
    assert violations == [], "\n".join(violations)
