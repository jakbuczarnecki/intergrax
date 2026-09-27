# © Artur Czarnecki. All rights reserved.

"""HARNESS-W6-R1 — Runtime Intelligence typed contract architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_W6_CONTRACT_DIR = _REPO_ROOT / "intergrax" / "contracts" / "runtime_intelligence"
_W6_RUNTIME_DIR = _REPO_ROOT / "intergrax" / "runtime" / "runtime_intelligence"
_W6_ADVISORY = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "runtime_intelligence_advisory.py"
)

_FORBIDDEN_GOD_CLASSES = frozenset(
    {
        "RuntimeIntelligenceManager",
        "AIManager",
        "DecisionManager",
        "UniversalAnalyzer",
    },
)
_PROBE_NAMES = frozenset({"getattr", "setattr", "hasattr", "isinstance"})
_CANONICAL_ID_NAMES = frozenset({"TaskId", "RunId", "AttemptId", "ExecutionId"})


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _annotation_name(node: ast.expr | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_name(node.left)
        right = _annotation_name(node.right)
        if right == "None":
            return left
    return ast.unparse(node)


def _dataclass_field_types(path: Path, class_name: str) -> dict[str, str]:
    tree = _read_ast(path)
    fields: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fields[item.target.id] = _annotation_name(item.annotation)
    return fields


def _contracts_import_runtime(rel_dir: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(rel_dir.glob("*.py")):
        tree = _read_ast(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("intergrax.runtime"):
                    violations.append(f"{rel}: {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("intergrax.runtime"):
                        violations.append(f"{rel}: {alias.name}")
    return violations


def _forbidden_classes_in_dir(directory: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        tree = _read_ast(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_GOD_CLASSES:
                found.append(f"{rel}:{node.name}")
    return found


def _global_registry_hints(directory: Path) -> list[str]:
    hints: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if "analyzer_registry" in source.lower() or "global_analyzer" in source.lower():
            hints.append(rel)
    return hints


def _semantic_probes_in_contract_functions(
    rel: str, function_names: frozenset[str]
) -> list[str]:
    tree = _read_ast(_REPO_ROOT / rel)
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in function_names:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in _PROBE_NAMES:
                    violations.append(f"{rel}:{node.name} uses {sub.func.id}")
    return violations


def _advisory_imports_concrete_service() -> bool:
    tree = _read_ast(_W6_ADVISORY)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "RuntimeIntelligenceService" in {alias.name for alias in node.names}:
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("RuntimeIntelligenceService"):
                    return True
    source = (_REPO_ROOT / _W6_ADVISORY.relative_to(_REPO_ROOT)).read_text(
        encoding="utf-8"
    )
    return "RuntimeIntelligenceService" in source


def _problem_mint_imports(directory: Path) -> list[str]:
    offenders: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        tree = _read_ast(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "diagnostics" in node.module and any(
                    alias.name == "Problem" for alias in node.names
                ):
                    offenders.append(rel)
    advisory_tree = _read_ast(_W6_ADVISORY)
    for node in ast.walk(advisory_tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(alias.name == "Problem" for alias in node.names):
                offenders.append(_W6_ADVISORY.relative_to(_REPO_ROOT).as_posix())
    return offenders


def _analyzer_direct_execution_deps() -> list[str]:
    forbidden_modules = (
        "intergrax.runtime.execution.execution_runtime",
        "intergrax.runtime.cancellation",
        "intergrax.runtime.resilience",
    )
    offenders: list[str] = []
    for path in sorted(_W6_RUNTIME_DIR.rglob("*.py")):
        tree = _read_ast(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module.startswith(mod) for mod in forbidden_modules):
                    offenders.append(f"{rel}: {node.module}")
    return offenders


def test_g1_no_string_outcome_fields_in_contracts() -> None:
    analyzer_fields = _dataclass_field_types(
        _W6_CONTRACT_DIR / "analyzer.py",
        "RuntimeIntelligenceAnalyzerOutcome",
    )
    integration_fields = _dataclass_field_types(
        _W6_CONTRACT_DIR / "integration.py",
        "RuntimeIntelligenceIntegrationOutcome",
    )
    assert analyzer_fields.get("outcome") != "str"
    assert integration_fields.get("outcome") != "str"


def test_g2_outcome_fields_use_typed_enums() -> None:
    analyzer_fields = _dataclass_field_types(
        _W6_CONTRACT_DIR / "analyzer.py",
        "RuntimeIntelligenceAnalyzerOutcome",
    )
    integration_fields = _dataclass_field_types(
        _W6_CONTRACT_DIR / "integration.py",
        "RuntimeIntelligenceIntegrationOutcome",
    )
    assert analyzer_fields.get("outcome") == "RuntimeIntelligenceAnalyzerOutcomeCode"
    assert (
        integration_fields.get("outcome") == "RuntimeIntelligenceIntegrationOutcomeCode"
    )


def test_g3_context_and_facts_use_canonical_execution_ids() -> None:
    context_fields = _dataclass_field_types(
        _W6_CONTRACT_DIR / "context.py",
        "RuntimeIntelligenceContext",
    )
    facts_input_fields = _dataclass_field_types(
        _W6_CONTRACT_DIR / "integration.py",
        "RuntimeIntelligenceFactsInput",
    )
    runtime_facts_fields = _dataclass_field_types(
        _W6_RUNTIME_DIR / "runtime_facts.py",
        "RuntimeIntelligenceFacts",
    )
    for fields in (context_fields, facts_input_fields, runtime_facts_fields):
        assert fields.get("task_id") == "TaskId"
        assert fields.get("run_id") == "RunId"
        assert "AttemptId" in fields.get("attempt_id", "")
        assert "ExecutionId" in fields.get("execution_id", "")


def test_g4_contracts_runtime_intelligence_no_runtime_imports() -> None:
    assert _contracts_import_runtime(_W6_CONTRACT_DIR) == []


def test_g5_no_forbidden_god_components_in_w6_scope() -> None:
    offenders = _forbidden_classes_in_dir(_W6_CONTRACT_DIR)
    offenders.extend(_forbidden_classes_in_dir(_W6_RUNTIME_DIR))
    assert offenders == []


def test_g6_no_global_analyzer_registry_in_w6_runtime() -> None:
    assert _global_registry_hints(_W6_RUNTIME_DIR) == []


def test_g7_integration_boundaries_avoid_semantic_reflection() -> None:
    violations = _semantic_probes_in_contract_functions(
        "intergrax/contracts/runtime_intelligence/analyzer.py",
        frozenset({"run_runtime_intelligence_analyzer_isolated"}),
    )
    violations.extend(
        _semantic_probes_in_contract_functions(
            "intergrax/contracts/runtime_intelligence/integration.py",
            frozenset({"invoke_runtime_intelligence_integration_isolated"}),
        ),
    )
    assert violations == []


def test_g8_execution_advisory_depends_on_port_not_concrete_service() -> None:
    assert _advisory_imports_concrete_service() is False


def test_g9_no_problem_mint_imports_in_w6_production() -> None:
    offenders = _problem_mint_imports(_W6_RUNTIME_DIR)
    assert offenders == []


def test_g10_analyzer_layer_no_direct_execution_authority_imports() -> None:
    assert _analyzer_direct_execution_deps() == []
