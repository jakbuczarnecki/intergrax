# © Artur Czarnecki. All rights reserved.

"""GR-10-R5 — INFERENCE Inner Governance enterprise qualification (semantics only)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_INFERENCE_CAPABILITY_SEMANTICS,
    GR10_PRODUCTION_INVENTORY,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_matrix_inference_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INFERENCE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference.py"

_FORBIDDEN_INFERENCE_INNER_GUARD_NAMES = frozenset(
    {
        "InferenceInnerGovernanceGuard",
        "InferenceGovernanceService",
        "InferenceGuardPort",
    },
)


@dataclass(frozen=True, slots=True)
class _InferenceInnerOperation:
    """GR-10-R5-R1 inventory row (test-only; not a public contract)."""

    action: str
    governance_permission_required: bool
    boundary: str
    classification: str


# INFERENCE inner operation inventory on canonical production path (GR-10-R5 / R5-R1).
_INFERENCE_INNER_OPERATIONS: tuple[_InferenceInnerOperation, ...] = (
    _InferenceInnerOperation(
        action="model/provider structured invocation",
        governance_permission_required=True,
        boundary="PRE_MODEL via enforce_pre_model_before_structured_inference",
        classification="governed execution — Policy evaluation row",
    ),
    _InferenceInnerOperation(
        action="inference profile / adapter resolution",
        governance_permission_required=False,
        boundary="configuration resolution feeding PRE_MODEL evaluation context",
        classification="configuration — not a separate Governance permission",
    ),
    _InferenceInnerOperation(
        action="structured-output capability inspection",
        governance_permission_required=False,
        boundary="adapter.supports_structured_output() before PRE_MODEL",
        classification="configuration / local capability introspection — no provider I/O",
    ),
    _InferenceInnerOperation(
        action="meaningful external side effect",
        governance_permission_required=False,
        boundary="NOT_APPLICABLE — not on inference-only seam",
        classification="NOT_APPLICABLE",
    ),
    _InferenceInnerOperation(
        action="tool invocation",
        governance_permission_required=False,
        boundary="NOT_APPLICABLE — no tool runtime on InferenceExecutor path",
        classification="NOT_APPLICABLE",
    ),
    _InferenceInnerOperation(
        action="agent decision / UAEP inner guard",
        governance_permission_required=False,
        boundary="NOT_APPLICABLE — agentic implementation, not inference strategy requirement",
        classification="NOT_APPLICABLE",
    ),
)


def test_gr10_r5_inference_inner_governance_semantics_not_applicable() -> None:
    row = next(
        entry for entry in GR10_INFERENCE_CAPABILITY_SEMANTICS if entry.capability == "Inner Governance"
    )
    assert row.applicability is Gr10Applicability.NOT_APPLICABLE
    assert row.coverage is None
    assert gr10_matrix_inference_status("Inner Governance") is Gr10CoverageStatus.NOT_APPLICABLE
    assert "GR-10-R5" in row.reason
    assert "UAEP" in row.reason


def test_gr10_r5_policy_evaluation_remains_qualified_separate_row() -> None:
    row = next(
        entry for entry in GR10_INFERENCE_CAPABILITY_SEMANTICS if entry.capability == "Policy evaluation"
    )
    assert row.applicability is Gr10Applicability.APPLICABLE
    assert row.coverage is Gr10CoverageStatus.QUALIFIED


def test_gr10_r5_inference_inventory_inner_path_not_applicable() -> None:
    inference = next(entry for entry in GR10_PRODUCTION_INVENTORY if entry.strategy == "INFERENCE")
    assert "NOT_APPLICABLE" in inference.inner_path
    assert "PRE_MODEL" in inference.inner_path


def test_gr10_r5_inference_executor_no_canonical_inner_guard_import() -> None:
    source = _INFERENCE.read_text(encoding="utf-8-sig")
    assert "canonical_inner_governance" not in source
    assert "CanonicalInnerExecutionGuardPort" not in source
    assert "MeaningfulSideEffectAuthorizationBoundary" not in source
    for forbidden in _FORBIDDEN_INFERENCE_INNER_GUARD_NAMES:
        assert forbidden not in source


def _inference_executor_execute_function(
    tree: ast.Module,
) -> ast.AsyncFunctionDef | ast.FunctionDef:
    executor_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InferenceExecutor"
    )
    return next(
        node
        for node in executor_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute"
    )


def _select_adapter_assignment_name(execute_fn: ast.AsyncFunctionDef | ast.FunctionDef) -> str | None:
    for node in execute_fn.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "_select_adapter":
            return target.id
    return None


def _enforce_pre_model_uses_adapter_name(
    execute_fn: ast.AsyncFunctionDef | ast.FunctionDef,
    adapter_name: str,
) -> bool:
    for node in execute_fn.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "enforce_pre_model_before_structured_inference":
            continue
        for keyword in call.keywords:
            if keyword.arg == "adapter" and isinstance(keyword.value, ast.Name):
                return keyword.value.id == adapter_name
    return False


def _invoke_uses_adapter_generate_structured(
    execute_fn: ast.AsyncFunctionDef | ast.FunctionDef,
    adapter_name: str,
) -> bool:
    for node in execute_fn.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "_invoke":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "generate_structured"
                and isinstance(func.value, ast.Name)
                and func.value.id == adapter_name
            ):
                return True
    return False


def test_gr10_r5_inference_executor_same_local_adapter_ast_gate() -> None:
    source = _INFERENCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    execute_fn = _inference_executor_execute_function(tree)
    adapter_name = _select_adapter_assignment_name(execute_fn)
    assert adapter_name is not None
    assert _enforce_pre_model_uses_adapter_name(execute_fn, adapter_name)
    assert _invoke_uses_adapter_generate_structured(execute_fn, adapter_name)


def test_gr10_r5_inference_inner_operation_inventory_documented() -> None:
    assert len(_INFERENCE_INNER_OPERATIONS) >= 3
    governed = [
        row for row in _INFERENCE_INNER_OPERATIONS if row.governance_permission_required
    ]
    assert len(governed) == 1
    assert governed[0].action == "model/provider structured invocation"
    assert "PRE_MODEL" in governed[0].boundary
    for row in _INFERENCE_INNER_OPERATIONS:
        if row.governance_permission_required:
            assert "PRE_MODEL" in row.boundary
        elif row.classification.startswith("configuration"):
            assert row.governance_permission_required is False
