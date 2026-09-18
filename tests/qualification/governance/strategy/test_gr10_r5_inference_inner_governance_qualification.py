# © Artur Czarnecki. All rights reserved.

"""GR-10-R5 — INFERENCE Inner Governance enterprise qualification (semantics only)."""

from __future__ import annotations

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

# Protected inner actions on canonical INFERENCE production path (GR-10-R5 inventory).
_INFERENCE_INNER_ACTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "model/provider structured invocation",
        "yes",
        "PRE_MODEL via enforce_pre_model_before_structured_inference (Policy evaluation row)",
    ),
    (
        "inference profile / adapter selection",
        "yes when profile_id set",
        "configuration resolution only; provider call remains behind PRE_MODEL",
    ),
    (
        "meaningful external side effect",
        "no",
        "NOT_APPLICABLE — not on inference-only seam",
    ),
    (
        "tool invocation",
        "no",
        "NOT_APPLICABLE — no tool runtime on InferenceExecutor path",
    ),
    (
        "agent decision / UAEP inner guard",
        "no",
        "NOT_APPLICABLE — agentic implementation, not inference strategy requirement",
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


def test_gr10_r5_inference_executor_provider_only_after_pre_model_ast() -> None:
    source = _INFERENCE.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    pre_model_line = None
    generate_structured_line = None
    for index, line in enumerate(lines, start=1):
        if "enforce_pre_model_before_structured_inference" in line:
            pre_model_line = index
        if "generate_structured" in line:
            generate_structured_line = index
    assert pre_model_line is not None
    assert generate_structured_line is not None
    assert pre_model_line < generate_structured_line


def test_gr10_r5_inner_action_inventory_documented() -> None:
    assert len(_INFERENCE_INNER_ACTIONS) >= 3
    governed = [row for row in _INFERENCE_INNER_ACTIONS if row[1].startswith("yes")]
    assert len(governed) == 2
    for _action, required, boundary in governed:
        assert "PRE_MODEL" in boundary or "NOT_APPLICABLE" in boundary
        if required.startswith("yes"):
            assert "PRE_MODEL" in boundary
