# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.architecture.cost_budget import BudgetEnvelope, BudgetScope
from intergrax.runtime.nexus.tools.runtime_bound_catalog import RUNTIME_BOUND_TOOL_IDS
from intergrax.runtime.nexus.tools.uaep_invocation_wiring import build_uaep_wiring_overlay
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from testing_support.builder import build_runtime_execution_context_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_BOUND_SEED = "runtime-bound-catalog"


@pytest.fixture
def exec_ctx(tmp_path: Path) -> RuntimeExecutionContext:
    task_id = str(canonical_task_id_for_tests(_BOUND_SEED))
    workspace = ShadowWorkspace.create(tmp_path, tenant_id="t1", task_id=task_id)
    ctx = build_runtime_execution_context_for_tests(seed=_BOUND_SEED, agent_id="agent-1")
    ctx.metadata["shadow_workspace"] = workspace
    return ctx


def test_uaep_overlay_projects_shadow_workspace(exec_ctx: RuntimeExecutionContext) -> None:
    overlay = build_uaep_wiring_overlay(exec_ctx)
    assert overlay.workspace is exec_ctx.metadata["shadow_workspace"]


def test_uaep_overlay_projects_cost_envelopes(exec_ctx: RuntimeExecutionContext) -> None:
    exec_ctx.metadata["cost_envelopes"] = (
        BudgetEnvelope(scope=BudgetScope.TENANT, scope_id="t-1", limit_amount=100.0, spent_amount=50.0),
    )
    overlay = build_uaep_wiring_overlay(exec_ctx)
    assert overlay.cost_envelopes is not None
    assert len(overlay.cost_envelopes) == 1


def test_runtime_bound_catalog_includes_cost_forecast() -> None:
    assert "cost.forecast_spend" in RUNTIME_BOUND_TOOL_IDS
