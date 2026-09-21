# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R6-R1 — RuntimeToolInvoker owner-controlled idempotency recomposition."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    build_production_runtime_tool_invoker,
    recompose_runtime_tool_invoker_with_idempotency_store,
)
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSITION = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "runtime_tool_invoker_composition.py"
_OVERLAY = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "agents" / "idempotency_runtime_overlay.py"

_PRIVATE_INVOKER_ATTRS: frozenset[str] = frozenset(
    {
        "_pre_effect_coordinator",
        "_executor",
        "_scope_policy",
        "_meaningful_side_effect_authorization",
        "_agent_runtime_governance",
        "_inner_execution_guard",
        "_sandbox_availability",
        "_dependency_attempt_boundary",
        "_external_operation_store",
        "_external_operation_owner",
        "_external_operation_cancellation_port",
        "_invocation_wiring_resolver",
    },
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


def _minimal_contract() -> ToolContract:
    return ToolContract(
        tool_id="probe.tool",
        name="probe",
        description="probe",
        input_schema=_In,
        output_schema=_Out,
        side_effects=False,
        error_mapping={},
        risk_level=ToolRiskLevel.LOW,
    )


def _build_invoker_without_idempotency() -> RuntimeToolInvoker:
    registry = FakeRegistry(_minimal_contract())
    return build_production_runtime_tool_invoker(registry=registry)


def _collect_invoker_private_attribute_accesses(source: str) -> list[str]:
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "invoker":
            continue
        if node.attr.startswith("_"):
            offenders.append(f"invoker.{node.attr}")
    return offenders


def test_harness_01_w2_r6_r1_composition_has_no_runtime_tool_invoker_private_coupling() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8")
    offenders = _collect_invoker_private_attribute_accesses(source)
    assert offenders == [], (
        "runtime_tool_invoker_composition.py must not access RuntimeToolInvoker private fields:\n"
        + "\n".join(offenders)
    )


def test_harness_01_w2_r6_r1_overlay_has_no_any_seam() -> None:
    source = _OVERLAY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "Any":
            pytest.fail("idempotency_runtime_overlay.py must not reference Any")
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            pytest.fail("idempotency_runtime_overlay.py must not reference typing.Any")


def test_harness_01_w2_r6_r1_with_idempotency_store_adds_coordinator_once() -> None:
    invoker = _build_invoker_without_idempotency()
    store = InMemoryIdempotencyStore()
    replacement = invoker.with_idempotency_store(store)
    assert replacement is not invoker
    assert replacement._pre_effect_coordinator is not None
    again = replacement.with_idempotency_store(InMemoryIdempotencyStore())
    assert again is replacement


def test_harness_01_w2_r6_r1_recompose_delegates_to_owner_api() -> None:
    invoker = _build_invoker_without_idempotency()
    store = InMemoryIdempotencyStore()
    replacement = recompose_runtime_tool_invoker_with_idempotency_store(
        invoker,
        idempotency_store=store,
        production_mode=False,
    )
    assert replacement._pre_effect_coordinator is not None
    assert invoker._execution_pool_closed is True


def test_harness_01_w2_r6_r1_reconfiguration_preserves_registry_and_executor() -> None:
    invoker = _build_invoker_without_idempotency()
    registry = invoker.registry
    executor = invoker._executor
    store = InMemoryIdempotencyStore()
    replacement = invoker.with_idempotency_store(store)
    assert replacement.registry is registry
    assert replacement._executor is executor


def test_harness_01_w2_r6_r1_reconfiguration_closes_replaced_execution_pool() -> None:
    invoker = _build_invoker_without_idempotency()
    assert invoker._execution_pool_closed is False
    replacement = invoker.with_idempotency_store(InMemoryIdempotencyStore())
    assert replacement._execution_pool_closed is False
    assert invoker._execution_pool_closed is True
    replacement.close()
