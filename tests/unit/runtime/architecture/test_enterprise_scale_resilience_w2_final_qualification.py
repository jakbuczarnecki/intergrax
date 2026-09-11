# © Artur Czarnecki. All rights reserved.

"""W2 Final — dependency resilience qualification gates (flow + orthogonality)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def test_llm_final_flow_tenant_then_resilience_then_admission_inside_physical() -> None:
    """Required order: tenant quota → retry budget → rate → CB → admission → SDK."""
    adapter = _read("intergrax/llm_adapters/contracts/llm_adapter.py")
    resilience = _read("intergrax/llm_adapters/_shared/resilience.py")

    execute_block = adapter[adapter.index("def _execute(") : adapter.index("def _execute_streaming(")]
    assert execute_block.index("check_llm_tenant_quota") < execute_block.index(
        "return execute_with_resilience"
    )
    assert "boundary.acquire" in adapter
    assert execute_block.index("_run_physical_provider_attempt") < execute_block.index(
        "return execute_with_resilience"
    )

    loop_start = resilience.index("def execute_with_resilience")
    loop_body = resilience[loop_start : loop_start + 3500]
    assert loop_body.index("begin_physical_attempt") < loop_body.index("_check_distributed_rate_limit")
    assert loop_body.index("_check_distributed_rate_limit") < loop_body.index("_acquire_local_rate_limit")
    assert loop_body.index("_acquire_local_rate_limit") < loop_body.index("_check_circuit")
    assert loop_body.index("_check_circuit") < loop_body.index("result = fn()")


def test_w2_local_implementations_do_not_import_orchestration_layers() -> None:
    """ETAP 4 — module-level orthogonality (contracts + local ports only)."""
    retry_budget = _read("intergrax/runtime/resilience/local_provider_retry_budget.py")
    rate_limit = _read("intergrax/runtime/resilience/local_provider_rate_limit.py")
    admission = _read("intergrax/runtime/resilience/local_dependency_concurrency_admission.py")
    boundary = _read("intergrax/runtime/resilience/dependency_attempt_execution_boundary.py")

    forbidden_in_retry_budget = (
        "circuit",
        "rate_limit",
        "resilience",
        "dependency_admission",
        "llm_adapters",
    )
    for token in forbidden_in_retry_budget:
        assert token not in retry_budget

    forbidden_in_rate_limit = ("retry_budget", "circuit", "admission", "resilience", "llm_adapters")
    for token in forbidden_in_rate_limit:
        assert token not in rate_limit

    forbidden_in_admission = ("retry_budget", "provider_rate_limit", "circuit", "execute_with_resilience")
    for token in forbidden_in_admission:
        assert token not in admission

    forbidden_in_boundary = ("retry_budget", "provider_rate_limit", "execute_with_resilience", "resilience.py")
    for token in forbidden_in_boundary:
        assert token not in boundary


def test_qualification_matrix_modules_exist() -> None:
    """Scenarios 1–8 map to existing behavioral suites (no duplicate matrix tests)."""
    expected = (
        "tests/unit/llm_adapters/test_enterprise_scale_resilience_w2_c_retry_containment.py",
        "tests/unit/llm_adapters/test_llm_provider_dependency_admission.py",
        "tests/unit/runtime/resilience/test_dependency_attempt_execution_boundary.py",
        "tests/unit/runtime/resilience/test_local_dependency_concurrency_admission.py",
        "tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_dependency_admission.py",
    )
    for rel in expected:
        assert (_REPO_ROOT / rel).is_file(), rel
