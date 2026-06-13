# © Artur Czarnecki. All rights reserved.

"""IDEAL-L3 W2 depth gate evidence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.contracts.subtask_contract import SubtaskContract
from intergrax.contracts.task_envelope_stream import TaskEnvelopeChunk, assemble_envelope_from_chunks
from intergrax.contracts.partial_result_contract import PartialResultContract
from intergrax.runtime.architecture.evaluation_scenario_loader import load_scenario_library
from intergrax.runtime.context.citation_chain import CitationChain
from intergrax.runtime.context.context_golden_harness import load_context_golden_cases
from intergrax.runtime.nexus.budget.quota_enforcement import QuotaAction, QuotaExceededError, TenantQuota, assert_quota_allows
from intergrax.runtime.nexus.subagents.delegation_contract_enforcer import (
    DelegationToolPolicyError,
    enforce_subtask_tool_allowlist,
)
from intergrax.runtime.nexus.subagents.delegation_decision import decision_record_for_delegation
from intergrax.runtime.registry.semver_compat import is_compatible_runtime
from intergrax.runtime.reliability.step_retry_budget import StepRetryBudget
from intergrax.runtime.security.pii_redaction import redact_pii
from intergrax.runtime.security.tool_injection_guard import ToolInjectionError, assert_tool_input_safe
from intergrax.runtime.task.task import TaskResult, TaskState

pytestmark = [pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ideal_w2_streaming_intake_assembly() -> None:
    envelope = assemble_envelope_from_chunks(
        [
            TaskEnvelopeChunk(sequence=0, content="hello "),
            TaskEnvelopeChunk(sequence=1, content="world", is_final=True),
        ],
        tenant_id="t1",
        user_id="u1",
    )
    assert envelope.message == "hello world"


def test_ideal_w2_semver_compat() -> None:
    assert is_compatible_runtime("1.0.0", "1.2.0").compatible
    assert not is_compatible_runtime("2.0.0", "1.9.0").compatible


def test_ideal_w2_delegation_decision_record() -> None:
    contract = SubtaskContract(child_agent_id="echo", objective="delegate search")
    record = decision_record_for_delegation(contract.to_delegation_spec(), parent_agent_id="parent")
    assert record.delegation_target == "echo"
    assert record.delegation_rationale == "delegate search"


def test_ideal_w2_subtask_tool_allowlist() -> None:
    contract = SubtaskContract(child_agent_id="echo", allowed_tools=("tool.a",))
    enforce_subtask_tool_allowlist(contract, "tool.a")
    with pytest.raises(DelegationToolPolicyError):
        enforce_subtask_tool_allowlist(contract, "tool.b")


def test_ideal_w2_partial_result_contract_on_task_result() -> None:
    result = TaskResult(
        task_id="t1",
        state=TaskState.PARTIALLY_COMPLETED,
        partial=PartialResultContract(completed_steps=("s1",), partial_answer="partial"),
    )
    assert result.partial is not None
    assert result.partial.completed_steps == ("s1",)


def test_ideal_w2_quota_hard_stop() -> None:
    with pytest.raises(QuotaExceededError):
        assert_quota_allows(spent_usd=10.0, quota=TenantQuota(max_cost_usd=5.0))
    assert assert_quota_allows(spent_usd=3.0, quota=TenantQuota(max_cost_usd=5.0)) is QuotaAction.ALLOW


def test_ideal_w2_pii_redaction() -> None:
    assert "[REDACTED_EMAIL]" in redact_pii("contact me at user@example.com")


def test_ideal_w2_step_retry_budget() -> None:
    budget = StepRetryBudget(max_retries=1)
    assert budget.can_retry()
    exhausted = budget.consume()
    assert not exhausted.can_retry()


def test_ideal_w2_scenario_and_context_libraries() -> None:
    library = load_scenario_library(REPO_ROOT)
    assert len(library.scenarios) >= 2
    cases = load_context_golden_cases(REPO_ROOT)
    assert cases


def test_ideal_w2_citation_chain() -> None:
    chain = CitationChain()
    chain.add(output_ref="out-1", fragment_id="frag-1", source_id="src-1")
    assert len(chain.links) == 1


def test_ideal_w2_w2_script_gates() -> None:
    scripts = [
        "check_registry_snapshot_diff.py",
        "check_context_golden.py",
        "check_eval_scenario_library.py",
        "check_capability_edge_catalog_sync.py",
        "check_pre_context_policy_wiring.py",
        "check_tool_injection_defense.py",
        "check_architecture_debt_register.py",
    ]
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"
