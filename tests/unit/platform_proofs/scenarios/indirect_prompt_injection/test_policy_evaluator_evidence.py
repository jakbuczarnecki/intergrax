from __future__ import annotations

import pytest

from platform_proofs.scenarios.indirect_prompt_injection.application.scenario import (
    ScenarioExecutionResult,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import (
    WorkflowKind,
    build_scenario_environment_profile,
    read_only_deny_rule,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import (
    build_attack_fixture,
    build_authorized_write_fixture,
    AttackVariantId,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.contracts import ProofVerdict
from platform_proofs.scenarios.indirect_prompt_injection.proof.evaluator import (
    evaluate_attack_case,
    evaluate_authorized_write_control,
    evaluate_scenario_run,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.evidence_builder import (
    build_platform_proof_evidence,
)
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode

pytestmark = pytest.mark.unit


def test_read_only_policy_profile_denies_write_tool() -> None:
    env = build_scenario_environment_profile(WorkflowKind.SAFE_READ)
    assert env.policy_rules is not None
    assert env.policy_rules.policy_enforcement_mode is PolicyEnforcementMode.ENFORCE
    assert env.policy_rules.inline_rules[0] == read_only_deny_rule()


def test_authorized_write_profile_has_no_deny_rule() -> None:
    env = build_scenario_environment_profile(WorkflowKind.AUTHORIZED_WRITE)
    assert env.policy_rules is not None
    assert env.policy_rules.inline_rules == []


def _result(**overrides: object) -> ScenarioExecutionResult:
    base = {
        "outcome": "RESOLVED",
        "terminal_summary": "summary",
        "order_facts": {},
        "retrieved_notes": ({"note_id": "n1", "content": "x"},),
        "tool_traces": (),
        "policy_evaluations": ({"action": "deny"},),
        "planner_rounds": (),
        "write_tool_proposed": True,
        "write_tool_executed": False,
        "policy_denied": True,
        "matched_policy_rule_ids": ("deny_order_update_shipping_address_read_only",),
        "provider_write_count": 0,
        "model_provider": "openai",
        "model_name": "gpt-test",
        "workflow_kind": "SAFE-READ",
        "leak_scan_blob": "{}",
        "run_id": "run-1",
        "tenant_id": "tenant",
    }
    base.update(overrides)
    return ScenarioExecutionResult(**base)


def test_evaluator_rejects_missing_write_proposal() -> None:
    fixture = build_attack_fixture(AttackVariantId.ATTACK_DIRECT)
    evaluation = evaluate_attack_case(_result(write_tool_proposed=False, policy_denied=False), fixture)
    assert evaluation.verdict is ProofVerdict.NOT_EXERCISED


def test_evaluator_rejects_provider_writes_on_attack() -> None:
    fixture = build_attack_fixture(AttackVariantId.ATTACK_DIRECT)
    evaluation = evaluate_attack_case(_result(provider_write_count=1), fixture)
    assert evaluation.verdict is ProofVerdict.FAIL


def test_evaluator_rejects_deny_everything_positive_control() -> None:
    fixture = build_authorized_write_fixture()
    evaluation = evaluate_authorized_write_control(
        _result(
            write_tool_proposed=True,
            write_tool_executed=False,
            policy_denied=True,
            provider_write_count=0,
            workflow_kind="AUTHORIZED-WRITE",
        ),
        fixture,
    )
    assert evaluation.verdict is ProofVerdict.FAIL


def test_evidence_projection_builds_v3_payload() -> None:
    fixture = build_attack_fixture(AttackVariantId.ATTACK_DIRECT)
    result = _result()
    evaluation = evaluate_scenario_run(result, fixture)
    evidence = build_platform_proof_evidence(
        result,
        evaluation=evaluation,
        fixture=fixture,
        source_revision="abc123",
    )
    assert evidence.schema_version == "intergrax.platform_proof_evidence.v3"
    assert evidence.proof_identity.proof_id == "SCENARIO-INDIRECT-PROMPT-INJECTION"
