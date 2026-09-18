# © Artur Czarnecki. All rights reserved.

"""GR-10-R2-C1 — PRE_MODEL public contract gates."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_context import PreModelPhase, PreModelPolicyContext
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_bridge import evaluate_pre_model_policy
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

REPO_ROOT = Path(__file__).resolve().parents[4]
PRODUCTION_ROOT = REPO_ROOT / "intergrax"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_pre_model_signature_requires_principal_and_optional_agent() -> None:
    sig = inspect.signature(RuntimePolicyEngine.evaluate_pre_llm)
    assert "principal_id" in sig.parameters
    principal_param = sig.parameters["principal_id"]
    assert principal_param.default is inspect.Parameter.empty
    agent_param = sig.parameters["agent_id"]
    assert agent_param.default is None


def test_pre_model_empty_principal_denies() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="",
        agent_id=None,
        message_count=1,
    )
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "pre_model_principal_missing"


def test_pre_model_generic_inference_shape_allows() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="principal-1",
        agent_id=None,
        message_count=1,
    )
    assert decision.action is PolicyAction.ALLOW


def test_pre_model_agent_step_with_agent_allows() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="principal-1",
        agent_id="agent-1",
        message_count=1,
        context=PreModelPolicyContext(phase=PreModelPhase.AGENT_STEP, model_id="balanced"),
    )
    assert decision.action is PolicyAction.ALLOW


def test_pre_model_agent_step_without_agent_denies() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="principal-1",
        agent_id=None,
        message_count=1,
        context=PreModelPolicyContext(phase=PreModelPhase.AGENT_STEP, model_id="balanced"),
    )
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "pre_model_agent_step_agent_missing"


def test_pre_model_nexus_planning_without_agent_allows() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="principal-1",
        agent_id=None,
        message_count=1,
        context=PreModelPolicyContext(
            phase=PreModelPhase.NEXUS_PLANNING,
            planner_model_id="planner-a",
        ),
    )
    assert decision.action is PolicyAction.ALLOW


def test_pre_model_empty_agent_string_denies() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="principal-1",
        agent_id="",
        message_count=1,
    )
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "pre_model_agent_id_empty"


def test_pre_model_agent_deny_rule_does_not_apply_without_roster_agent() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="agent-A",
        agent_id=None,
        message_count=1,
        context=PreModelPolicyContext(
            phase=PreModelPhase.AGENT_STEP,
            model_id="blocked-model",
            denied_model_ids=("blocked-model",),
        ),
    )
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "pre_model_agent_step_agent_missing"


def test_pre_model_agent_specific_deny_still_applies() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t1",
        principal_id="principal-1",
        agent_id="agent-A",
        message_count=1,
        context=PreModelPolicyContext(
            phase=PreModelPhase.AGENT_STEP,
            model_id="blocked-model",
            denied_model_ids=("blocked-model",),
        ),
    )
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "agent_model_denied"


def test_pre_model_bridge_requires_principal_keyword() -> None:
    engine = PolicyEngine()
    with pytest.raises(TypeError):
        evaluate_pre_model_policy(
            engine,
            tenant_id="t1",
            agent_id="agent-1",
            message_count=1,
        )


def _production_pre_model_call_sites() -> list[tuple[Path, ast.Call]]:
    sites: list[tuple[Path, ast.Call]] = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        if "test" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Attribute) and func.attr in {
                "evaluate_pre_llm",
                "evaluate_pre_model_policy",
            }:
                name = func.attr
            elif isinstance(func, ast.Name) and func.id in {
                "evaluate_pre_llm",
                "evaluate_pre_model_policy",
            }:
                name = func.id
            if name is None:
                continue
            sites.append((path, node))
    return sites


def test_production_pre_model_callers_pass_principal_id() -> None:
    for path, call in _production_pre_model_call_sites():
        keyword_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "principal_id" in keyword_names, f"{path}: missing principal_id on PRE_MODEL call"


def test_production_pre_model_no_empty_agent_id_literal() -> None:
    for path, call in _production_pre_model_call_sites():
        for kw in call.keywords:
            if kw.arg != "agent_id":
                continue
            if isinstance(kw.value, ast.Constant) and kw.value.value == "":
                pytest.fail(f"{path}: empty agent_id literal on PRE_MODEL call")
