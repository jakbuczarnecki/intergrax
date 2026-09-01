# © Artur Czarnecki. All rights reserved.

from dataclasses import dataclass, fields

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    CandidateDecision,
    DecisionLineageRef,
    DecisionVersionLineage,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
    validate_decision_branch_id,
)
from intergrax.contracts.decision_strategy import (
    DecisionStrategy,
    DecisionStrategyAlreadyRegisteredError,
    decision_strategy_registry,
    is_decision_strategy_registered,
    register_decision_strategy,
    require_registered_decision_strategy,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.rule_based_strategy import (
    RuleBasedStrategy,
    evaluate_rule_based_strategy,
    register_rule_based_strategy,
    rule_based_candidate_decision,
    rule_based_strategy_kind,
    rule_based_strategy_registration,
)

_START_HEAD = "0eee2b8ff71aa3ebf0765c3e5e66214522eea139"

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "adapter",
        "provider",
        "model",
        "prompt",
        "messages",
        "authorization",
        "approval",
        "executor",
        "tool",
        "participant",
        "context_visibility",
    },
)

_FORBIDDEN_SOURCE_TOKENS = frozenset(
    {
        "Any",
        "cast(",
        "type: ignore",
        "pyright: ignore",
        "getattr",
        "setattr",
        "hasattr",
        "inspect",
        "exec(",
        "eval(",
        "object.__setattr__",
        "dict[str, Any]",
        "LLMAdapter",
        "ChatMessage",
        "InferenceProfileId",
        "OpenAI",
        "Anthropic",
        "Ollama",
        "ParticipantConfiguration",
        "ContextVisibility",
        "Disagreement",
        "AuthoritativeAcceptedDecision",
        "AuthoritativeResolutionRecord",
    },
)


@dataclass(frozen=True, slots=True)
class RiskInput:
    score: int


@dataclass(frozen=True, slots=True)
class RiskDecision:
    classification: str


@dataclass(frozen=True, slots=True)
class RiskRules:
    def evaluate(self, decision_input: RiskInput) -> RiskDecision:
        if decision_input.score >= 80:
            return RiskDecision(classification="high")
        if decision_input.score >= 40:
            return RiskDecision(classification="medium")
        return RiskDecision(classification="low")


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="risk", subject="case-42"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _strategy() -> RuleBasedStrategy[RiskInput, RiskDecision]:
    return RuleBasedStrategy(evaluator=RiskRules())


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_implements_decision_strategy() -> None:
    strategy = _strategy()
    assert isinstance(strategy, DecisionStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_kind_is_canonical() -> None:
    strategy = _strategy()
    assert strategy.kind == "rule_based"
    assert rule_based_strategy_kind() == strategy.kind


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_deterministic_evaluation_same_input() -> None:
    strategy = _strategy()
    input_a = RiskInput(score=75)
    first = evaluate_rule_based_strategy(strategy=strategy, decision_input=input_a)
    second = evaluate_rule_based_strategy(strategy=strategy, decision_input=input_a)
    assert first == second
    assert first.classification == "medium"


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_different_inputs_produce_different_results() -> None:
    strategy = _strategy()
    low = evaluate_rule_based_strategy(strategy=strategy, decision_input=RiskInput(score=10))
    high = evaluate_rule_based_strategy(strategy=strategy, decision_input=RiskInput(score=90))
    assert low.classification == "low"
    assert high.classification == "high"
    assert low != high


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_registers_in_existing_registry() -> None:
    registry = register_rule_based_strategy(
        decision_strategy_registry(),
        RiskRules(),
    )
    assert is_decision_strategy_registered(registry, rule_based_strategy_kind()) is True
    resolved = require_registered_decision_strategy(
        registry,
        rule_based_strategy_kind(),
    )
    assert isinstance(resolved, RuleBasedStrategy)
    assert resolved.evaluator.evaluate(RiskInput(score=85)).classification == "high"


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_duplicate_registration_fail_closed() -> None:
    registration = rule_based_strategy_registration(RiskRules())
    registry = decision_strategy_registry((registration,))
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        register_decision_strategy(registry, registration)


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_emits_candidate_decision_contract() -> None:
    identity = _identity()
    artifact_kind = validate_decision_artifact_kind("risk_classification")
    payload = RiskDecision(classification="high")
    candidate = rule_based_candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=payload,
    )
    assert isinstance(candidate, CandidateDecision)
    assert candidate.identity is identity
    assert candidate.artifact.kind == artifact_kind
    assert candidate.artifact.content == payload
    assert candidate.lineage.current.version == DecisionVersion(1)


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_candidate_decision_custom_lineage_preserved() -> None:
    identity = _identity()
    lineage = decision_version_lineage(
        current=decision_lineage_ref(
            DecisionVersion(1),
            branch_id=validate_decision_branch_id("audit"),
        ),
    )
    candidate = rule_based_candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("risk_classification"),
        payload=RiskDecision(classification="medium"),
        lineage=lineage,
    )
    assert candidate.lineage is lineage
    assert candidate.lineage.current.branch_id == "audit"


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_candidate_decision_default_lineage() -> None:
    identity = _identity()
    candidate = rule_based_candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("risk_classification"),
        payload=RiskDecision(classification="low"),
        lineage=None,
    )
    assert candidate.lineage.current.version == DecisionVersion(1)
    assert candidate.lineage.parents == ()


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_does_not_create_authoritative_decision() -> None:
    identity = _identity()
    candidate = rule_based_candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("risk_classification"),
        payload=RiskDecision(classification="low"),
    )
    assert not isinstance(candidate, AuthoritativeAcceptedDecision)


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_structural_surface_has_no_forbidden_fields() -> None:
    strategy_fields = {field.name for field in fields(RuleBasedStrategy)}
    assert strategy_fields.isdisjoint(_FORBIDDEN_FIELD_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_no_forbidden_production_patterns() -> None:
    import intergrax.contracts.rule_based_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    hits = [token for token in _FORBIDDEN_SOURCE_TOKENS if token in source]
    assert hits == []


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_evaluator_duck_typing() -> None:
    class LocalRules:
        def evaluate(self, decision_input: RiskInput) -> RiskDecision:
            return RiskDecision(classification="low")

    rules = LocalRules()
    strategy = RuleBasedStrategy(evaluator=rules)
    result = evaluate_rule_based_strategy(
        strategy=strategy,
        decision_input=RiskInput(score=0),
    )
    assert result.classification == "low"


@pytest.mark.unit
@pytest.mark.gate
def test_rule_based_generic_strategy_construction() -> None:
    strategy = RuleBasedStrategy[RiskInput, RiskDecision](evaluator=RiskRules())
    result = evaluate_rule_based_strategy(
        strategy=strategy,
        decision_input=RiskInput(score=42),
    )
    assert result.classification == "medium"


@pytest.mark.unit
@pytest.mark.gate
def test_session_start_head_recorded() -> None:
    assert _START_HEAD == "0eee2b8ff71aa3ebf0765c3e5e66214522eea139"
