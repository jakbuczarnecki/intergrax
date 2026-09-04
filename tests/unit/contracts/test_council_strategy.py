# © Artur Czarnecki. All rights reserved.

from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from intergrax.contracts.council_strategy import (
    CouncilDeliberationInput,
    CouncilDeliberationResult,
    CouncilParticipantFailurePolicy,
    CouncilResolutionDisposition,
    CouncilRoundPolicy,
    CouncilStrategy,
    CouncilSynthesisConfiguration,
    council_participant_failure_policy,
    council_round_policy,
    council_strategy_kind,
    council_strategy_registration,
    register_council_strategy,
)
from intergrax.contracts.decision_context_visibility import (
    DeliberationContextId,
    participant_context_visibility_configuration,
    participant_context_visibility_policy,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_participants import (
    participant_binding,
    participant_configuration,
    participant_role_definition,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord
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
from intergrax.llm.messages import ChatMessage

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "adapter",
        "provider",
        "model",
        "executor",
        "runtime",
        "scheduler",
        "retry",
        "checkpoint",
        "budget",
        "authorization",
        "verification",
        "finalization",
        "governance",
        "hitl",
        "nexus",
        "majority",
        "vote",
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
        "CouncilRuntime",
        "runtime.nexus",
        "AuthoritativeAcceptedDecision",
        "AuthoritativeResolutionRecord",
    },
)

_COUNCIL_PRODUCTION_FILES = (
    Path("intergrax/contracts/council_strategy.py"),
    Path("intergrax/runtime/execution/council_deliberation.py"),
    Path("intergrax/runtime/execution/concurrent_execution_work.py"),
)


@dataclass(frozen=True, slots=True)
class SamplePayload:
    recommendation: str


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-123"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _two_participant_council(
    *,
    max_rounds: int = 1,
    minimum_successful_participants: int = 2,
) -> CouncilStrategy:
    roles = (
        participant_role_definition(role_id="architect", instruction="Architect role."),
        participant_role_definition(role_id="risk", instruction="Risk analyst role."),
    )
    participants = participant_configuration(
        roles=roles,
        participants=(
            participant_binding(
                participant_id="participant-a",
                role_id="architect",
                inference_profile_id="profile-a",
            ),
            participant_binding(
                participant_id="participant-b",
                role_id="risk",
                inference_profile_id="profile-b",
            ),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=participants,
        policies=(
            participant_context_visibility_policy(
                role_id="architect",
                visible_contexts=(DeliberationContextId("customer_context"),),
            ),
            participant_context_visibility_policy(
                role_id="risk",
                visible_contexts=(
                    DeliberationContextId("customer_context"),
                    DeliberationContextId("internal_risk_context"),
                ),
            ),
        ),
    )
    return CouncilStrategy(
        participants=participants,
        visibility=visibility,
        round_policy=council_round_policy(max_rounds=max_rounds),
        synthesis=CouncilSynthesisConfiguration(
            synthesis_instruction="Synthesize competing proposals conservatively.",
            failure_policy=council_participant_failure_policy(
                minimum_successful_participants=minimum_successful_participants,
            ),
        ),
    )


def _three_participant_council(
    *,
    minimum_successful_participants: int = 2,
) -> CouncilStrategy:
    roles = (
        participant_role_definition(role_id="architect", instruction="Architect."),
        participant_role_definition(role_id="risk", instruction="Risk."),
        participant_role_definition(role_id="domain", instruction="Domain expert."),
    )
    participants = participant_configuration(
        roles=roles,
        participants=(
            participant_binding(
                participant_id="participant-a",
                role_id="architect",
                inference_profile_id="profile-a",
            ),
            participant_binding(
                participant_id="participant-b",
                role_id="risk",
                inference_profile_id="profile-b",
            ),
            participant_binding(
                participant_id="participant-c",
                role_id="domain",
                inference_profile_id="profile-c",
            ),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=participants,
        policies=(
            participant_context_visibility_policy(
                role_id="architect",
                visible_contexts=(DeliberationContextId("customer_context"),),
            ),
            participant_context_visibility_policy(
                role_id="risk",
                visible_contexts=(DeliberationContextId("customer_context"),),
            ),
            participant_context_visibility_policy(
                role_id="domain",
                visible_contexts=(DeliberationContextId("customer_context"),),
            ),
        ),
    )
    return CouncilStrategy(
        participants=participants,
        visibility=visibility,
        round_policy=council_round_policy(max_rounds=1),
        synthesis=CouncilSynthesisConfiguration(
            synthesis_instruction="Synthesize proposals.",
            failure_policy=council_participant_failure_policy(
                minimum_successful_participants=minimum_successful_participants,
            ),
        ),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_council_implements_decision_strategy() -> None:
    strategy = _two_participant_council()
    assert isinstance(strategy, DecisionStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_council_kind_is_canonical() -> None:
    strategy = _two_participant_council()
    assert strategy.kind == "council"
    assert council_strategy_kind() == strategy.kind


@pytest.mark.unit
@pytest.mark.gate
def test_council_registers_in_existing_registry() -> None:
    strategy = _two_participant_council()
    registry = register_council_strategy(
        decision_strategy_registry(),
        participants=strategy.participants,
        visibility=strategy.visibility,
        round_policy=strategy.round_policy,
        synthesis=strategy.synthesis,
    )
    assert is_decision_strategy_registered(registry, council_strategy_kind()) is True
    resolved = require_registered_decision_strategy(registry, council_strategy_kind())
    assert isinstance(resolved, CouncilStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_council_requires_at_least_two_participants() -> None:
    roles = (participant_role_definition(role_id="solo", instruction="Solo role."),)
    participants = participant_configuration(
        roles=roles,
        participants=(
            participant_binding(
                participant_id="participant-a",
                role_id="solo",
                inference_profile_id="profile-a",
            ),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=participants,
        policies=(
            participant_context_visibility_policy(
                role_id="solo",
                visible_contexts=(DeliberationContextId("customer_context"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="at least two participants"):
        CouncilStrategy(
            participants=participants,
            visibility=visibility,
            round_policy=council_round_policy(max_rounds=1),
            synthesis=CouncilSynthesisConfiguration(
                synthesis_instruction="Synthesize.",
                failure_policy=council_participant_failure_policy(
                    minimum_successful_participants=2,
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_council_round_policy_requires_positive_max_rounds() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        council_round_policy(max_rounds=0)


@pytest.mark.unit
@pytest.mark.gate
def test_council_failure_policy_requires_minimum_two() -> None:
    with pytest.raises(ValueError, match="minimum_successful_participants"):
        council_participant_failure_policy(minimum_successful_participants=1)


@pytest.mark.unit
@pytest.mark.gate
def test_council_deliberation_result_rejects_mixed_states() -> None:
    with pytest.raises(ValueError, match="SYNTHESIZED requires candidate"):
        CouncilDeliberationResult(
            disposition=CouncilResolutionDisposition.SYNTHESIZED,
            proposal_refs=(),
            disagreement=None,
            rounds_used=1,
            candidate=None,
            deadlock_reason=None,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_council_structural_surface_has_no_forbidden_fields() -> None:
    strategy_fields = {field.name for field in fields(CouncilStrategy)}
    assert strategy_fields.isdisjoint(_FORBIDDEN_FIELD_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_council_no_forbidden_production_patterns() -> None:
    for relative_path in _COUNCIL_PRODUCTION_FILES:
        source = relative_path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_SOURCE_TOKENS:
            assert token not in source, f"{relative_path}: forbidden token {token!r}"


@pytest.mark.unit
@pytest.mark.gate
def test_council_result_cannot_contain_authoritative_types() -> None:
    result_fields = {field.name for field in fields(CouncilDeliberationResult)}
    assert "authoritative" not in result_fields
    assert AuthoritativeAcceptedDecision not in CouncilDeliberationResult.__mro__
    assert AuthoritativeResolutionRecord not in CouncilDeliberationResult.__mro__


@pytest.mark.unit
@pytest.mark.gate
def test_council_duplicate_registration_rejected() -> None:
    strategy = _two_participant_council()
    registration = council_strategy_registration(
        participants=strategy.participants,
        visibility=strategy.visibility,
        round_policy=strategy.round_policy,
        synthesis=strategy.synthesis,
    )
    registry = register_decision_strategy(decision_strategy_registry(), registration)
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        register_decision_strategy(registry, registration)


@pytest.mark.unit
@pytest.mark.gate
def test_council_deliberation_input_validates_task_messages() -> None:
    with pytest.raises(ValueError, match="task_messages"):
        CouncilDeliberationInput(
            identity=_identity(),
            task_messages=(),
            context_surfaces=(),
            output_type=SamplePayload,
            artifact_kind=validate_decision_artifact_kind("incident_resolution"),
        )
