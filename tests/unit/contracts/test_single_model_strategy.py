# © Artur Czarnecki. All rights reserved.

from dataclasses import dataclass

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
    validate_decision_artifact_kind,
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
from intergrax.contracts.single_model_strategy import (
    SingleModelDeliberationInput,
    SingleModelInferenceConfiguration,
    SingleModelStrategy,
    register_single_model_strategy,
    single_model_candidate_decision,
    single_model_strategy_kind,
    single_model_strategy_registration,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution.request import ExecutionCapability
from intergrax.runtime.execution.single_model_deliberation import (
    single_model_inference_execution_request,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver

_START_HEAD = "5daed37924b2a99884bf318e7e7da07a45470212"


@dataclass(frozen=True, slots=True)
class SampleDecisionPayload:
    recommendation: str


def _inference_config() -> SingleModelInferenceConfiguration:
    return SingleModelInferenceConfiguration(llm_profile_id="primary")


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


def _deliberation_input() -> SingleModelDeliberationInput[SampleDecisionPayload]:
    return SingleModelDeliberationInput(
        messages=(
            ChatMessage(role="user", content="Recommend escalation action."),
        ),
        output_type=SampleDecisionPayload,
        artifact_kind=validate_decision_artifact_kind("incident_resolution"),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_implements_decision_strategy() -> None:
    strategy = SingleModelStrategy(inference=_inference_config())
    assert isinstance(strategy, DecisionStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_kind_is_canonical() -> None:
    strategy = SingleModelStrategy(inference=_inference_config())
    assert strategy.kind == "single_model"
    assert single_model_strategy_kind() == strategy.kind


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_registers_in_existing_registry() -> None:
    registry = register_single_model_strategy(
        decision_strategy_registry(),
        _inference_config(),
    )
    assert is_decision_strategy_registered(registry, single_model_strategy_kind()) is True
    resolved = require_registered_decision_strategy(
        registry,
        single_model_strategy_kind(),
    )
    assert isinstance(resolved, SingleModelStrategy)
    assert resolved.inference.llm_profile_id == "primary"


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_duplicate_registration_fail_closed() -> None:
    registration = single_model_strategy_registration(_inference_config())
    registry = decision_strategy_registry((registration,))
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        register_decision_strategy(registry, registration)


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_no_global_default() -> None:
    import intergrax.contracts.single_model_strategy as module

    registry = decision_strategy_registry()
    assert registry.registrations == ()
    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "GLOBAL_DEFAULT" not in source
    assert "CURRENT_STRATEGY" not in source
    assert "set_default_strategy" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_no_nexus_dependency() -> None:
    import intergrax.contracts.single_model_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "nexus" not in source.lower()


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_no_platform_plugin_discovery() -> None:
    import intergrax.contracts.single_model_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "intergrax.core.plugins.discovery" not in source
    assert "importlib" not in source
    assert "entry_points" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_no_direct_provider_sdk_dependency() -> None:
    import intergrax.contracts.single_model_strategy as module
    import intergrax.runtime.execution.single_model_deliberation as seam_module

    for module_path in (module.__file__, seam_module.__file__):
        assert module_path is not None
        source = open(module_path, encoding="utf-8").read()
        assert "openai" not in source.lower()
        assert "anthropic" not in source.lower()
        assert "gemini" not in source.lower()
        assert "qwen" not in source.lower()


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_uses_canonical_inference_execution_seam() -> None:
    deliberation_input = _deliberation_input()
    request = single_model_inference_execution_request(deliberation_input)
    assert request.input == deliberation_input.messages
    assert request.output_type is SampleDecisionPayload
    assert ExecutionCapability.ORCHESTRATION not in request.capabilities
    assert ExecutionCapability.AGENT not in request.capabilities
    assert ExecutionCapability.TOOLS not in request.capabilities
    assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_emits_candidate_decision_contract() -> None:
    identity = _identity()
    artifact_kind = validate_decision_artifact_kind("incident_resolution")
    payload = SampleDecisionPayload(recommendation="escalate")
    candidate = single_model_candidate_decision(
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
def test_single_model_does_not_create_authoritative_decision() -> None:
    import intergrax.contracts.single_model_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "AuthoritativeAcceptedDecision" not in source
    identity = _identity()
    candidate = single_model_candidate_decision(
        identity=identity,
        artifact_kind=validate_decision_artifact_kind("incident_resolution"),
        payload=SampleDecisionPayload(recommendation="hold"),
    )
    assert not isinstance(candidate, AuthoritativeAcceptedDecision)


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_does_not_run_verification() -> None:
    import intergrax.contracts.single_model_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "decision_verification" not in source.lower()
    assert "critic" not in source.lower()
    assert "judge" not in source.lower()
    assert ".verify(" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_provider_failure_not_mapped_to_rejected() -> None:
    import intergrax.contracts.single_model_strategy as module
    import intergrax.runtime.execution.single_model_deliberation as seam_module

    for module_path in (module.__file__, seam_module.__file__):
        assert module_path is not None
        source = open(module_path, encoding="utf-8").read()
        assert "REJECTED" not in source
        assert "DecisionResolution" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_registry_remains_immutable() -> None:
    from dataclasses import FrozenInstanceError

    registry = register_single_model_strategy(
        decision_strategy_registry(),
        _inference_config(),
    )
    with pytest.raises((AttributeError, FrozenInstanceError)):
        setattr(registry, "registrations", ())


@pytest.mark.unit
@pytest.mark.gate
def test_single_model_inference_configuration_rejects_blank_profile_id() -> None:
    with pytest.raises(ValueError):
        SingleModelInferenceConfiguration(llm_profile_id="   ")


@pytest.mark.unit
@pytest.mark.gate
def test_session_start_head_recorded() -> None:
    assert _START_HEAD == "5daed37924b2a99884bf318e7e7da07a45470212"
