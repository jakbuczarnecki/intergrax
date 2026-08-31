# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
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
    CandidateDecision,
    validate_decision_artifact_kind,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.single_model_strategy import (
    SingleModelDeliberationInput,
    SingleModelInferenceConfiguration,
    SingleModelStrategy,
    single_model_candidate_decision,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileAlreadyRegisteredError,
    InferenceProfileCatalog,
    InferenceProfileId,
    InferenceProfileNotFoundError,
    InferenceProfileResolutionError,
    validate_inference_profile_id,
)
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.execution.result import ExecutionResult
from intergrax.runtime.execution.single_model_deliberation import (
    single_model_inference_execution_request,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass(frozen=True, slots=True)
class SampleDecisionPayload:
    recommendation: str


class FakeAdapterA(LLMAdapter):
    provider = LLMProvider.GROQ
    model = "fake-a"

    def __init__(self) -> None:
        super().__init__()
        self.generate_structured_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        raise AssertionError("generate_messages must not be called")

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        raise AssertionError("generate_with_tools must not be called")

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[SampleDecisionPayload]:
        self.generate_structured_calls += 1
        return LLMStructuredResult(
            parsed=SampleDecisionPayload(recommendation="adapter-a"),
            response=build_adapter_response(content=""),
        )


class FakeAdapterB(LLMAdapter):
    provider = LLMProvider.OLLAMA
    model = "fake-b"

    def __init__(self) -> None:
        super().__init__()
        self.generate_structured_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        raise AssertionError("generate_messages must not be called")

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        raise AssertionError("generate_with_tools must not be called")

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[SampleDecisionPayload]:
        self.generate_structured_calls += 1
        return LLMStructuredResult(
            parsed=SampleDecisionPayload(recommendation="adapter-b"),
            response=build_adapter_response(content=""),
        )


def _profile_catalog() -> InferenceProfileCatalog:
    return InferenceProfileCatalog(
        (
            ("primary", FakeAdapterA()),
            ("cheap", FakeAdapterB()),
        ),
    )


def _deliberation_input() -> SingleModelDeliberationInput[SampleDecisionPayload]:
    return SingleModelDeliberationInput(
        messages=(ChatMessage(role="user", content="Recommend action."),),
        output_type=SampleDecisionPayload,
        artifact_kind=validate_decision_artifact_kind("incident_resolution"),
    )


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


def test_inference_profile_id_validates() -> None:
    profile_id = validate_inference_profile_id("primary")
    assert profile_id == InferenceProfileId("primary")


def test_inference_profile_id_rejects_blank() -> None:
    with pytest.raises(ValueError):
        validate_inference_profile_id("   ")


def test_inference_profile_catalog_duplicate_registration_fails() -> None:
    with pytest.raises(InferenceProfileAlreadyRegisteredError):
        InferenceProfileCatalog(
            (
                ("primary", FakeAdapterA()),
                ("primary", FakeAdapterB()),
            ),
        )


def test_inference_profile_catalog_unknown_profile_fails_closed() -> None:
    catalog = _profile_catalog()
    with pytest.raises(InferenceProfileNotFoundError):
        catalog.resolve(InferenceProfileId("missing"))


@pytest.mark.asyncio
async def test_two_profiles_map_to_two_different_adapters() -> None:
    catalog = _profile_catalog()
    adapter_a = catalog.resolve(InferenceProfileId("primary"))
    adapter_b = catalog.resolve(InferenceProfileId("cheap"))
    assert adapter_a is not adapter_b
    assert adapter_a.provider == LLMProvider.GROQ
    assert adapter_b.provider == LLMProvider.OLLAMA


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile_name", "expected_recommendation", "unused_adapter_provider"),
    [
        ("primary", "adapter-a", "fake-b"),
        ("cheap", "adapter-b", "fake-a"),
    ],
)
async def test_single_model_profile_selects_adapter_through_execution_path(
    profile_name: str,
    expected_recommendation: str,
    unused_adapter_provider: str,
) -> None:
    inference = SingleModelInferenceConfiguration(
        inference_profile_id=validate_inference_profile_id(profile_name),
    )
    strategy = SingleModelStrategy(inference=inference)
    deliberation_input = _deliberation_input()
    request = single_model_inference_execution_request(
        deliberation_input,
        inference=strategy.inference,
    )
    assert request.inference_profile_id == InferenceProfileId(profile_name)
    assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE

    default_adapter = FakeAdapterA() if unused_adapter_provider == "fake-a" else FakeAdapterB()
    executor = InferenceExecutor[SampleDecisionPayload](
        default_adapter,
        profile_resolver=_profile_catalog(),
    )
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        SampleDecisionPayload,
        ExecutionResult[SampleDecisionPayload],
    ](inference_executor=executor)

    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        result = await router.execute(request)
    finally:
        reset_active_execution_identity(token)

    assert result.output.recommendation == expected_recommendation
    assert default_adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_explicit_unknown_profile_fails_closed() -> None:
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="x"),),
        output_type=SampleDecisionPayload,
        inference_profile_id=InferenceProfileId("missing"),
    )
    executor = InferenceExecutor[SampleDecisionPayload](
        FakeAdapterA(),
        profile_resolver=_profile_catalog(),
    )
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        SampleDecisionPayload,
        ExecutionResult[SampleDecisionPayload],
    ](inference_executor=executor)

    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        with pytest.raises(InferenceProfileNotFoundError):
            await router.execute(request)
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_explicit_profile_never_silently_falls_back_to_default() -> None:
    default_adapter = FakeAdapterA()
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="x"),),
        output_type=SampleDecisionPayload,
        inference_profile_id=InferenceProfileId("missing"),
    )
    executor = InferenceExecutor[SampleDecisionPayload](
        default_adapter,
        profile_resolver=_profile_catalog(),
    )
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        SampleDecisionPayload,
        ExecutionResult[SampleDecisionPayload],
    ](inference_executor=executor)

    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        with pytest.raises(InferenceProfileNotFoundError):
            await router.execute(request)
    finally:
        reset_active_execution_identity(token)

    assert default_adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_absent_profile_uses_host_default_adapter() -> None:
    default_adapter = FakeAdapterA()
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="x"),),
        output_type=SampleDecisionPayload,
    )
    executor = InferenceExecutor[SampleDecisionPayload](
        default_adapter,
        profile_resolver=_profile_catalog(),
    )
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        SampleDecisionPayload,
        ExecutionResult[SampleDecisionPayload],
    ](inference_executor=executor)

    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        result = await router.execute(request)
    finally:
        reset_active_execution_identity(token)

    assert result.output.recommendation == "adapter-a"
    assert default_adapter.generate_structured_calls == 1


def test_inference_profile_catalog_is_immutable() -> None:
    profile_entries: list[tuple[str, FakeAdapterA | FakeAdapterB]] = [
        ("primary", FakeAdapterA()),
        ("cheap", FakeAdapterB()),
    ]
    catalog = InferenceProfileCatalog(tuple(profile_entries))
    resolved_primary = catalog.resolve(InferenceProfileId("primary"))
    resolved_cheap = catalog.resolve(InferenceProfileId("cheap"))

    profile_entries.append(("intruder", FakeAdapterB()))
    profile_entries[0] = ("primary", FakeAdapterB())

    assert catalog.resolve(InferenceProfileId("primary")) is resolved_primary
    assert catalog.resolve(InferenceProfileId("cheap")) is resolved_cheap
    with pytest.raises(InferenceProfileNotFoundError):
        catalog.resolve(InferenceProfileId("intruder"))


def test_single_model_candidate_decision_uses_explicit_none_lineage_check() -> None:
    identity = _identity()
    artifact_kind = validate_decision_artifact_kind("incident_resolution")
    payload = SampleDecisionPayload(recommendation="hold")
    candidate = single_model_candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=payload,
        lineage=None,
    )
    assert isinstance(candidate, CandidateDecision)
    assert candidate.lineage.current.version == DecisionVersion(1)


def test_single_model_deliberation_input_rejects_empty_messages() -> None:
    with pytest.raises(ValueError):
        SingleModelDeliberationInput(
            messages=(),
            output_type=SampleDecisionPayload,
            artifact_kind=validate_decision_artifact_kind("incident_resolution"),
        )


@pytest.mark.asyncio
async def test_explicit_profile_without_resolver_fails_closed() -> None:
    request = ExecutionRequest(
        input=(ChatMessage(role="user", content="x"),),
        output_type=SampleDecisionPayload,
        inference_profile_id=InferenceProfileId("primary"),
    )
    executor = InferenceExecutor[SampleDecisionPayload](FakeAdapterA())
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        SampleDecisionPayload,
        ExecutionResult[SampleDecisionPayload],
    ](inference_executor=executor)

    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        with pytest.raises(InferenceProfileResolutionError):
            await router.execute(request)
    finally:
        reset_active_execution_identity(token)
