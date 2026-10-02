# © Artur Czarnecki. All rights reserved.

"""CE-01-R1-R1 — structural UCL / executor / event recorder replaceability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, get_type_hints

import pytest

from intergrax.context.assembly_runtime import (
    ContextAssemblyUCLRuntime,
    validate_context_assembly_event_recorder,
    validate_context_assembly_ucl_runtime,
)
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_recording import RuntimeEventRecorderPort
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.context_lifecycle.in_memory_repository import (
    InMemoryOptimizationArtifactRepository,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    build_context_assembly_runtime_dependencies,
)
from intergrax.runtime.context_lifecycle.message_sequence_execution_contract import (
    MessageSequenceArtifactExecutionPort,
    MessageSequenceArtifactExecutionRequest,
    MessageSequenceArtifactExecutionResult,
)
from intergrax.runtime.context_lifecycle.message_sequence_execution_port import (
    MessageSequenceArtifactExecutionPort as MessageSequenceArtifactExecutionPortReexport,
)
from intergrax.runtime.events.context_skill_recording import record_context_validation_failed
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.ucl_orchestration import NexusUCLRuntimeDependencies
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutor,
)
from testing_support.builder import canonical_execution_identity_scope

pytestmark = [pytest.mark.unit, pytest.mark.gate]


_TASK = "task_00000000000000000000000000000001"
_RUN = "run_00000000000000000000000000000001"


class _Adapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-r1-r1"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


class _SyntheticExecutor:
    def execute(
        self,
        request: MessageSequenceArtifactExecutionRequest,
    ) -> MessageSequenceArtifactExecutionResult:
        _ = request
        raise AssertionError("synthetic executor must not run in this proof")


class _BrokenExecutor:
    pass


@dataclass(frozen=True, slots=True)
class _SyntheticUCLRuntime:
    repository: InMemoryOptimizationArtifactRepository
    message_sequence_executor: _SyntheticExecutor
    strategy_versions: dict[str, str]
    artifact_id_factory: Callable[[], str]
    wait_timeout_seconds: float = 0.25


class _CapturingEventRecorder:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def record(
        self,
        event: RuntimeEvent,
        *,
        tenant_id: str | None = None,
    ) -> None:
        _ = tenant_id
        self.events.append(event)


def _assembly_request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-r1-r1",
        run_id=_RUN,
        task_id=_TASK,
        tenant_id="tenant-a",
        assembly_scope="acp_step",
        objective="ce-r1-r1 structural boundary",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=2000),
        assembly_options=TaskContextAssemblyOptions(),
    )


def test_synthetic_ucl_runtime_satisfies_contract_not_nexus_class() -> None:
    repo = InMemoryOptimizationArtifactRepository()
    synthetic = _SyntheticUCLRuntime(
        repository=repo,
        message_sequence_executor=_SyntheticExecutor(),
        strategy_versions={"strategy.a": "1.0.0"},
        artifact_id_factory=lambda: "artifact-synthetic",
    )
    assert not isinstance(synthetic, NexusUCLRuntimeDependencies)
    assert isinstance(synthetic, ContextAssemblyUCLRuntime)
    validate_context_assembly_ucl_runtime(synthetic)


def test_nexus_ucl_runtime_accepts_synthetic_executor_without_subclass() -> None:
    repo = InMemoryOptimizationArtifactRepository()
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=_SyntheticExecutor(),
        strategy_versions={"strategy.a": "1.0.0"},
        artifact_id_factory=lambda: "artifact-1",
    )
    assert isinstance(runtime.message_sequence_executor, MessageSequenceArtifactExecutionPort)
    assert not isinstance(runtime.message_sequence_executor, MessageSequenceArtifactExecutor)


def test_concrete_executor_structurally_satisfies_execution_port() -> None:
    executor = MessageSequenceArtifactExecutor(
        preflight=lambda _call: None,
        invoke_model=lambda _call: LLMAdapterResponse(content="x"),
        count_tokens=lambda text: len(text),
    )
    assert isinstance(executor, MessageSequenceArtifactExecutionPort)


def test_invalid_ucl_runtime_fail_closed() -> None:
    repo = InMemoryOptimizationArtifactRepository()
    invalid = _SyntheticUCLRuntime(
        repository=repo,
        message_sequence_executor=_BrokenExecutor(),  # type: ignore[arg-type]
        strategy_versions={},
        artifact_id_factory=lambda: "artifact-1",
    )
    with pytest.raises(ValueError, match="message_sequence_executor"):
        validate_context_assembly_ucl_runtime(invalid)


def test_invalid_event_recorder_fail_closed() -> None:
    with pytest.raises(ValueError, match="RuntimeEventRecorderPort"):
        validate_context_assembly_event_recorder(object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_engine_accepts_structural_ucl_runtime_without_nexus_type() -> None:
    repo = InMemoryOptimizationArtifactRepository()
    synthetic = _SyntheticUCLRuntime(
        repository=repo,
        message_sequence_executor=_SyntheticExecutor(),
        strategy_versions={"strategy.a": "1.0.0"},
        artifact_id_factory=lambda: "artifact-synthetic",
    )
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[ChatMessage(role="user", content="structural-ucl")],
        ucl_runtime=synthetic,
    )
    ctx = ContextProviderContext(engine_id="default", runtime=runtime)
    assembled = await DefaultNexusContextEngine().assemble(_assembly_request(), provider_ctx=ctx)
    assert assembled.messages


def test_synthetic_event_recorder_receives_ce_context_events() -> None:
    recorder = _CapturingEventRecorder()
    with canonical_execution_identity_scope(_RUN):
        record_context_validation_failed(
            recorder,
            errors=("synthetic proof",),
            stage="ce_r1_r1",
            task_id=_TASK,
            run_id=_RUN,
        )
    assert len(recorder.events) == 1


def test_message_sequence_execution_port_reexport_is_canonical_contract() -> None:
    assert MessageSequenceArtifactExecutionPortReexport is MessageSequenceArtifactExecutionPort


def test_message_sequence_artifact_compatibility_reexport_is_canonical() -> None:
    from intergrax.runtime.token_optimization import message_sequence_artifact as impl

    from intergrax.runtime.context_lifecycle import message_sequence_execution_contract as contract

    assert impl.MessageSequenceArtifactExecutionRequest is contract.MessageSequenceArtifactExecutionRequest
    assert impl.MessageSequenceArtifactExecutionResult is contract.MessageSequenceArtifactExecutionResult
    assert impl.MessageSequenceArtifactSourceGroupProof is contract.MessageSequenceArtifactSourceGroupProof
    assert impl.MessageSequenceArtifactExecutionReceipt is contract.MessageSequenceArtifactExecutionReceipt


def test_synthetic_executor_structural_proof_uses_contract_dtos_only() -> None:
    assert isinstance(_SyntheticExecutor(), MessageSequenceArtifactExecutionPort)
    hints = get_type_hints(_SyntheticExecutor.execute, globalns=globals())
    assert hints["request"] is MessageSequenceArtifactExecutionRequest
    assert hints["return"] is MessageSequenceArtifactExecutionResult
