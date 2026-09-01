# © Artur Czarnecki. All rights reserved.

"""DS-NEXUS-01 — Decision-facing canonical Execution work submission seam."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_task_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.runtime.execution import ExecutionCapability, ExecutionStatus
from intergrax.runtime.execution.active_execution_work_port import (
    get_active_execution_work_port,
    require_active_execution_work_port,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.execution_work_port import (
    ExecutionWorkPort,
    child_execution_work_port,
)
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.orchestration import OrchestrationExecutor
from intergrax.runtime.execution.request import ExecutionRequest as NeutralExecutionRequest
from intergrax.runtime.execution.result import ExecutionResult
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
    resolve_root_execution_context,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
_DECISION_CONTRACT_GLOB = "intergrax/contracts/decision*.py"
_FORBIDDEN_DECISION_TOKENS = (
    "intergrax.runtime.nexus",
    "NexusLoop",
    "OrchestrationExecutor",
    "StrategyExecutionRouter(",
)


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: str


@dataclass(slots=True)
class ChildLineageCapture:
    child_execution_id: ExecutionId | None = None
    parent_execution_id: ExecutionId | None = None
    root_execution_id: ExecutionId | None = None


class StructuredInferenceAdapter(LLMAdapter):
    provider = LLMProvider.OLLAMA
    model = "test"

    def __init__(self, parsed_output: RiskAssessment) -> None:
        super().__init__()
        self.parsed_output = parsed_output
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
    ) -> LLMStructuredResult[RiskAssessment]:
        self.generate_structured_calls += 1
        require_active_execution_identity()
        require_active_execution_id()
        return LLMStructuredResult(
            parsed=self.parsed_output,
            response=build_adapter_response(content=""),
        )


class FakeOrchestrationPort:
    """Composition-root orchestration backend implementing NexusOrchestrationPort."""

    __slots__ = ("_capture", "_result", "calls")

    def __init__(
        self,
        result: TaskResult,
        capture: ChildLineageCapture,
    ) -> None:
        self._result = result
        self._capture = capture
        self.calls: list[Task] = []

    async def handle_task(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult:
        del run_id, attempt_id
        self.calls.append(task)
        self._capture.child_execution_id = require_active_execution_id()
        self._capture.parent_execution_id = peek_active_parent_execution_id()
        require_active_execution_identity()
        return self._result


class OrchestrationOnlyRouter:
    """Composition-root router delegate for orchestration-only child work."""

    __slots__ = ("_executor", "_task")

    def __init__(self, task: Task, backend: FakeOrchestrationPort) -> None:
        self._task = task
        self._executor = OrchestrationExecutor(backend)

    async def execute(
        self,
        request: NeutralExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        if StrategyResolver().resolve(request) is not ExecutionStrategy.ORCHESTRATION:
            raise RuntimeError("expected ORCHESTRATION strategy")
        del request
        return await self._executor.execute(self._task)


class DecisionFacingOrchestrationProbe:
    """Decision-aware helper that knows only canonical Execution abstractions."""

    async def request_orchestration_work(
        self,
        *,
        message: str,
    ) -> TaskResult:
        work_port = require_active_execution_work_port()
        request = NeutralExecutionRequest(
            input=TaskExecutionInput(message=message),
            output_type=TaskResult,
            capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
        )
        return await work_port.execute(request)


class DecisionFacingInferenceProbe:
    """Decision-aware helper requesting non-orchestration inference work."""

    async def request_inference_work(self) -> ExecutionResult[RiskAssessment]:
        work_port = require_active_execution_work_port()
        request = NeutralExecutionRequest(
            input=(ChatMessage(role="user", content="assess"),),
            output_type=RiskAssessment,
        )
        return await work_port.execute(request)


@dataclass(frozen=True, slots=True)
class RootProbeRequest:
    value: str


@dataclass(frozen=True, slots=True)
class RootProbeResult:
    value: str


def _root_context() -> RootExecutionContext:
    return resolve_root_execution_context(
        RootExecutionOptions(authority=ParentExecutionAuthority.unrestricted_root()),
    )


class OrchestrationRootProbeDelegate:
    __slots__ = ("_capture",)

    def __init__(self, capture: ChildLineageCapture) -> None:
        self._capture = capture

    async def execute(self, request: RootProbeRequest) -> RootProbeResult:
        self._capture.root_execution_id = require_active_execution_id()
        probe = DecisionFacingOrchestrationProbe()
        result = await probe.request_orchestration_work(message=request.value)
        return RootProbeResult(value=result.answer or "")


@pytest.mark.asyncio
async def test_decision_facing_probe_submits_orchestration_without_nexus_import() -> None:
    capture = ChildLineageCapture()
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="orchestrate",
        context=TaskContext(capability="echo.basic"),
    )
    expected = TaskResult(
        task_id=task.task_id,
        state=TaskState.COMPLETED,
        answer="orchestrated",
    )
    backend = FakeOrchestrationPort(expected, capture)
    router = OrchestrationOnlyRouter(task, backend)
    work_port = child_execution_work_port(router, ledger=_UNLIMITED_LEDGER)
    runtime = ExecutionRuntime(
        OrchestrationRootProbeDelegate(capture),
        execution_work_port=work_port,
    )
    root_context = _root_context()
    request = NeutralExecutionRequest(
        input=TaskExecutionInput(message="root"),
        output_type=TaskResult,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )

    assert StrategyResolver().resolve(request) is ExecutionStrategy.ORCHESTRATION

    result = await runtime.execute(RootProbeRequest(value="orchestrate"), root_context)

    assert result.value == "orchestrated"
    assert len(backend.calls) == 1
    assert backend.calls[0] is task
    assert capture.child_execution_id is not None
    assert capture.root_execution_id is not None
    assert capture.child_execution_id != capture.root_execution_id
    assert capture.parent_execution_id == capture.root_execution_id


@pytest.mark.asyncio
async def test_inference_child_work_does_not_require_orchestration_backend() -> None:
    expected = RiskAssessment(risk="low")
    adapter = StructuredInferenceAdapter(parsed_output=expected)
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        RiskAssessment,
        ExecutionResult[RiskAssessment],
    ](inference_executor=InferenceExecutor(adapter))
    work_port = child_execution_work_port(router, ledger=_UNLIMITED_LEDGER)

    class InferenceRootDelegate:
        async def execute(self, request: RootProbeRequest) -> ExecutionResult[RiskAssessment]:
            del request
            probe = DecisionFacingInferenceProbe()
            return await probe.request_inference_work()

    runtime = ExecutionRuntime(
        InferenceRootDelegate(),
        execution_work_port=work_port,
    )
    inference_request = NeutralExecutionRequest(
        input=(ChatMessage(role="user", content="assess"),),
        output_type=RiskAssessment,
    )

    assert StrategyResolver().resolve(inference_request) is ExecutionStrategy.INFERENCE

    result = await runtime.execute(RootProbeRequest(value="assess"), _root_context())

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == expected
    assert adapter.generate_structured_calls == 1


@pytest.mark.asyncio
async def test_orchestration_request_fails_closed_without_backend() -> None:
    router = StrategyExecutionRouter[
        TaskExecutionInput,
        TaskResult,
        TaskResult,
    ]()
    work_port = child_execution_work_port(router, ledger=_UNLIMITED_LEDGER)
    runtime = ExecutionRuntime(
        OrchestrationRootProbeDelegate(ChildLineageCapture()),
        execution_work_port=work_port,
    )

    with pytest.raises(RuntimeError, match="ORCHESTRATION strategy is not configured"):
        await runtime.execute(RootProbeRequest(value="missing-backend"), _root_context())


@pytest.mark.asyncio
async def test_ordinary_execution_without_work_port_has_no_active_binding() -> None:
    observed: list[ExecutionWorkPort | None] = []

    class ObservingDelegate:
        async def execute(self, request: RootProbeRequest) -> RootProbeResult:
            observed.append(get_active_execution_work_port())
            return RootProbeResult(value=request.value)

    runtime = ExecutionRuntime(ObservingDelegate())
    assert get_active_execution_work_port() is None

    result = await runtime.execute(RootProbeRequest(value="plain"), _root_context())

    assert result.value == "plain"
    assert observed == [None]
    assert get_active_execution_work_port() is None


def test_decision_contracts_have_no_nexus_visibility() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    contract_paths = sorted(repo_root.glob(_DECISION_CONTRACT_GLOB))
    assert contract_paths, "expected decision contract modules"

    for path in contract_paths:
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_DECISION_TOKENS:
            assert token not in source, f"{path} contains forbidden token: {token}"


def test_decision_facing_probe_classes_have_no_nexus_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    probe_class_names = {
        "DecisionFacingOrchestrationProbe",
        "DecisionFacingInferenceProbe",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in probe_class_names:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom) and child.module is not None:
                assert "intergrax.runtime.nexus" not in child.module
            if isinstance(child, ast.Import):
                for alias in child.names:
                    assert "intergrax.runtime.nexus" not in alias.name
