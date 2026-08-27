# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.runtime.execution import (
    ExecutionCapability,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_INFERENCE_TOKENS = frozenset(
    {
        "Nexus",
        "GraphExecutor",
        "AgentEngine",
        "UAEP",
        "ToolRuntime",
        "generate_with_tools",
        "stream_with_tools",
        "RuntimeRequest",
        "Task",
        "TaskResult",
        "ExecutionMode",
        "metadata",
        "agent_id",
        "planner",
        "classifier",
    }
)

_FORBIDDEN_DYNAMIC_TOKENS = frozenset(
    {
        "Any",
        "dict[",
        "Mapping[",
        "MutableMapping[",
        "getattr",
        "setattr",
        "hasattr",
        "__getattr__",
        "__dict__",
        "vars(",
        "inspect",
        "importlib",
        "isinstance(",
        "issubclass(",
        "callable(",
        "**kwargs",
    }
)


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    risk: str


class StructuredTestAdapter(LLMAdapter):
  """Deterministic structured-output adapter for inference executor tests."""

  provider = "test-structured"
  model = "test-structured"

  def __init__(self, parsed_output: RiskAssessment) -> None:
    super().__init__()
    self.parsed_output = parsed_output
    self.generate_messages_calls = 0
    self.generate_with_tools_calls = 0
    self.generate_structured_calls = 0
    self.last_messages: tuple[ChatMessage, ...] | None = None
    self.last_output_model: type | None = None
    self.last_run_id: str | None = None

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
    self.generate_messages_calls += 1
    raise AssertionError("generate_messages must not be called for structured inference")

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
    self.generate_with_tools_calls += 1
    raise AssertionError("generate_with_tools must not be called for structured inference")

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
    self.last_messages = tuple(messages)
    self.last_output_model = output_model
    self.last_run_id = run_id
    return LLMStructuredResult(
      parsed=self.parsed_output,
      response=build_adapter_response(content=""),
    )


class NoStructuredSupportAdapter(LLMAdapter):
  provider = "no-structured"
  model = "no-structured"

  def __init__(self) -> None:
    super().__init__()

  @property
  def context_window_tokens(self) -> int:
    return 8192

  def supports_structured_output(self) -> bool:
    return False

  def generate_messages(
    self,
    messages: Sequence[ChatMessage],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    run_id: str | None = None,
  ) -> LLMAdapterResponse:
    return build_adapter_response(content="unused")

  def generate_structured(
    self,
    messages: Sequence[ChatMessage],
    output_model: type,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    run_id: str | None = None,
  ) -> LLMStructuredResult[RiskAssessment]:
    raise AssertionError("generate_structured must not be called when unsupported")


class FailingStructuredAdapter(StructuredTestAdapter):
  def generate_structured(
    self,
    messages: Sequence[ChatMessage],
    output_model: type,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    run_id: str | None = None,
  ) -> LLMStructuredResult[RiskAssessment]:
    raise RuntimeError("adapter-failure")


class IdentityProbingAdmissionHook:
  def __init__(self, captured: dict[str, RunId | AttemptId | ExecutionId]) -> None:
    self._captured = captured
    self.admit_count = 0

  async def admit(
    self,
    request: ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
  ) -> None:
    self.admit_count += 1
    run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    self._captured["hook_run_id"] = run_id
    self._captured["hook_attempt_id"] = attempt_id
    self._captured["hook_execution_id"] = execution_id


def _risk_request(
  *,
  capabilities: frozenset[ExecutionCapability] = frozenset(),
  output_type: type[RiskAssessment] | None = RiskAssessment,
  content: str = "Assess risk",
) -> ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment]:
  return ExecutionRequest(
    input=(ChatMessage(role="user", content=content),),
    output_type=output_type,
    capabilities=capabilities,
  )


def _identity_binding() -> ExecutionIdentityBinding:
  return ExecutionIdentityBinding(
    run_id=mint_run_id(),
    attempt_id=mint_attempt_id(),
    execution_id=mint_execution_id(),
  )


def _inference_stack(
  adapter: LLMAdapter,
  *,
  identity: ExecutionIdentityBinding | None = None,
  admission_hooks: tuple[ExecutionAdmissionHook, ...] = (),
) -> Execution[
    ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
    ExecutionResult[RiskAssessment],
]:
  executor = InferenceExecutor[RiskAssessment](adapter)
  boundary = ExecutionBoundary(
    executor,
    admission_hooks=admission_hooks,
    identity=identity,
  )
  return Execution(boundary)


def test_empty_capabilities_resolve_to_inference() -> None:
  request = _risk_request()

  assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE


@pytest.mark.asyncio
async def test_direct_structured_request_executes_full_path() -> None:
  expected = RiskAssessment(risk="low")
  adapter = StructuredTestAdapter(parsed_output=expected)
  identity = _identity_binding()
  captured: dict[str, RunId | AttemptId | ExecutionId] = {}
  admission_hook = IdentityProbingAdmissionHook(captured)
  execution = _inference_stack(
    adapter,
    identity=identity,
    admission_hooks=(admission_hook,),
  )
  request = _risk_request()

  assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE

  result = await execution.execute(request)

  assert result.status is ExecutionStatus.COMPLETED
  assert result.output == expected
  assert result.output is expected
  assert adapter.generate_structured_calls == 1
  assert adapter.last_output_model is RiskAssessment
  assert adapter.last_messages == request.input
  assert adapter.last_run_id == identity.run_id
  assert admission_hook.admit_count == 1
  assert captured["hook_execution_id"] == identity.execution_id
  assert peek_active_execution_identity() is None
  assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_adapter_exception_propagates_unchanged() -> None:
  adapter = FailingStructuredAdapter(parsed_output=RiskAssessment(risk="low"))
  execution = _inference_stack(adapter, identity=_identity_binding())
  with pytest.raises(RuntimeError, match="adapter-failure"):
    await execution.execute(_risk_request())
  assert peek_active_execution_identity() is None


@pytest.mark.asyncio
async def test_unsupported_structured_output_adapter_fails_before_invoke() -> None:
  adapter = NoStructuredSupportAdapter()
  execution = _inference_stack(
    adapter,
    identity=_identity_binding(),
  )

  with pytest.raises(RuntimeError, match="inference adapter does not support structured output"):
    await execution.execute(_risk_request())


@pytest.mark.asyncio
async def test_output_type_none_fails_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution = _inference_stack(adapter, identity=_identity_binding())

  with pytest.raises(RuntimeError, match="structured inference requires output_type"):
    await execution.execute(_risk_request(output_type=None))

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_tools_request_fails_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution = _inference_stack(adapter, identity=_identity_binding())

  with pytest.raises(RuntimeError, match="InferenceExecutor requires INFERENCE strategy"):
    await execution.execute(
      _risk_request(capabilities=frozenset({ExecutionCapability.TOOLS}))
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_orchestration_request_fails_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution = _inference_stack(adapter, identity=_identity_binding())

  with pytest.raises(RuntimeError, match="InferenceExecutor requires INFERENCE strategy"):
    await execution.execute(
      _risk_request(capabilities=frozenset({ExecutionCapability.ORCHESTRATION}))
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_tools_and_orchestration_fail_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution = _inference_stack(adapter, identity=_identity_binding())

  with pytest.raises(RuntimeError, match="InferenceExecutor requires INFERENCE strategy"):
    await execution.execute(
      _risk_request(
        capabilities=frozenset(
          {ExecutionCapability.TOOLS, ExecutionCapability.ORCHESTRATION}
        )
      )
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_streaming_request_fails_explicitly() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution = _inference_stack(adapter, identity=_identity_binding())

  with pytest.raises(RuntimeError, match="structured inference streaming is not implemented"):
    await execution.execute(
      _risk_request(capabilities=frozenset({ExecutionCapability.STREAMING}))
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_executor_requires_active_execution_identity() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  executor = InferenceExecutor(adapter)

  with pytest.raises(RuntimeError, match="active execution identity required"):
    await executor.execute(_risk_request())


@pytest.mark.asyncio
async def test_inference_executor_requires_active_execution_id() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  run_id = mint_run_id()
  attempt_id = mint_attempt_id()
  token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
  try:
    executor = InferenceExecutor(adapter)
    with pytest.raises(RuntimeError, match="active ExecutionId required"):
      await executor.execute(_risk_request())
  finally:
    reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_generate_messages_and_generate_with_tools_never_used() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="high"))
  execution = _inference_stack(adapter, identity=_identity_binding())

  await execution.execute(_risk_request())

  assert adapter.generate_messages_calls == 0
  assert adapter.generate_with_tools_calls == 0


def test_inference_module_has_no_forbidden_import_tokens() -> None:
  inference_path = Path("intergrax/runtime/execution/inference.py")
  source = inference_path.read_text(encoding="utf-8")
  module = ast.parse(source)
  imported: list[str] = []
  for node in ast.walk(module):
    if isinstance(node, ast.Import):
      imported.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module is not None:
      imported.append(node.module)

  for forbidden in ("intergrax.runtime.nexus", "intergrax.agents", "intergrax.runtime.nexus.tools"):
    assert not any(
      name == forbidden or name.startswith(f"{forbidden}.") for name in imported
    )


def test_inference_source_has_no_forbidden_tokens() -> None:
  source = Path("intergrax/runtime/execution/inference.py").read_text(encoding="utf-8")
  for token in _FORBIDDEN_INFERENCE_TOKENS:
    assert token not in source, f"forbidden token in inference.py: {token}"


def test_inference_source_has_no_forbidden_dynamic_mechanisms() -> None:
  source = Path("intergrax/runtime/execution/inference.py").read_text(encoding="utf-8")
  for token in _FORBIDDEN_DYNAMIC_TOKENS:
    assert token not in source, f"forbidden dynamic token in inference.py: {token}"


def test_inference_executor_not_exported_from_package_root() -> None:
  import intergrax.runtime.execution as execution_package

  assert "InferenceExecutor" not in execution_package.__all__


def test_inference_executor_does_not_bind_or_reset_identity() -> None:
  source = Path("intergrax/runtime/execution/inference.py").read_text(encoding="utf-8")
  assert "bind_active_execution_identity" not in source
  assert "reset_active_execution_identity" not in source
  assert "mint_execution_id" not in source
  assert "mint_run_id" not in source
  assert "mint_attempt_id" not in source
