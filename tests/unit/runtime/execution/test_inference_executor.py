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
    validate_execution_id,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.runtime.execution import (
    ExecutionCapability,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.boundary import ExecutionAdmissionHook
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionOptions,
)
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileCatalog,
    InferenceProfileId,
    InferenceProfileNotFoundError,
)
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
    build_in_memory_governance_evidence_persistence,
)
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_bridge import PreModelPolicyBlockedError
from intergrax.runtime.policy.pre_model_policy_evaluation import PreModelPolicyConfigurationError
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from testing_support.inference_governance_wiring import (
    TEST_INFERENCE_PRINCIPAL_ID,
    TEST_INFERENCE_TENANT_ID,
    TEST_INFERENCE_WORKSPACE_ID,
    bind_test_inference_governance_identity,
    build_test_inference_executor_without_evidence,
    default_test_inference_evidence_persistence,
    governed_inference_executor,
    governed_root_execution_options,
    reset_test_inference_governance_identity,
)

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


class StructuredTestAdapter(BaseLLMAdapter):
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


class NoStructuredSupportAdapter(BaseLLMAdapter):
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


def _root_options(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
) -> RootExecutionOptions:
    return governed_root_execution_options(
        run_id=run_id,
        attempt_id=attempt_id,
    )


def _inference_stack(
  adapter: LLMAdapter,
  *,
  options: RootExecutionOptions | None = None,
  admission_hooks: tuple[ExecutionAdmissionHook, ...] = (),
  policy_engine: PolicyEngine | None = None,
  governance_evidence_persistence: GovernanceEvidencePersistencePort | None = None,
  wire_default_governance_evidence: bool = True,
) -> tuple[
    Execution[
        ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
        ExecutionResult[RiskAssessment],
    ],
    RootExecutionOptions,
]:
  persistence = governance_evidence_persistence
  if persistence is None and wire_default_governance_evidence:
    persistence = build_in_memory_governance_evidence_persistence()
  if persistence is not None:
    executor = governed_inference_executor(
      adapter,
      policy_engine=policy_engine,
      governance_evidence_persistence=persistence,
    )
  else:
    executor = build_test_inference_executor_without_evidence(
      adapter,
      policy_engine=policy_engine,
    )
  router = StrategyExecutionRouter[
    tuple[ChatMessage, ...],
    RiskAssessment,
    ExecutionResult[RiskAssessment],
  ](inference_executor=executor)
  runtime = ExecutionRuntime[
    ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
    ExecutionResult[RiskAssessment],
  ](router, admission_hooks=admission_hooks)
  context = options or _root_options()
  return Execution(runtime), context


def test_empty_capabilities_resolve_to_inference() -> None:
  request = _risk_request()

  assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE


@pytest.mark.asyncio
async def test_direct_structured_request_executes_full_path() -> None:
  expected = RiskAssessment(risk="low")
  adapter = StructuredTestAdapter(parsed_output=expected)
  captured: dict[str, RunId | AttemptId | ExecutionId] = {}
  admission_hook = IdentityProbingAdmissionHook(captured)
  execution, options = _inference_stack(
    adapter,
    admission_hooks=(admission_hook,),
  )
  request = _risk_request()

  assert StrategyResolver().resolve(request) is ExecutionStrategy.INFERENCE

  result = await execution.execute(request, options=options)

  assert result.status is ExecutionStatus.COMPLETED
  assert result.output == expected
  assert result.output is expected
  assert adapter.generate_structured_calls == 1
  assert adapter.last_output_model is RiskAssessment
  assert adapter.last_messages == request.input
  assert adapter.last_run_id == str(captured["hook_run_id"])
  assert admission_hook.admit_count == 1
  assert validate_execution_id(captured["hook_execution_id"])
  assert peek_active_execution_identity() is None
  assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_adapter_exception_propagates_unchanged() -> None:
  adapter = FailingStructuredAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter)
  with pytest.raises(RuntimeError, match="adapter-failure"):
    await execution.execute(_risk_request(), options=options)
  assert peek_active_execution_identity() is None


@pytest.mark.asyncio
async def test_unsupported_structured_output_adapter_fails_before_invoke() -> None:
  adapter = NoStructuredSupportAdapter()
  execution, options = _inference_stack(adapter)

  with pytest.raises(RuntimeError, match="inference adapter does not support structured output"):
    await execution.execute(_risk_request(), options=options)


@pytest.mark.asyncio
async def test_output_type_none_fails_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter)

  with pytest.raises(RuntimeError, match="structured inference requires output_type"):
    await execution.execute(_risk_request(output_type=None), options=options)

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_tools_request_fails_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter)

  with pytest.raises(RuntimeError, match="AGENTIC strategy is not configured"):
    await execution.execute(
      _risk_request(capabilities=frozenset({ExecutionCapability.TOOLS})),
      options=options,
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_orchestration_request_fails_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter)

  with pytest.raises(RuntimeError, match="ORCHESTRATION strategy is not configured"):
    await execution.execute(
      _risk_request(capabilities=frozenset({ExecutionCapability.ORCHESTRATION})),
      options=options,
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_tools_and_orchestration_fail_before_adapter_invocation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter)

  with pytest.raises(RuntimeError, match="ORCHESTRATION strategy is not configured"):
    await execution.execute(
      _risk_request(
        capabilities=frozenset(
          {ExecutionCapability.TOOLS, ExecutionCapability.ORCHESTRATION}
        )
      ),
      options=options,
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_streaming_request_fails_explicitly() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter)

  with pytest.raises(RuntimeError, match="structured inference streaming is not implemented"):
    await execution.execute(
      _risk_request(capabilities=frozenset({ExecutionCapability.STREAMING})),
      options=options,
    )

  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_executor_requires_active_execution_identity() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  executor = governed_inference_executor(
    adapter,
    governance_evidence_persistence=default_test_inference_evidence_persistence(),
  )

  with pytest.raises(RuntimeError, match="active execution identity required"):
    await executor.execute(_risk_request())


@pytest.mark.asyncio
async def test_inference_executor_requires_active_execution_id() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  run_id = mint_run_id()
  attempt_id = mint_attempt_id()
  token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
  try:
    executor = governed_inference_executor(
    adapter,
    governance_evidence_persistence=default_test_inference_evidence_persistence(),
  )
    with pytest.raises(RuntimeError, match="active ExecutionId required"):
      await executor.execute(_risk_request())
  finally:
    reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_generate_messages_and_generate_with_tools_never_used() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="high"))
  execution, options = _inference_stack(adapter)

  await execution.execute(_risk_request(), options=options)

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


def _bind_direct_executor_context() -> tuple[object, object]:
  identity_token = bind_active_execution_identity(
    run_id=mint_run_id(),
    attempt_id=mint_attempt_id(),
    execution_id=mint_execution_id(),
  )
  governance_token = bind_test_inference_governance_identity()
  return identity_token, governance_token


def _reset_direct_executor_context(identity_token: object, governance_token: object) -> None:
  reset_test_inference_governance_identity(governance_token)
  reset_active_execution_identity(identity_token)


class _DenyPreModelRuntime(RuntimePolicyEngine):
  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    return PolicyDecision(
      action=PolicyAction.DENY,
      reason="inference_pre_model_denied",
      policy_rule_id="test.inference_pre_model_denied",
    )


class _RequireHumanPreModelRuntime(RuntimePolicyEngine):
  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    return PolicyDecision(
      action=PolicyAction.REQUIRE_HUMAN,
      reason="inference_pre_model_require_human",
      policy_rule_id="test.inference_pre_model_require_human",
    )


class _EscalatePreModelRuntime(RuntimePolicyEngine):
  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    return PolicyDecision(
      action=PolicyAction.ESCALATE,
      reason="inference_pre_model_escalate",
      policy_rule_id="test.inference_pre_model_escalate",
    )


class _ModifyPreModelRuntime(RuntimePolicyEngine):
  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    return PolicyDecision(
      action=PolicyAction.MODIFY,
      reason="inference_pre_model_modify",
      policy_rule_id="test.inference_pre_model_modify",
    )


class _ExplodingPreModelRuntime(RuntimePolicyEngine):
  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    raise RuntimeError("policy_engine_failure")


class _RecordingPreModelRuntime(RuntimePolicyEngine):
  def __init__(self) -> None:
    super().__init__()
    self.calls = 0
    self.last_tenant_id: str | None = None
    self.last_message_count: int | None = None
    self.last_model_id: str | None = None

  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    self.calls += 1
    self.last_tenant_id = tenant_id
    self.last_principal_id = principal_id
    self.last_agent_id = agent_id
    self.last_message_count = message_count
    if context is not None:
      self.last_model_id = context.model_id
    return PolicyDecision(
      action=PolicyAction.ALLOW,
      reason="allow",
      policy_rule_id="test.allow",
    )


@pytest.mark.asyncio
async def test_inference_pre_model_allow_invokes_provider_once() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  recorder = build_governance_evidence_recorder(persistence=store)
  execution, options = _inference_stack(
    adapter,
    governance_evidence_persistence=store,
  )
  result = await execution.execute(_risk_request(), options=options)
  assert result.status is ExecutionStatus.COMPLETED
  assert adapter.generate_structured_calls == 1
  assert len(store.facts) == 1
  fact = store.facts[0]
  assert fact.evaluation_point is GovernedExecutionEvaluationPoint.PRE_MODEL
  assert fact.decision is PolicyAction.ALLOW
  assert fact.tenant_id == TEST_INFERENCE_TENANT_ID
  assert fact.workspace_id == TEST_INFERENCE_WORKSPACE_ID
  assert fact.principal_id == TEST_INFERENCE_PRINCIPAL_ID
  assert fact.run_id is not None
  assert fact.attempt_id is not None
  assert fact.execution_id is not None
  assert fact.action == "structured_inference.model_invoke"
  assert fact.resource_type == "llm_model"
  assert fact.resource_scope == adapter.model
  assert fact.evaluation_point is not GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION


@pytest.mark.asyncio
async def test_inference_pre_model_deny_blocks_provider() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  engine = PolicyEngine(runtime=_DenyPreModelRuntime())
  store = build_in_memory_governance_evidence_persistence()
  recorder = build_governance_evidence_recorder(persistence=store)
  execution, options = _inference_stack(
    adapter,
    policy_engine=engine,
    governance_evidence_persistence=store,
  )
  with pytest.raises(PreModelPolicyBlockedError):
    await execution.execute(_risk_request(), options=options)
  assert adapter.generate_structured_calls == 0
  assert len(store.facts) == 1
  assert store.facts[0].decision is PolicyAction.DENY
  assert store.facts[0].evaluation_point is GovernedExecutionEvaluationPoint.PRE_MODEL


class _CapturingGovernanceEvidencePersistence(GovernanceEvidencePersistencePort):
  def __init__(self) -> None:
    self.captured: list[GovernanceDecisionEvidenceFact] = []

  def persist(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome:
    self.captured.append(fact)
    return GovernanceEvidencePersistenceOutcome(persisted=True, evidence_id=fact.evidence_id)


@pytest.mark.asyncio
async def test_inference_pre_model_custom_persistence_port_records_typed_fact() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  port = _CapturingGovernanceEvidencePersistence()
  execution, options = _inference_stack(
    adapter,
    governance_evidence_persistence=port,
  )
  await execution.execute(_risk_request(), options=options)
  assert len(port.captured) == 1
  assert isinstance(port.captured[0], GovernanceDecisionEvidenceFact)


@pytest.mark.asyncio
async def test_inference_pre_model_deny_evidence_failure_still_denies_zero_provider() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  store.fail_on_persist = True
  recorder = build_governance_evidence_recorder(persistence=store)
  execution, options = _inference_stack(
    adapter,
    policy_engine=PolicyEngine(runtime=_DenyPreModelRuntime()),
    governance_evidence_persistence=store,
  )
  with pytest.raises(PreModelPolicyBlockedError):
    await execution.execute(_risk_request(), options=options)
  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_pre_model_require_human_emits_fact_before_fail_closed() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  recorder = build_governance_evidence_recorder(persistence=store)
  execution, options = _inference_stack(
    adapter,
    policy_engine=PolicyEngine(runtime=_RequireHumanPreModelRuntime()),
    governance_evidence_persistence=store,
  )
  with pytest.raises(PreModelPolicyBlockedError) as exc_info:
    await execution.execute(_risk_request(), options=options)
  assert len(store.facts) == 1
  assert store.facts[0].decision is PolicyAction.REQUIRE_HUMAN
  assert exc_info.value.decision.action is PolicyAction.DENY


@pytest.mark.asyncio
async def test_inference_pre_model_no_recorder_emits_no_fact() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(
    adapter,
    wire_default_governance_evidence=False,
  )
  await execution.execute(_risk_request(), options=options)
  assert adapter.generate_structured_calls == 1


@pytest.mark.asyncio
async def test_inference_pre_model_distinct_executions_distinct_correlation() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  recorder = build_governance_evidence_recorder(persistence=store)
  execution, options_a = _inference_stack(
    adapter,
    governance_evidence_persistence=store,
  )
  await execution.execute(_risk_request(content="first"), options=options_a)
  options_b = governed_root_execution_options(
    run_id=mint_run_id(),
    attempt_id=mint_attempt_id(),
  )
  two_messages = ExecutionRequest(
    input=(
      ChatMessage(role="user", content="first"),
      ChatMessage(role="user", content="second"),
    ),
    output_type=RiskAssessment,
  )
  await execution.execute(two_messages, options=options_b)
  assert len(store.facts) == 2
  assert store.facts[0].execution_id != store.facts[1].execution_id
  assert store.facts[0].request_digest != store.facts[1].request_digest


@pytest.mark.asyncio
async def test_inference_pre_model_escalate_no_fact_fail_closed() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  execution, options = _inference_stack(
    adapter,
    policy_engine=PolicyEngine(runtime=_EscalatePreModelRuntime()),
    governance_evidence_persistence=store,
  )
  with pytest.raises(PreModelPolicyBlockedError) as exc_info:
    await execution.execute(_risk_request(), options=options)
  assert len(store.facts) == 0
  assert exc_info.value.decision.action is PolicyAction.DENY
  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_pre_model_modify_no_fact_fail_closed() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  execution, options = _inference_stack(
    adapter,
    policy_engine=PolicyEngine(runtime=_ModifyPreModelRuntime()),
    governance_evidence_persistence=store,
  )
  with pytest.raises(PreModelPolicyBlockedError) as exc_info:
    await execution.execute(_risk_request(), options=options)
  assert len(store.facts) == 0
  assert exc_info.value.decision.action is PolicyAction.DENY
  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_pre_model_policy_exception_blocks_provider() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(
    adapter,
    policy_engine=PolicyEngine(runtime=_ExplodingPreModelRuntime()),
  )
  with pytest.raises(RuntimeError, match="policy_engine_failure"):
    await execution.execute(_risk_request(), options=options)
  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_pre_model_missing_policy_dependency_blocks_provider() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  identity_token, governance_token = _bind_direct_executor_context()
  try:
    executor = InferenceExecutor(adapter, policy_engine=None)
    with pytest.raises(PreModelPolicyConfigurationError, match="policy engine required"):
      await executor.execute(_risk_request())
  finally:
    _reset_direct_executor_context(identity_token, governance_token)
  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_pre_model_unsupported_action_blocks_provider() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(
    adapter,
    policy_engine=PolicyEngine(runtime=_RequireHumanPreModelRuntime()),
  )
  with pytest.raises(PreModelPolicyBlockedError):
    await execution.execute(_risk_request(), options=options)
  assert adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_pre_model_evaluator_receives_context() -> None:
  runtime = _RecordingPreModelRuntime()
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter, policy_engine=PolicyEngine(runtime=runtime))
  messages = _risk_request().input
  await execution.execute(_risk_request(), options=options)
  assert runtime.last_tenant_id == TEST_INFERENCE_TENANT_ID
  assert runtime.last_principal_id == TEST_INFERENCE_PRINCIPAL_ID
  assert runtime.last_agent_id is None
  assert runtime.last_message_count == len(messages)
  assert runtime.last_model_id == adapter.model


@pytest.mark.asyncio
async def test_inference_pre_model_runs_before_provider_invocation() -> None:
  call_order: list[str] = []
  runtime = _RecordingPreModelRuntime()

  class OrderedAdapter(StructuredTestAdapter):
    def generate_structured(self, messages, output_model, **kwargs):
      call_order.append("provider")
      return super().generate_structured(messages, output_model, **kwargs)

  original_evaluate = runtime.evaluate_pre_llm

  def _ordered_evaluate(**kwargs):
    call_order.append("policy")
    return original_evaluate(**kwargs)

  runtime.evaluate_pre_llm = _ordered_evaluate  # type: ignore[method-assign]
  adapter = OrderedAdapter(parsed_output=RiskAssessment(risk="low"))
  execution, options = _inference_stack(adapter, policy_engine=PolicyEngine(runtime=runtime))
  await execution.execute(_risk_request(), options=options)
  assert call_order == ["policy", "provider"]


@pytest.mark.asyncio
async def test_inference_pre_model_evidence_failure_still_allows_provider() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  store.fail_on_persist = True
  recorder = build_governance_evidence_recorder(persistence=store)
  execution, options = _inference_stack(
    adapter,
    governance_evidence_persistence=store,
  )
  result = await execution.execute(_risk_request(), options=options)
  assert result.status is ExecutionStatus.COMPLETED
  assert adapter.generate_structured_calls == 1


@pytest.mark.asyncio
async def test_inference_pre_model_evidence_has_no_raw_prompt() -> None:
  adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  store = build_in_memory_governance_evidence_persistence()
  recorder = build_governance_evidence_recorder(persistence=store)
  secret = "super-secret-prompt-token"
  request = ExecutionRequest(
    input=(ChatMessage(role="user", content=secret),),
    output_type=RiskAssessment,
  )
  execution, options = _inference_stack(
    adapter,
    governance_evidence_persistence=store,
  )
  await execution.execute(request, options=options)
  serialized = repr(store.facts[0].model_dump())
  assert secret not in serialized


def _inference_executor_execute_function(tree: ast.Module) -> ast.AsyncFunctionDef | ast.FunctionDef:
  executor_class = next(
    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InferenceExecutor"
  )
  return next(
    node
    for node in executor_class.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute"
  )


def _select_adapter_assignment_name(execute_fn: ast.AsyncFunctionDef | ast.FunctionDef) -> str | None:
  for node in execute_fn.body:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
      continue
    target = node.targets[0]
    if not isinstance(target, ast.Name):
      continue
    if not isinstance(node.value, ast.Call):
      continue
    call = node.value
    if isinstance(call.func, ast.Attribute) and call.func.attr == "_select_adapter":
      return target.id
  return None


def _enforce_pre_model_uses_adapter_name(
  execute_fn: ast.AsyncFunctionDef | ast.FunctionDef,
  adapter_name: str,
) -> bool:
  for node in execute_fn.body:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
      continue
    call = node.value
    if not isinstance(call.func, ast.Name):
      continue
    if call.func.id != "enforce_pre_model_before_structured_inference":
      continue
    for keyword in call.keywords:
      if keyword.arg == "adapter" and isinstance(keyword.value, ast.Name):
        return keyword.value.id == adapter_name
  return False


def _invoke_uses_adapter_generate_structured(
  execute_fn: ast.AsyncFunctionDef | ast.FunctionDef,
  adapter_name: str,
) -> bool:
  for node in execute_fn.body:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "_invoke":
      continue
    for inner in ast.walk(node):
      if not isinstance(inner, ast.Call):
        continue
      func = inner.func
      if (
        isinstance(func, ast.Attribute)
        and func.attr == "generate_structured"
        and isinstance(func.value, ast.Name)
        and func.value.id == adapter_name
      ):
        return True
  return False


def test_inference_executor_pre_model_before_generate_structured_ast_gate() -> None:
  source = Path("intergrax/runtime/execution/inference.py").read_text(encoding="utf-8")
  tree = ast.parse(source)
  execute_fn = _inference_executor_execute_function(tree)
  adapter_name = _select_adapter_assignment_name(execute_fn)
  assert adapter_name is not None
  assert _enforce_pre_model_uses_adapter_name(execute_fn, adapter_name)
  assert _invoke_uses_adapter_generate_structured(execute_fn, adapter_name)


class _ProfileSelectedAdapter(StructuredTestAdapter):
  """Adapter with a distinct model marker for profile-resolution proofs."""

  def __init__(self, marker_model: str, parsed_output: RiskAssessment) -> None:
    super().__init__(parsed_output=parsed_output)
    self.model = marker_model
    self.provider = f"provider-{marker_model}"


class _SingleAdapterProfileResolver:
  def __init__(self, adapter: LLMAdapter) -> None:
    self._adapter = adapter
    self.resolve_calls = 0

  def resolve(self, profile_id: InferenceProfileId) -> LLMAdapter:
    self.resolve_calls += 1
    return self._adapter


class _DenyWhenModelRuntime(RuntimePolicyEngine):
  def __init__(self, blocked_model_id: str) -> None:
    super().__init__()
    self._blocked_model_id = blocked_model_id
    self.evaluate_calls = 0
    self.last_model_id: str | None = None

  def evaluate_pre_llm(
    self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
  ):
    self.evaluate_calls += 1
    if context is not None:
      self.last_model_id = context.model_id
    if context is not None and context.model_id == self._blocked_model_id:
      return PolicyDecision(
        action=PolicyAction.DENY,
        reason="blocked_profile_model",
        policy_rule_id="test.blocked_profile_model",
      )
    return PolicyDecision(
      action=PolicyAction.ALLOW,
      reason="allow",
      policy_rule_id="test.allow",
    )


@pytest.mark.asyncio
async def test_inference_custom_resolver_pre_model_allow_invokes_selected_adapter_only() -> None:
  default_adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  profile_marker = "profile-selected-adapter-x"
  selected_adapter = _ProfileSelectedAdapter(
    profile_marker,
    parsed_output=RiskAssessment(risk="profile"),
  )
  resolver = _SingleAdapterProfileResolver(selected_adapter)
  runtime = _DenyWhenModelRuntime(blocked_model_id="never-this-model")
  executor = governed_inference_executor(
    default_adapter,
    policy_engine=PolicyEngine(runtime=runtime),
    profile_resolver=resolver,
    governance_evidence_persistence=default_test_inference_evidence_persistence(),
  )
  router = StrategyExecutionRouter[
    tuple[ChatMessage, ...],
    RiskAssessment,
    ExecutionResult[RiskAssessment],
  ](inference_executor=executor)
  runtime_exec = ExecutionRuntime[
    ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
    ExecutionResult[RiskAssessment],
  ](router)
  request = ExecutionRequest(
    input=(ChatMessage(role="user", content="x"),),
    output_type=RiskAssessment,
    inference_profile_id=InferenceProfileId("custom"),
  )
  options = _root_options()
  result = await Execution(runtime_exec).execute(request, options=options)
  assert result.status is ExecutionStatus.COMPLETED
  assert resolver.resolve_calls == 1
  assert runtime.evaluate_calls == 1
  assert runtime.last_model_id == profile_marker
  assert selected_adapter.generate_structured_calls == 1
  assert default_adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_custom_resolver_pre_model_deny_blocks_selected_adapter() -> None:
  default_adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  profile_marker = "profile-selected-adapter-deny"
  selected_adapter = _ProfileSelectedAdapter(
    profile_marker,
    parsed_output=RiskAssessment(risk="profile"),
  )
  resolver = _SingleAdapterProfileResolver(selected_adapter)
  runtime = _DenyWhenModelRuntime(blocked_model_id=profile_marker)
  executor = governed_inference_executor(
    default_adapter,
    policy_engine=PolicyEngine(runtime=runtime),
    profile_resolver=resolver,
    governance_evidence_persistence=default_test_inference_evidence_persistence(),
  )
  router = StrategyExecutionRouter[
    tuple[ChatMessage, ...],
    RiskAssessment,
    ExecutionResult[RiskAssessment],
  ](inference_executor=executor)
  runtime_exec = ExecutionRuntime[
    ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
    ExecutionResult[RiskAssessment],
  ](router)
  request = ExecutionRequest(
    input=(ChatMessage(role="user", content="x"),),
    output_type=RiskAssessment,
    inference_profile_id=InferenceProfileId("custom"),
  )
  execution = Execution(runtime_exec)
  options = _root_options()
  with pytest.raises(PreModelPolicyBlockedError):
    await execution.execute(request, options=options)
  assert resolver.resolve_calls == 1
  assert runtime.evaluate_calls == 1
  assert runtime.last_model_id == profile_marker
  assert selected_adapter.generate_structured_calls == 0
  assert default_adapter.generate_structured_calls == 0


@pytest.mark.asyncio
async def test_inference_profile_resolution_failure_skips_pre_model_and_provider() -> None:
  default_adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
  runtime = _RecordingPreModelRuntime()
  catalog = InferenceProfileCatalog((("primary", default_adapter),))
  executor = governed_inference_executor(
    default_adapter,
    policy_engine=PolicyEngine(runtime=runtime),
    profile_resolver=catalog,
    governance_evidence_persistence=default_test_inference_evidence_persistence(),
  )
  router = StrategyExecutionRouter[
    tuple[ChatMessage, ...],
    RiskAssessment,
    ExecutionResult[RiskAssessment],
  ](inference_executor=executor)
  runtime_exec = ExecutionRuntime[
    ExecutionRequest[tuple[ChatMessage, ...], RiskAssessment],
    ExecutionResult[RiskAssessment],
  ](router)
  request = ExecutionRequest(
    input=(ChatMessage(role="user", content="x"),),
    output_type=RiskAssessment,
    inference_profile_id=InferenceProfileId("missing"),
  )
  execution = Execution(runtime_exec)
  options = _root_options()
  with pytest.raises(InferenceProfileNotFoundError):
    await execution.execute(request, options=options)
  assert runtime.calls == 0
  assert default_adapter.generate_structured_calls == 0
