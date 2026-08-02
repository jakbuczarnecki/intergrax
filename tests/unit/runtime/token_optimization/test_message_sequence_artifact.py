# © Artur Czarnecki. All rights reserved.

"""Unit tests for MessageSequenceArtifactExecutor (CTX-UCL-4)."""

from __future__ import annotations

import ast
import hashlib
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.context.session_history import SessionHistoryMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ArtifactCreationCoordinationStatus,
    ArtifactCreationReservation,
    ArtifactValidationStatus,
    ArtifactValidationSummary,
    ContextOptimizationDecision,
    ContextOptimizationPolicy,
    ContextOptimizationReasonCode,
    ModelCallExecutionScope,
    OptimizationArtifactType,
    OptimizationExecutionGuard,
)
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    compute_artifact_content_hash,
)
from intergrax.runtime.context_lifecycle.serialization import compute_artifact_lookup_key_hash
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    InternalMessageSequenceModelCall,
    MessageSequenceArtifactExecutionError,
    MessageSequenceArtifactExecutionReason,
    MessageSequenceArtifactExecutionReceipt,
    MessageSequenceArtifactExecutionRequest,
    MessageSequenceArtifactExecutionResult,
    MessageSequenceArtifactExecutor,
    MessageSequenceArtifactSourceGroupProof,
)

_FIXED_NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC)
_TENANT = "tenant-1"
_SCOPE = "scope-1"
_POLICY_VERSION = "pol-v1"
_VALIDATION_VERSION = "val-v1"
_STRATEGY_ID = "message_sequence_v1"
_STRATEGY_VERSION = "1.0"
_PARENT_OP = "parent-op-1"


def _group_hash(
  messages: tuple[SessionHistoryMessage, ...],
) -> str:
  payload = "|".join(
    f"{message.message_id}:{message.content_hash}"
    for message in messages
  )
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _composite_hash(
  group_hashes: tuple[str, ...],
) -> str:
  return hashlib.sha256(
    "|".join(group_hashes).encode("utf-8")
  ).hexdigest()


def _default_group_proofs(
  messages: tuple[SessionHistoryMessage, ...],
) -> tuple[MessageSequenceArtifactSourceGroupProof, ...]:
  return tuple(
    MessageSequenceArtifactSourceGroupProof(
      source_refs=(message.message_id,),
      source_content_hash=_group_hash((message,)),
    )
    for message in messages
  )


def _message(message_id: str, sequence: int, content: str = "content") -> SessionHistoryMessage:
  return SessionHistoryMessage(
    message_id=message_id,
    sequence=sequence,
    role="user",
    content=content,
  )


def _lookup_key(
  source_refs: tuple[str, ...] = ("msg-1", "msg-2"),
  *,
  tenant_id: str = _TENANT,
  strategy_id: str = _STRATEGY_ID,
  policy_version: str = _POLICY_VERSION,
  validation_version: str = _VALIDATION_VERSION,
  target_tokens: int = 50,
  lossiness_profile: str = "lossy",
  protected_region_policy_version: str | None = None,
  artifact_type: OptimizationArtifactType = OptimizationArtifactType.MESSAGE_SEQUENCE,
  budget_class: str | None = None,
  source_content_hash: str | None = None,
) -> Any:
  from intergrax.runtime.context_lifecycle.contracts import ArtifactLookupKey

  compression_target = (
    ArtifactCompressionTarget(budget_class=budget_class)
    if budget_class is not None
    else ArtifactCompressionTarget(target_tokens=target_tokens)
  )
  if source_content_hash is None:
    messages = _source_messages()
    proofs = _default_group_proofs(messages)
    source_content_hash = _composite_hash(
      tuple(proof.source_content_hash for proof in proofs)
    )
  return ArtifactLookupKey(
    tenant_id=tenant_id,
    context_scope_id=_SCOPE,
    artifact_type=artifact_type,
    source_content_hash=source_content_hash,
    strategy_id=strategy_id,
    strategy_version=_STRATEGY_VERSION,
    policy_version=policy_version,
    validation_contract_version=validation_version,
    compression_target=compression_target,
    lossiness_profile=lossiness_profile,
    source_refs=source_refs,
    protected_region_policy_version=protected_region_policy_version,
  )


def _policy(
  *,
  enabled: bool = True,
  allow_lossy: bool = True,
  allow_llm_summarization: bool = True,
  allowed_artifact_types: tuple[OptimizationArtifactType, ...] = (
    OptimizationArtifactType.MESSAGE_SEQUENCE,
  ),
  allowed_strategy_ids: tuple[str, ...] = (_STRATEGY_ID,),
  policy_version: str = _POLICY_VERSION,
  validation_version: str = _VALIDATION_VERSION,
  minimum_quality_score: float | None = None,
  protected_region_policy_version: str | None = None,
) -> ContextOptimizationPolicy:
  return ContextOptimizationPolicy(
    policy_version=policy_version,
    validation_contract_version=validation_version,
    enabled=enabled,
    allow_lossy=allow_lossy,
    allow_llm_summarization=allow_llm_summarization,
    allowed_artifact_types=allowed_artifact_types,
    allowed_strategy_ids=allowed_strategy_ids,
    minimum_quality_score=minimum_quality_score,
    protected_region_policy_version=protected_region_policy_version,
  )


def _parent_guard(
  *,
  operation_id: str = _PARENT_OP,
  active_hashes: tuple[str, ...] = (),
  active_strategies: tuple[str, ...] = (),
) -> OptimizationExecutionGuard:
  return OptimizationExecutionGuard(
    execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    operation_id=operation_id,
    parent_operation_id=None,
    optimization_depth=0,
    active_artifact_lookup_key_hashes=active_hashes,
    active_strategy_ids=active_strategies,
  )


def _coordination(
  lookup_key: Any,
  *,
  status: ArtifactCreationCoordinationStatus = ArtifactCreationCoordinationStatus.ACQUIRED,
  owner_operation_id: str = _PARENT_OP,
  tenant_id: str = _TENANT,
  lookup_hash_override: str | None = None,
  reservation_hash_override: str | None = None,
  reservation_tenant_override: str | None = None,
  reservation_owner_override: str | None = None,
) -> ArtifactCreationCoordinationResult:
  lookup_hash = lookup_hash_override or compute_artifact_lookup_key_hash(lookup_key)
  reservation_hash = reservation_hash_override or lookup_hash
  reservation_tenant = reservation_tenant_override or tenant_id
  reservation_owner = reservation_owner_override or owner_operation_id
  acquired = _FIXED_NOW
  reservation = ArtifactCreationReservation(
    reservation_id="res-1",
    artifact_lookup_key_hash=reservation_hash,
    tenant_id=reservation_tenant,
    owner_operation_id=reservation_owner,
    acquired_at=acquired,
    lease_deadline=acquired + timedelta(seconds=60),
  )
  if status is ArtifactCreationCoordinationStatus.ACQUIRED:
    return ArtifactCreationCoordinationResult(
      status=status,
      artifact_lookup_key_hash=lookup_hash,
      state_version=1,
      reservation=reservation,
    )
  if status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE:
    from intergrax.runtime.context_lifecycle.repository import OptimizationArtifactReference

    ref = OptimizationArtifactReference(
      tenant_id=_TENANT,
      artifact_id="artifact-1",
      artifact_lookup_key_hash=lookup_hash,
      artifact_content_hash="contenthash",
      artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
    )
    return ArtifactCreationCoordinationResult(
      status=status,
      artifact_lookup_key_hash=lookup_hash,
      state_version=1,
      artifact_reference=ref,
    )
  if status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS:
    return ArtifactCreationCoordinationResult(
      status=status,
      artifact_lookup_key_hash=lookup_hash,
      state_version=1,
      reservation=reservation,
      reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_IN_PROGRESS,
    )
  if status is ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT:
    return ArtifactCreationCoordinationResult(
      status=status,
      artifact_lookup_key_hash=lookup_hash,
      state_version=1,
      reservation=reservation,
      reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT,
    )
  raise ValueError(f"unsupported coordination status for fixture: {status}")


def _source_messages() -> tuple[SessionHistoryMessage, ...]:
  return (_message("msg-1", 0, "hello"), _message("msg-2", 1, "world"))


def _request(
  *,
  decision: ContextOptimizationDecision = ContextOptimizationDecision.CREATE_ARTIFACT,
  lookup_key: Any | None = None,
  policy: ContextOptimizationPolicy | None = None,
  parent_guard: OptimizationExecutionGuard | None = None,
  source_messages: tuple[SessionHistoryMessage, ...] | None = None,
  source_group_proofs: tuple[MessageSequenceArtifactSourceGroupProof, ...] | None = None,
  coordination: ArtifactCreationCoordinationResult | None = None,
) -> MessageSequenceArtifactExecutionRequest:
  messages = source_messages or _source_messages()
  proofs = source_group_proofs or _default_group_proofs(messages)
  if lookup_key is None:
    key = _lookup_key(
      source_refs=tuple(message.message_id for message in messages),
      source_content_hash=_composite_hash(
        tuple(proof.source_content_hash for proof in proofs)
      ),
    )
  else:
    key = lookup_key
  guard = parent_guard or _parent_guard()
  pol = policy or _policy()
  coord = coordination or _coordination(key)
  return MessageSequenceArtifactExecutionRequest(
    decision=decision,
    coordination=coord,
    lookup_key=key,
    policy=pol,
    parent_guard=guard,
    source_messages=messages,
    source_group_proofs=proofs,
  )


def _executor(
  *,
  preflight: Any | None = None,
  invoke_model: Any | None = None,
  count_tokens: Any | None = None,
  quality_evaluator: Any | None = None,
  clock: Any | None = None,
  operation_id_factory: Any | None = None,
  receipt_id_factory: Any | None = None,
) -> MessageSequenceArtifactExecutor:
  preflight_calls: list[InternalMessageSequenceModelCall] = []
  model_calls: list[InternalMessageSequenceModelCall] = []
  token_count_calls: list[str] = []

  def _preflight(call: InternalMessageSequenceModelCall) -> None:
    preflight_calls.append(call)

  def _invoke(call: InternalMessageSequenceModelCall) -> LLMAdapterResponse:
    model_calls.append(call)
    return LLMAdapterResponse(content="summary output")

  def _count(text: str) -> int:
    token_count_calls.append(text)
    return len(text.split())

  exec_ = MessageSequenceArtifactExecutor(
    preflight=preflight or _preflight,
    invoke_model=invoke_model or _invoke,
    count_tokens=count_tokens or _count,
    quality_evaluator=quality_evaluator,
    clock=clock or (lambda: _FIXED_NOW),
    operation_id_factory=operation_id_factory or (lambda: "internal-op-1"),
    receipt_id_factory=receipt_id_factory or (lambda: "receipt-1"),
  )
  exec_._test_preflight_calls = preflight_calls  # type: ignore[attr-defined]
  exec_._test_model_calls = model_calls  # type: ignore[attr-defined]
  exec_._test_token_count_calls = token_count_calls  # type: ignore[attr-defined]
  return exec_


def test_valid_execution() -> None:
  request = _request()
  lookup_hash = compute_artifact_lookup_key_hash(request.lookup_key)
  executor = _executor()
  result = executor.execute(request)

  preflight_calls = executor._test_preflight_calls  # type: ignore[attr-defined]
  model_calls = executor._test_model_calls  # type: ignore[attr-defined]
  assert len(preflight_calls) == 1
  assert len(model_calls) == 1
  assert preflight_calls[0] is model_calls[0]

  call = preflight_calls[0]
  assert call.execution_guard.execution_scope is ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL
  assert call.execution_guard.optimization_depth == 1
  assert call.execution_guard.parent_operation_id == _PARENT_OP
  assert lookup_hash in call.execution_guard.active_artifact_lookup_key_hashes
  assert _STRATEGY_ID in call.execution_guard.active_strategy_ids
  assert call.max_output_tokens == 50
  assert call.temperature == 0.0
  assert call.run_id == call.execution_guard.operation_id

  assert result.validation.status is ArtifactValidationStatus.PASSED
  assert result.artifact_content_hash == compute_artifact_content_hash(result.payload)
  assert result.receipt.parent_operation_id == _PARENT_OP
  assert result.receipt.artifact_lookup_key_hash == lookup_hash
  assert result.validation.safe_metadata["parent_operation_id"] == _PARENT_OP
  assert result.validation.safe_metadata["artifact_lookup_key_hash"] == lookup_hash


@pytest.mark.parametrize(
  "decision",
  [
    ContextOptimizationDecision.NO_OP,
    ContextOptimizationDecision.SELECT_ONLY,
    ContextOptimizationDecision.REUSE_ARTIFACT,
    ContextOptimizationDecision.POLICY_BLOCKED,
    ContextOptimizationDecision.FAIL_CLOSED,
  ],
)
def test_decision_gate_blocks_non_create(decision: ContextOptimizationDecision) -> None:
  request = _request(decision=decision)
  clock = MagicMock()
  op_factory = MagicMock()
  receipt_factory = MagicMock()
  executor = _executor(
    clock=clock,
    operation_id_factory=op_factory,
    receipt_id_factory=receipt_factory,
  )
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.EXECUTOR_REQUIRES_CREATE_ARTIFACT.value
  )
  assert len(executor._test_preflight_calls) == 0  # type: ignore[attr-defined]
  assert len(executor._test_model_calls) == 0  # type: ignore[attr-defined]
  clock.assert_not_called()
  op_factory.assert_not_called()
  receipt_factory.assert_not_called()


@pytest.mark.parametrize(
  "status",
  [
    ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
    ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS,
    ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT,
  ],
)
def test_coordination_gate(status: ArtifactCreationCoordinationStatus) -> None:
  key = _lookup_key()
  coordination = _coordination(key, status=status)
  request = _request(lookup_key=key, coordination=coordination)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.EXECUTOR_REQUIRES_ACQUIRED_RESERVATION.value
  )
  assert len(executor._test_model_calls) == 0  # type: ignore[attr-defined]


def test_lookup_coordination_hash_mismatch() -> None:
  key = _lookup_key()
  coordination = _coordination(key, lookup_hash_override="wrong-hash")
  request = _request(lookup_key=key, coordination=coordination)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.LOOKUP_IDENTITY_MISMATCH.value


def test_reservation_hash_mismatch() -> None:
  key = _lookup_key()
  coordination = _coordination(key, reservation_hash_override="wrong-hash")
  request = _request(lookup_key=key, coordination=coordination)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.LOOKUP_IDENTITY_MISMATCH.value


def test_tenant_mismatch() -> None:
  key = _lookup_key()
  coordination = _coordination(key, reservation_tenant_override="other-tenant")
  request = _request(lookup_key=key, coordination=coordination)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.LOOKUP_IDENTITY_MISMATCH.value


def test_reservation_owner_mismatch() -> None:
  key = _lookup_key()
  coordination = _coordination(key, reservation_owner_override="other-owner")
  request = _request(lookup_key=key, coordination=coordination)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.LOOKUP_IDENTITY_MISMATCH.value


@pytest.mark.parametrize(
  "policy_kwargs",
  [
    {"enabled": False},
    {"allow_llm_summarization": False},
    {"allow_lossy": False, "allow_llm_summarization": False},
    {"allowed_artifact_types": (OptimizationArtifactType.TEXT,)},
    {"allowed_strategy_ids": ("other-strategy",)},
    {"policy_version": "other-version"},
    {"validation_version": "other-validation"},
    {"lossiness_profile": "lossless"},
  ],
  indirect=False,
)
def test_policy_gate(policy_kwargs: dict[str, Any]) -> None:
  lookup_kwargs: dict[str, Any] = {}
  if "lossiness_profile" in policy_kwargs:
    lookup_kwargs["lossiness_profile"] = policy_kwargs.pop("lossiness_profile")
  if "policy_version" in policy_kwargs:
    lookup_kwargs["policy_version"] = policy_kwargs.pop("policy_version")
  if "validation_version" in policy_kwargs:
    lookup_kwargs["validation_version"] = policy_kwargs.pop("validation_version")
  key = _lookup_key(**lookup_kwargs)
  pol = _policy(**policy_kwargs)
  request = _request(lookup_key=key, policy=pol)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.POLICY_DISALLOWED.value


def test_budget_class_rejected() -> None:
  key = _lookup_key(budget_class="small")
  request = _request(lookup_key=key)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.POLICY_DISALLOWED.value


def test_source_sequence_missing_message() -> None:
  messages = (_message("msg-1", 0),)
  proofs = _default_group_proofs(messages)
  key = _lookup_key()
  request = _request(lookup_key=key, source_messages=messages, source_group_proofs=proofs)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.SOURCE_SEQUENCE_MISMATCH.value


def test_source_sequence_additional_message() -> None:
  messages = (*_source_messages(), _message("msg-3", 2))
  proofs = _default_group_proofs(messages)
  key = _lookup_key()
  request = _request(lookup_key=key, source_messages=messages, source_group_proofs=proofs)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.SOURCE_SEQUENCE_MISMATCH.value


def test_source_sequence_reversed() -> None:
  messages = (_message("msg-2", 1), _message("msg-1", 0))
  proofs = _default_group_proofs(messages)
  key = _lookup_key()
  request = _request(lookup_key=key, source_messages=messages, source_group_proofs=proofs)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.SOURCE_SEQUENCE_MISMATCH.value


def test_duplicate_source_message_id_in_request() -> None:
  messages = (_message("msg-1", 0), _message("msg-1", 1))
  with pytest.raises(ValueError, match="duplicates"):
    MessageSequenceArtifactExecutionRequest(
      decision=ContextOptimizationDecision.CREATE_ARTIFACT,
      coordination=_coordination(_lookup_key()),
      lookup_key=_lookup_key(),
      policy=_policy(),
      parent_guard=_parent_guard(),
      source_messages=messages,
      source_group_proofs=_default_group_proofs(messages),
    )


def test_recursion_parent_internal_scope() -> None:
  internal_parent = OptimizationExecutionGuard(
    execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
    operation_id="internal-parent",
    parent_operation_id="grandparent",
    optimization_depth=1,
  )
  key = _lookup_key()
  coordination = _coordination(key, owner_operation_id="internal-parent")
  request = _request(lookup_key=key, parent_guard=internal_parent, coordination=coordination)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == ContextOptimizationReasonCode.OPTIMIZATION_DEPTH_EXCEEDED.value
  assert len(executor._test_preflight_calls) == 0  # type: ignore[attr-defined]


def test_recursion_parent_depth_one_primary_rejected_at_guard() -> None:
  with pytest.raises(ValueError, match="optimization_depth"):
    OptimizationExecutionGuard(
    execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    operation_id="parent-bad",
    parent_operation_id=None,
    optimization_depth=1,
  )
  with pytest.raises(ValueError):
    OptimizationExecutionGuard(
      execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
      operation_id="parent-bad2",
      parent_operation_id=None,
      optimization_depth=1,
    )


def test_recursion_active_lookup_hash() -> None:
  key = _lookup_key()
  lookup_hash = compute_artifact_lookup_key_hash(key)
  parent = _parent_guard(active_hashes=(lookup_hash,))
  request = _request(lookup_key=key, parent_guard=parent)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == ContextOptimizationReasonCode.OPTIMIZATION_RECURSION_BLOCKED.value
  assert len(executor._test_preflight_calls) == 0  # type: ignore[attr-defined]


def test_recursion_active_strategy() -> None:
  parent = _parent_guard(active_strategies=(_STRATEGY_ID,))
  request = _request(parent_guard=parent)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == ContextOptimizationReasonCode.OPTIMIZATION_RECURSION_BLOCKED.value


def test_preflight_runs_before_model() -> None:
  order: list[str] = []

  def preflight(call: InternalMessageSequenceModelCall) -> None:
    order.append("preflight")
    assert call.execution_guard.execution_scope is ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL

  def invoke(call: InternalMessageSequenceModelCall) -> LLMAdapterResponse:
    order.append("model")
    return LLMAdapterResponse(content="summary")

  executor = _executor(preflight=preflight, invoke_model=invoke)
  executor.execute(_request())
  assert order == ["preflight", "model"]


def test_preflight_failure_blocks_model() -> None:
  def preflight(_: InternalMessageSequenceModelCall) -> None:
    raise RuntimeError("preflight failed")

  executor = _executor(preflight=preflight)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.INTERNAL_PREFLIGHT_FAILED.value
  )
  assert len(executor._test_model_calls) == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
  "response",
  [
    LLMAdapterResponse(content=""),
    LLMAdapterResponse(content="   "),
    LLMAdapterResponse(content="ok", refusal="no"),
    LLMAdapterResponse(content="ok", tool_calls=(object(),)),  # type: ignore[arg-type]
  ],
)
def test_invalid_model_output(response: LLMAdapterResponse) -> None:
  executor = _executor(invoke_model=lambda _: response)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.INVALID_MODEL_OUTPUT.value


def test_output_exceeds_target() -> None:
  executor = _executor(count_tokens=lambda _: 100)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.OUTPUT_EXCEEDS_TARGET.value


def test_model_exception() -> None:
  def invoke(_: InternalMessageSequenceModelCall) -> LLMAdapterResponse:
    raise RuntimeError("model failed")

  executor = _executor(invoke_model=invoke)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.INTERNAL_MODEL_CALL_FAILED.value
  )


def test_quality_minimum_absent_evaluator_not_called() -> None:
  evaluator = MagicMock()
  executor = _executor(quality_evaluator=evaluator)
  executor.execute(_request())
  evaluator.assert_not_called()


def test_quality_minimum_present_evaluator_absent() -> None:
  request = _request(policy=_policy(minimum_quality_score=0.5))
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.QUALITY_VALIDATION_UNAVAILABLE.value
  )


def test_quality_score_below_minimum() -> None:
  request = _request(policy=_policy(minimum_quality_score=0.8))
  executor = _executor(quality_evaluator=lambda _m, _s: 0.5)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.QUALITY_VALIDATION_FAILED.value


def test_quality_score_equal_minimum_passes() -> None:
  request = _request(policy=_policy(minimum_quality_score=0.8))
  executor = _executor(quality_evaluator=lambda _m, _s: 0.8)
  result = executor.execute(request)
  assert result.validation.status is ArtifactValidationStatus.PASSED


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, True])
def test_quality_invalid_score(bad_score: Any) -> None:
  request = _request(policy=_policy(minimum_quality_score=0.5))
  executor = _executor(quality_evaluator=lambda _m, _s: bad_score)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == MessageSequenceArtifactExecutionReason.QUALITY_VALIDATION_FAILED.value


def test_deterministic_payload_and_messages() -> None:
  request = _request()
  executor = _executor()
  result_a = executor.execute(request)
  result_b = executor.execute(request)
  assert result_a.payload == result_b.payload
  assert result_a.artifact_content_hash == result_b.artifact_content_hash

  call_a = executor._test_preflight_calls[0]  # type: ignore[attr-defined]
  call_b = executor._test_preflight_calls[1]  # type: ignore[attr-defined]
  assert call_a.messages[0].entry_id == call_b.messages[0].entry_id
  assert call_a.messages[1].content == call_b.messages[1].content


def test_constructor_does_not_invoke_providers() -> None:
  clock = MagicMock()
  op_factory = MagicMock()
  receipt_factory = MagicMock()
  evaluator = MagicMock()
  MessageSequenceArtifactExecutor(
    preflight=lambda _: None,
    invoke_model=lambda _: LLMAdapterResponse(content="x"),
    count_tokens=lambda _: 1,
    quality_evaluator=evaluator,
    clock=clock,
    operation_id_factory=op_factory,
    receipt_id_factory=receipt_factory,
  )
  clock.assert_not_called()
  op_factory.assert_not_called()
  receipt_factory.assert_not_called()
  evaluator.assert_not_called()


def test_import_boundary() -> None:
  module_path = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "message_sequence_artifact.py"
  )
  source = module_path.read_text(encoding="utf-8")
  tree = ast.parse(source)
  forbidden = {
    "intergrax.runtime.nexus",
    "InMemoryOptimizationArtifactRepository",
    "OptimizationArtifactRepository",
    "ContextPlanner",
    "DefaultNexusContextEngine",
  }
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        assert alias.name not in forbidden
    if isinstance(node, ast.ImportFrom):
      module = node.module or ""
      assert "intergrax.runtime.nexus" not in module
      for alias in node.names:
        assert alias.name not in forbidden

  assert "ArtifactCreationCoordinationResult" in source


# --- Source identity tests ---


def test_valid_source_group_proof_executes() -> None:
  request = _request()
  executor = _executor()
  result = executor.execute(request)
  assert result.validation.status is ArtifactValidationStatus.PASSED


def test_changed_source_content_is_rejected_before_providers() -> None:
  messages = _source_messages()
  proofs = _default_group_proofs(messages)
  key = _lookup_key(
    source_refs=tuple(message.message_id for message in messages),
    source_content_hash=_composite_hash(
      tuple(proof.source_content_hash for proof in proofs)
    ),
  )
  changed_messages = (messages[0], _message("msg-2", 1, "changed content"))
  request = _request(
    lookup_key=key,
    source_messages=changed_messages,
    source_group_proofs=proofs,
  )
  clock = MagicMock()
  op_factory = MagicMock()
  receipt_factory = MagicMock()
  count_tokens = MagicMock()
  executor = _executor(
    clock=clock,
    operation_id_factory=op_factory,
    receipt_id_factory=receipt_factory,
    count_tokens=count_tokens,
  )
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.SOURCE_CONTENT_IDENTITY_MISMATCH.value
  )
  assert len(executor._test_preflight_calls) == 0  # type: ignore[attr-defined]
  assert len(executor._test_model_calls) == 0  # type: ignore[attr-defined]
  clock.assert_not_called()
  op_factory.assert_not_called()
  receipt_factory.assert_not_called()
  count_tokens.assert_not_called()


def test_wrong_source_group_hash_is_rejected() -> None:
  messages = _source_messages()
  proofs = (
    MessageSequenceArtifactSourceGroupProof(
      source_refs=(messages[0].message_id,),
      source_content_hash="wrong-group-hash",
    ),
    MessageSequenceArtifactSourceGroupProof(
      source_refs=(messages[1].message_id,),
      source_content_hash=_group_hash((messages[1],)),
    ),
  )
  request = _request(source_messages=messages, source_group_proofs=proofs)
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.SOURCE_CONTENT_IDENTITY_MISMATCH.value
  )


def test_wrong_lookup_composite_hash_is_rejected() -> None:
  messages = _source_messages()
  proofs = _default_group_proofs(messages)
  key = _lookup_key(
    source_refs=tuple(message.message_id for message in messages),
    source_content_hash="wrong-composite-hash",
  )
  request = _request(
    lookup_key=key,
    source_messages=messages,
    source_group_proofs=proofs,
  )
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.SOURCE_CONTENT_IDENTITY_MISMATCH.value
  )


def test_changed_group_boundaries_change_source_identity() -> None:
  messages = _source_messages()
  single_group_proofs = (
    MessageSequenceArtifactSourceGroupProof(
      source_refs=tuple(message.message_id for message in messages),
      source_content_hash=_group_hash(messages),
    ),
  )
  split_proofs = _default_group_proofs(messages)
  key = _lookup_key(
    source_refs=tuple(message.message_id for message in messages),
    source_content_hash=_composite_hash(
      tuple(proof.source_content_hash for proof in split_proofs)
    ),
  )
  request = _request(
    lookup_key=key,
    source_messages=messages,
    source_group_proofs=single_group_proofs,
  )
  executor = _executor()
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.SOURCE_CONTENT_IDENTITY_MISMATCH.value
  )


def test_source_group_refs_must_match_source_messages() -> None:
  messages = _source_messages()
  proofs = (
    MessageSequenceArtifactSourceGroupProof(
      source_refs=("msg-2", "msg-1"),
      source_content_hash=_group_hash((messages[1], messages[0])),
    ),
  )
  with pytest.raises(ValueError, match="source_messages"):
    MessageSequenceArtifactExecutionRequest(
      decision=ContextOptimizationDecision.CREATE_ARTIFACT,
      coordination=_coordination(_lookup_key()),
      lookup_key=_lookup_key(),
      policy=_policy(),
      parent_guard=_parent_guard(),
      source_messages=messages,
      source_group_proofs=proofs,
    )


def test_source_ref_cannot_appear_in_multiple_group_proofs() -> None:
  messages = _source_messages()
  proofs = (
    MessageSequenceArtifactSourceGroupProof(
      source_refs=(messages[0].message_id, messages[1].message_id),
      source_content_hash=_group_hash(messages),
    ),
    MessageSequenceArtifactSourceGroupProof(
      source_refs=(messages[1].message_id,),
      source_content_hash=_group_hash((messages[1],)),
    ),
  )
  with pytest.raises(ValueError, match="multiple group proofs"):
    MessageSequenceArtifactExecutionRequest(
      decision=ContextOptimizationDecision.CREATE_ARTIFACT,
      coordination=_coordination(_lookup_key()),
      lookup_key=_lookup_key(),
      policy=_policy(),
      parent_guard=_parent_guard(),
      source_messages=messages,
      source_group_proofs=proofs,
    )


# --- Receipt validation tests ---


def _valid_receipt(**overrides: Any) -> MessageSequenceArtifactExecutionReceipt:
  defaults: dict[str, Any] = {
    "receipt_id": "receipt-1",
    "parent_operation_id": "parent-op-1",
    "internal_operation_id": "internal-op-1",
    "artifact_lookup_key_hash": "lookup-hash",
    "strategy_id": _STRATEGY_ID,
    "strategy_version": _STRATEGY_VERSION,
    "source_content_hash": "source-hash",
    "source_ref_count": 2,
    "input_tokens": 10,
    "output_tokens": 5,
    "target_tokens": 50,
    "created_at": _FIXED_NOW,
  }
  defaults.update(overrides)
  return MessageSequenceArtifactExecutionReceipt(**defaults)


@pytest.mark.parametrize(
  "overrides",
  [
    {"receipt_id": ""},
    {"parent_operation_id": ""},
    {"internal_operation_id": ""},
    {"internal_operation_id": "parent-op-1", "parent_operation_id": "parent-op-1"},
    {"artifact_lookup_key_hash": ""},
    {"strategy_id": ""},
    {"strategy_version": ""},
    {"source_content_hash": ""},
    {"source_ref_count": 0},
    {"source_ref_count": True},
    {"input_tokens": -1},
    {"input_tokens": True},
    {"output_tokens": -1},
    {"output_tokens": True},
    {"target_tokens": 0},
    {"target_tokens": True},
    {"output_tokens": 51, "target_tokens": 50},
    {"created_at": datetime(2026, 8, 2, 12, 0, 0)},
  ],
)
def test_receipt_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
  with pytest.raises(ValueError):
    _valid_receipt(**overrides)


# --- Result consistency tests ---


def _valid_result() -> MessageSequenceArtifactExecutionResult:
  request = _request()
  return _executor().execute(request)


def _result_with_mutation(**mutations: Any) -> MessageSequenceArtifactExecutionResult:
  result = _valid_result()
  receipt = result.receipt
  validation = result.validation
  internal_guard = result.internal_guard

  if "receipt" in mutations:
    receipt = mutations["receipt"]
  if "validation" in mutations:
    validation = mutations["validation"]
  if "internal_guard" in mutations:
    internal_guard = mutations["internal_guard"]

  return MessageSequenceArtifactExecutionResult(
    payload=result.payload,
    media_type=result.media_type,
    encoding=result.encoding,
    artifact_content_hash=result.artifact_content_hash,
    validation=validation,
    receipt=receipt,
    internal_guard=internal_guard,
  )


def test_result_validation_parent_operation_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["parent_operation_id"] = "wrong-parent"
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="parent_operation_id"):
    _result_with_mutation(validation=validation)


def test_result_validation_internal_operation_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["internal_operation_id"] = "wrong-internal"
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="internal_operation_id"):
    _result_with_mutation(validation=validation)


def test_result_validation_lookup_hash_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["artifact_lookup_key_hash"] = "wrong-hash"
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="artifact_lookup_key_hash"):
    _result_with_mutation(validation=validation)


def test_result_validation_strategy_id_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["strategy_id"] = "wrong-strategy"
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="strategy_id"):
    _result_with_mutation(validation=validation)


def test_result_validation_source_ref_count_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["source_ref_count"] = 99
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="source_ref_count"):
    _result_with_mutation(validation=validation)


def test_result_validation_input_tokens_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["input_tokens"] = 99
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="input_tokens"):
    _result_with_mutation(validation=validation)


def test_result_validation_output_tokens_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["output_tokens"] = 99
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="output_tokens"):
    _result_with_mutation(validation=validation)


def test_result_validation_target_tokens_mismatch() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["target_tokens"] = 99
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="target_tokens"):
    _result_with_mutation(validation=validation)


def test_result_validation_timestamp_mismatch() -> None:
  result = _valid_result()
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at + timedelta(seconds=1),
    reason_codes=(),
    safe_metadata=result.validation.safe_metadata,
  )
  with pytest.raises(ValueError, match="validated_at"):
    _result_with_mutation(validation=validation)


def test_result_internal_guard_missing_active_lookup_hash() -> None:
  result = _valid_result()
  guard = OptimizationExecutionGuard(
    execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
    operation_id=result.internal_guard.operation_id,
    parent_operation_id=result.internal_guard.parent_operation_id,
    optimization_depth=1,
    active_artifact_lookup_key_hashes=(),
    active_strategy_ids=result.internal_guard.active_strategy_ids,
  )
  with pytest.raises(ValueError, match="artifact_lookup_key_hash"):
    _result_with_mutation(internal_guard=guard)


def test_result_internal_guard_missing_active_strategy_id() -> None:
  result = _valid_result()
  guard = OptimizationExecutionGuard(
    execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
    operation_id=result.internal_guard.operation_id,
    parent_operation_id=result.internal_guard.parent_operation_id,
    optimization_depth=1,
    active_artifact_lookup_key_hashes=result.internal_guard.active_artifact_lookup_key_hashes,
    active_strategy_ids=(),
  )
  with pytest.raises(ValueError, match="strategy_id"):
    _result_with_mutation(internal_guard=guard)


def test_result_unexpected_validation_metadata_key() -> None:
  result = _valid_result()
  metadata = dict(result.validation.safe_metadata)
  metadata["unexpected_key"] = "value"
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="required set"):
    _result_with_mutation(validation=validation)


def test_result_missing_validation_metadata_key() -> None:
  result = _valid_result()
  metadata = {
    key: value
    for key, value in result.validation.safe_metadata.items()
    if key != "target_tokens"
  }
  validation = ArtifactValidationSummary(
    status=ArtifactValidationStatus.PASSED,
    validation_contract_version=result.validation.validation_contract_version,
    validated_at=result.receipt.created_at,
    reason_codes=(),
    safe_metadata=metadata,
  )
  with pytest.raises(ValueError, match="required set"):
    _result_with_mutation(validation=validation)


# --- Callback normalization tests ---


def test_preflight_callback_error_normalized() -> None:
  def preflight(_: InternalMessageSequenceModelCall) -> None:
    raise MessageSequenceArtifactExecutionError(
      MessageSequenceArtifactExecutionReason.QUALITY_VALIDATION_FAILED.value
    )

  executor = _executor(preflight=preflight)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.INTERNAL_PREFLIGHT_FAILED.value
  )


def test_model_callback_error_normalized() -> None:
  def invoke(_: InternalMessageSequenceModelCall) -> LLMAdapterResponse:
    raise MessageSequenceArtifactExecutionError(
      MessageSequenceArtifactExecutionReason.SOURCE_SEQUENCE_MISMATCH.value
    )

  executor = _executor(invoke_model=invoke)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.INTERNAL_MODEL_CALL_FAILED.value
  )


def test_quality_evaluator_callback_error_normalized() -> None:
  def evaluator(_m: Any, _s: str) -> float:
    raise MessageSequenceArtifactExecutionError(
      MessageSequenceArtifactExecutionReason.INTERNAL_PREFLIGHT_FAILED.value
    )

  request = _request(policy=_policy(minimum_quality_score=0.5))
  executor = _executor(quality_evaluator=evaluator)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(request)
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.QUALITY_VALIDATION_FAILED.value
  )


@pytest.mark.parametrize(
  "counter",
  [
    lambda _: (_ for _ in ()).throw(RuntimeError("tokenizer failed")),
    lambda _: True,
    lambda _: False,
    lambda _: 1.5,
    lambda _: -1,
  ],
)
def test_token_counter_invalid_outputs_normalized(counter: Any) -> None:
  executor = _executor(count_tokens=counter)
  with pytest.raises(MessageSequenceArtifactExecutionError) as exc_info:
    executor.execute(_request())
  assert exc_info.value.reason == (
    MessageSequenceArtifactExecutionReason.TOKEN_COUNT_FAILED.value
  )
