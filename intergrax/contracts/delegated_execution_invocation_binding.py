# © Artur Czarnecki. All rights reserved.

"""Platform-owned delegated invocation correlation (P2.1-S2B-C1).

``ProviderInvocation`` is provider-neutral and does not carry canonical
``ExecutionId``. This binding proves invocation evidence belongs to an
admitted child execution via the platform ``request_digest`` (minted on the
S2A dispatch path), not caller-supplied pairings.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.delegated_execution_provider import (
    DelegatedExecutionBudgetMode,
    DelegatedExecutionBudgetProjection,
    DelegatedExecutionContext,
    DelegatedExecutionContractError,
    DelegatedExecutionOperationMetadata,
    DelegatedExecutionOutcome,
    DelegatedExecutionOutcomeCategory,
    assert_provider_native_ids_distinct_from_execution,
    delegated_failure_outcome,
    digest_delegated_execution_request,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.provider_invocation import ProviderInvocation

ResultT = TypeVar("ResultT")

SCHEMA_DELEGATED_EXECUTION_INVOCATION_BINDING_V1: Final = (
    "delegated_execution_invocation_binding.v1"
)
_NON_EMPTY = Field(min_length=1)


class DelegatedExecutionInvocationBinding(BaseModel):
    """Immutable proof that a provider invocation belongs to a child execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_execution_invocation_binding.v1"] = (
        SCHEMA_DELEGATED_EXECUTION_INVOCATION_BINDING_V1
    )
    execution_id: ExecutionId
    parent_execution_id: ExecutionId
    run_id: RunId
    attempt_id: AttemptId
    provider_id: str = _NON_EMPTY
    operation: DelegatedExecutionOperationMetadata
    payload_digest: str = _NON_EMPTY
    provider_invocation: ProviderInvocation

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("parent_execution_id", mode="before")
    @classmethod
    def _validate_parent_execution_id(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("provider_id", "payload_digest")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _prove_platform_correlation(self) -> DelegatedExecutionInvocationBinding:
        inv = self.provider_invocation
        if str(self.run_id) != inv.run_id:
            raise ValueError("provider_invocation.run_id must match binding run_id")
        if inv.provider_id != self.provider_id:
            raise ValueError(
                "provider_invocation.provider_id must match binding provider_id",
            )
        if inv.operation != self.operation.operation:
            raise ValueError(
                "provider_invocation.operation must match binding operation",
            )
        if inv.task_id != self.operation.task_id:
            raise ValueError("provider_invocation.task_id must match binding task_id")
        assert_provider_native_ids_distinct_from_execution(
            execution_id=self.execution_id,
            provider_request_id=inv.provider_request_id,
            provider_operation_id=inv.provider_operation_id,
            invocation_id=inv.invocation_id,
        )
        digest_context = DelegatedExecutionContext(
            execution_id=self.execution_id,
            parent_execution_id=self.parent_execution_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            authority=ParentExecutionAuthority.scoped(("delegated.binding.verify",)),
            budget=DelegatedExecutionBudgetProjection(
                allocation_mode=DelegatedExecutionBudgetMode.SHARED,
            ),
        )
        expected_digest = digest_delegated_execution_request(
            context=digest_context,
            operation=self.operation,
            payload_digest=self.payload_digest,
        )
        if inv.request_digest != expected_digest:
            raise ValueError(
                "provider_invocation.request_digest does not prove binding execution_id",
            )
        return self


def mint_delegated_execution_invocation_binding(
    *,
    context: DelegatedExecutionContext,
    operation: DelegatedExecutionOperationMetadata,
    payload_digest: str,
    provider_invocation: ProviderInvocation,
) -> DelegatedExecutionInvocationBinding:
    """Create a validated binding using admitted child context from the S2A path."""
    if not payload_digest or not payload_digest.strip():
        raise DelegatedExecutionContractError("payload_digest must be non-empty")
    return DelegatedExecutionInvocationBinding(
        execution_id=context.execution_id,
        parent_execution_id=context.parent_execution_id,
        run_id=context.run_id,
        attempt_id=context.attempt_id,
        provider_id=provider_invocation.provider_id,
        operation=operation,
        payload_digest=payload_digest.strip(),
        provider_invocation=provider_invocation,
    )


def delegated_provider_outcome_contract_mismatch_failure(
    *,
    failure_message: str,
) -> DelegatedExecutionOutcome[ResultT]:
    """Platform-owned failure with no unvalidated provider correlation evidence."""
    return delegated_failure_outcome(
        category=DelegatedExecutionOutcomeCategory.PROVIDER_FAILURE,
        failure_code="OUTCOME_CONTRACT_MISMATCH",
        failure_message=failure_message,
        provider_invocation=None,
        provider_outcome=None,
    )


def assert_provider_outcome_has_no_invocation_binding(
    outcome: DelegatedExecutionOutcome[ResultT],
) -> None:
    """Reject provider-supplied platform-owned invocation correlation."""
    if outcome.invocation_binding is not None:
        raise DelegatedExecutionContractError(
            "provider outcome must not carry invocation_binding; "
            "platform-owned enrichment only",
        )


def delegated_invocation_correlation_persistence_failure(
    *,
    provider_invocation: ProviderInvocation,
    provider_outcome: object,
    failure_message: str,
) -> DelegatedExecutionOutcome[ResultT]:
    """Fail closed when durable correlation cannot be stored after provider dispatch."""
    from intergrax.contracts.provider_invocation import ProviderInvocationOutcome

    if not isinstance(provider_outcome, ProviderInvocationOutcome):
        raise DelegatedExecutionContractError(
            "provider_outcome required for correlation persistence failure",
        )
    return delegated_failure_outcome(
        category=DelegatedExecutionOutcomeCategory.PLATFORM_FAILURE,
        failure_code="INVOCATION_CORRELATION_PERSISTENCE_FAILURE",
        failure_message=failure_message,
        provider_invocation=provider_invocation,
        provider_outcome=provider_outcome,
    )


def enrich_delegated_outcome_with_platform_invocation_binding(
    *,
    outcome: DelegatedExecutionOutcome[ResultT],
    context: DelegatedExecutionContext,
    operation: DelegatedExecutionOperationMetadata,
    payload_digest: str,
) -> DelegatedExecutionOutcome[ResultT]:
    """Attach platform-issued binding on the Execution-owned S2A dispatch path."""
    assert_provider_outcome_has_no_invocation_binding(outcome)
    if outcome.provider_invocation is None:
        return outcome
    binding = mint_delegated_execution_invocation_binding(
        context=context,
        operation=operation,
        payload_digest=payload_digest,
        provider_invocation=outcome.provider_invocation,
    )
    return replace(outcome, invocation_binding=binding)


__all__ = [
    "DelegatedExecutionInvocationBinding",
    "assert_provider_outcome_has_no_invocation_binding",
    "delegated_invocation_correlation_persistence_failure",
    "delegated_provider_outcome_contract_mismatch_failure",
    "enrich_delegated_outcome_with_platform_invocation_binding",
    "mint_delegated_execution_invocation_binding",
]
