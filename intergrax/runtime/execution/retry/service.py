# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution-attempt retry orchestration (NPSC-5E/R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError, AttemptTransitionReason
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    peek_active_execution_identity,
    rebind_active_attempt_for_retry,
)
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.contracts.execution_retry import (
    BackoffPolicyConfig,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
    ExecutionRetryEligibilityResult,
)
from intergrax.contracts.resilience_policy import ResiliencePolicy
from intergrax.runtime.execution.attempt_lifecycle.durability_policy import (
    DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG,
)
from intergrax.runtime.execution.attempt_lifecycle.service import AttemptLifecycleService
from intergrax.runtime.execution.retry.backoff import (
    backoff_config_from_resilience_policy,
    compute_backoff_delay,
)
from intergrax.runtime.execution.retry.policy import evaluate_execution_retry_eligibility


@dataclass(frozen=True, slots=True)
class ExecutionAttemptRetryTransitionResult:
    """Successful canonical attempt retry transition."""

    run_id: RunId
    previous_attempt_id: AttemptId
    active_attempt_id: AttemptId
    generation: int
    eligibility: ExecutionRetryEligibilityResult


class ExecutionAttemptRetryService:
    """
    Canonical execution-attempt retry authority seam.

    Policy answers MAY retry; AttemptLifecycleService performs attempt transition.
    """

    __slots__ = ("_attempt_lifecycle", "_lineage_persistence", "_production_mode")

    def __init__(
        self,
        attempt_lifecycle: AttemptLifecycleService,
        *,
        lineage_persistence: ExecutionLineagePersistence | None = None,
        production_mode: bool = False,
    ) -> None:
        self._attempt_lifecycle = attempt_lifecycle
        self._lineage_persistence = lineage_persistence
        self._production_mode = production_mode

    def evaluate_eligibility(
        self,
        request: ExecutionRetryEligibilityRequest,
    ) -> ExecutionRetryEligibilityResult:
        return evaluate_execution_retry_eligibility(request)

    def compute_backoff(
        self,
        *,
        attempt_number: int,
        resilience_policy: ResiliencePolicy | None = None,
        config: BackoffPolicyConfig | None = None,
        retry_after_seconds: float | None = None,
    ) -> float:
        resolved = config or (
            backoff_config_from_resilience_policy(resilience_policy)
            if resilience_policy is not None
            else BackoffPolicyConfig()
        )
        return compute_backoff_delay(
            attempt_number=attempt_number,
            config=resolved,
            retry_after_seconds=retry_after_seconds,
        )

    def transition_for_retry(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        expected_attempt_id: AttemptId,
        request: ExecutionRetryEligibilityRequest,
    ) -> ExecutionAttemptRetryTransitionResult | None:
        eligibility = self.evaluate_eligibility(request)
        if eligibility.action is not ExecutionRetryAction.RETRY:
            return None

        if self._production_mode:
            self._attempt_lifecycle.require_durable()

        try:
            transition = self._attempt_lifecycle.transition_to_next_attempt(
                tenant_id=tenant_id,
                run_id=run_id,
                expected_attempt_id=expected_attempt_id,
                reason=AttemptTransitionReason.RETRY,
            )
        except AttemptLifecycleError as exc:
            if str(exc) == DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG:
                raise
            return None
        except Exception:
            return None

        if self._lineage_persistence is not None:
            from intergrax.contracts.execution_lineage import ExecutionLineageAttemptClosureKind
            from intergrax.runtime.execution.lineage.seal import seal_lineage_attempt

            seal_lineage_attempt(
                self._lineage_persistence,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=expected_attempt_id,
                closure_kind=ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED,
            )

        if peek_active_execution_identity() is not None:
            rebound = rebind_active_attempt_for_retry(
                run_id=transition.run_id,
                attempt_id=transition.active_attempt_id,
            )
            if rebound != transition.active_attempt_id:
                return None

        return ExecutionAttemptRetryTransitionResult(
            run_id=transition.run_id,
            previous_attempt_id=transition.previous_attempt_id,
            active_attempt_id=transition.active_attempt_id,
            generation=transition.generation,
            eligibility=eligibility,
        )
