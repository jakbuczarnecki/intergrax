# © Artur Czarnecki. All rights reserved.

"""Attempt lifecycle domain service (P0C-4)."""

from __future__ import annotations

from intergrax.contracts.attempt_lifecycle import (
    AttemptLifecycleError,
    AttemptLifecycleState,
    AttemptLifecycleStore,
    AttemptTransitionReason,
    AttemptTransitionResult,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    mint_attempt_id,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.execution.attempt_lifecycle.persistence import (
    decode_attempt_lifecycle_state,
    encode_attempt_lifecycle_state,
)


class AttemptLifecycleService:
    """
    Canonical authority for durable attempt transitions.

    Retry policy decides whether to retry; this service creates the next canonical
    Attempt; active execution identity propagation binds it in-process.
    """

    __slots__ = ("_store",)

    def __init__(self, store: AttemptLifecycleStore) -> None:
        self._store = store

    @property
    def store(self) -> AttemptLifecycleStore:
        return self._store

    def record_initial_attempt(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        attempt_id: AttemptId,
    ) -> AttemptLifecycleState:
        validated_run_id = validate_run_id(run_id)
        validated_attempt_id = validate_attempt_id(attempt_id)
        initial = AttemptLifecycleState(
            run_id=validated_run_id,
            active_attempt_id=validated_attempt_id,
            previous_attempt_id=None,
            generation=1,
            transition_reason=AttemptTransitionReason.INITIAL,
        )
        if self._store.compare_and_swap(
            tenant_id=tenant_id,
            run_id=validated_run_id,
            expected=None,
            new_state=initial,
        ):
            return initial
        existing_raw = self._store.load_raw(tenant_id=tenant_id, run_id=validated_run_id)
        if existing_raw is None:
            raise AttemptLifecycleError("attempt lifecycle initial record race lost")
        existing = self._load_state(existing_raw)
        if existing.active_attempt_id != validated_attempt_id:
            raise StaleClaimError(
                "attempt lifecycle initial record already exists with different active attempt",
            )
        return existing

    def transition_to_next_attempt(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected_attempt_id: AttemptId,
        reason: AttemptTransitionReason,
    ) -> AttemptTransitionResult:
        if reason is not AttemptTransitionReason.RETRY:
            raise ValueError("transition_to_next_attempt supports RETRY reason only")
        validated_run_id = validate_run_id(run_id)
        validated_expected = validate_attempt_id(expected_attempt_id)
        new_attempt_id = mint_attempt_id()

        for _ in range(8):
            current_raw = self._load_raw_or_raise(tenant_id=tenant_id, run_id=validated_run_id)
            if current_raw is None:
                next_state = AttemptLifecycleState(
                    run_id=validated_run_id,
                    active_attempt_id=new_attempt_id,
                    previous_attempt_id=validated_expected,
                    generation=2,
                    transition_reason=reason,
                )
                if self._compare_and_swap_or_raise(
                    tenant_id=tenant_id,
                    run_id=validated_run_id,
                    expected=None,
                    new_state=next_state,
                ):
                    return AttemptTransitionResult(
                        run_id=validated_run_id,
                        previous_attempt_id=validated_expected,
                        active_attempt_id=new_attempt_id,
                        generation=next_state.generation,
                    )
                continue

            current = self._load_state(current_raw)
            if current.run_id != validated_run_id:
                raise AttemptLifecycleError("attempt lifecycle run_id mismatch")
            if current.active_attempt_id != validated_expected:
                raise StaleClaimError(
                    "attempt lifecycle transition rejected: expected attempt is stale",
                )
            next_state = AttemptLifecycleState(
                run_id=validated_run_id,
                active_attempt_id=new_attempt_id,
                previous_attempt_id=validated_expected,
                generation=current.generation + 1,
                transition_reason=reason,
            )
            if self._compare_and_swap_or_raise(
                tenant_id=tenant_id,
                run_id=validated_run_id,
                expected=current_raw,
                new_state=next_state,
            ):
                return AttemptTransitionResult(
                    run_id=validated_run_id,
                    previous_attempt_id=validated_expected,
                    active_attempt_id=new_attempt_id,
                    generation=next_state.generation,
                )

        raise AttemptLifecycleError("attempt lifecycle transition failed after retries")

    def get_active_attempt_id(self, *, tenant_id: str, run_id: RunId) -> AttemptId | None:
        raw = self._load_raw_or_raise(tenant_id=tenant_id, run_id=validate_run_id(run_id))
        if raw is None:
            return None
        return self._load_state(raw).active_attempt_id

    def _load_raw_or_raise(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        try:
            return self._store.load_raw(tenant_id=tenant_id, run_id=run_id)
        except AttemptLifecycleError:
            raise
        except Exception as exc:
            raise AttemptLifecycleError("attempt lifecycle store load failed") from exc

    def _compare_and_swap_or_raise(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        new_state: AttemptLifecycleState,
    ) -> bool:
        try:
            return self._store.compare_and_swap(
                tenant_id=tenant_id,
                run_id=run_id,
                expected=expected,
                new_state=new_state,
            )
        except AttemptLifecycleError:
            raise
        except Exception as exc:
            raise AttemptLifecycleError("attempt lifecycle store write failed") from exc

    @staticmethod
    def _load_state(raw: bytes) -> AttemptLifecycleState:
        state = decode_attempt_lifecycle_state(raw)
        if encode_attempt_lifecycle_state(state) != raw:
            raise AttemptLifecycleError("attempt lifecycle record failed canonical round-trip")
        return state
