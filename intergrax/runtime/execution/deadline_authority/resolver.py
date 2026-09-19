# © Artur Czarnecki. All rights reserved.

"""Root durable deadline authority resolution (create-once / resume load)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution_deadline.persistence_port import (
    ExecutionDeadlineAuthorityPersistencePort,
    ExecutionDeadlinePersistenceError,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.contracts.execution_deadline.snapshot import (
    EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION,
    ExecutionDeadlineAuthoritySnapshot,
)
from intergrax.contracts.execution_identity import RunId
from intergrax.runtime.execution.deadline_authority.codec import (
    decode_execution_deadline_authority_snapshot,
    encode_execution_deadline_authority_snapshot,
)
from intergrax.runtime.execution.deadline_authority.persistence import ExecutionDeadlineCodecError
from intergrax.runtime.execution.deadline_authority.projection import project_execution_deadline
from intergrax.contracts.execution_deadline.clock import MonotonicClockPort, UtcClockPort
from intergrax.runtime.execution.deadline_authority.system_clocks import (
    SystemMonotonicClock,
    SystemUtcClock,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget


@dataclass(frozen=True, slots=True)
class RootDeadlineResolution:
    snapshot: ExecutionDeadlineAuthoritySnapshot
    projection: ExecutionDeadlineProjection


class ExecutionDeadlineAuthorityResolver:
    def __init__(
        self,
        persistence: ExecutionDeadlineAuthorityPersistencePort,
        *,
        utc_clock: UtcClockPort | None = None,
        monotonic_clock: MonotonicClockPort | None = None,
    ) -> None:
        self._persistence = persistence
        self._utc_clock = utc_clock if utc_clock is not None else SystemUtcClock()
        self._monotonic_clock = (
            monotonic_clock if monotonic_clock is not None else SystemMonotonicClock()
        )

    @property
    def monotonic_clock(self) -> MonotonicClockPort:
        return self._monotonic_clock

    def resolve_for_root(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        run_budget: RunBudget | None,
        existing_run_materialized: bool,
    ) -> RootDeadlineResolution:
        raw = self._persistence.load(tenant_id=tenant_id, run_id=run_id)
        if raw is not None:
            snapshot = self._decode_or_fail(raw)
            projection = project_execution_deadline(
                snapshot,
                utc_clock=self._utc_clock,
                monotonic_clock=self._monotonic_clock,
            )
            return RootDeadlineResolution(snapshot=snapshot, projection=projection)

        policy_max = (
            run_budget.max_wall_time_seconds if run_budget is not None else None
        )
        if existing_run_materialized and policy_max is not None:
            raise ExecutionDeadlinePersistenceError(
                "missing durable execution deadline authority for existing run with wall limit",
            )

        now_utc = self._utc_clock.now_utc()
        deadline_at_utc: datetime | None
        if policy_max is None:
            deadline_at_utc = None
        else:
            deadline_at_utc = now_utc + timedelta(seconds=policy_max)
        snapshot = ExecutionDeadlineAuthoritySnapshot(
            schema_version=EXECUTION_DEADLINE_AUTHORITY_SCHEMA_VERSION,
            run_id=run_id,
            deadline_at_utc=deadline_at_utc,
            authority_created_at_utc=now_utc,
            policy_max_wall_time_seconds=policy_max,
        )
        encoded = encode_execution_deadline_authority_snapshot(snapshot)
        if not self._persistence.compare_and_create(
            tenant_id=tenant_id,
            run_id=run_id,
            expected=None,
            encoded_snapshot=encoded,
        ):
            return self.resolve_for_root(
                tenant_id=tenant_id,
                run_id=run_id,
                run_budget=run_budget,
                existing_run_materialized=True,
            )
        projection = project_execution_deadline(
            snapshot,
            utc_clock=self._utc_clock,
            monotonic_clock=self._monotonic_clock,
        )
        return RootDeadlineResolution(snapshot=snapshot, projection=projection)

    def _decode_or_fail(self, raw: bytes) -> ExecutionDeadlineAuthoritySnapshot:
        try:
            return decode_execution_deadline_authority_snapshot(raw)
        except ExecutionDeadlineCodecError as exc:
            raise ExecutionDeadlinePersistenceError(str(exc)) from exc
