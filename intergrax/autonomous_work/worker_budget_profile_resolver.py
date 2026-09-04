# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Budget profile resolution adapters for worker accounting (AW-5B)."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.autonomous_work.worker_budget_ports import WorkerBudgetProfileResolutionError
from intergrax.contracts.autonomous_work.profile_reference import BudgetProfileRef
from intergrax.contracts.autonomous_work.worker_budget_accounting import WorkerBudgetPolicy


class MappingWorkerBudgetProfileResolver:
    """Resolve ``BudgetProfileRef`` from an explicit in-memory mapping."""

    def __init__(
        self,
        policies: Mapping[tuple[str, int], WorkerBudgetPolicy],
        *,
        default_policy: WorkerBudgetPolicy | None = None,
    ) -> None:
        self._policies = dict(policies)
        self._default_policy = default_policy

    def resolve(self, profile_ref: BudgetProfileRef) -> WorkerBudgetPolicy:
        key = (profile_ref.profile_id, profile_ref.version.value)
        policy = self._policies.get(key)
        if policy is not None:
            return policy
        if self._default_policy is not None:
            return self._default_policy
        raise WorkerBudgetProfileResolutionError(
            f"budget profile unavailable: {profile_ref.profile_id}@{profile_ref.version.value}"
        )


class StaticWorkerBudgetProfileResolver:
    """Always return one configured policy — explicit platform default."""

    def __init__(self, policy: WorkerBudgetPolicy) -> None:
        self._policy = policy

    def resolve(self, profile_ref: BudgetProfileRef) -> WorkerBudgetPolicy:
        del profile_ref
        return self._policy
