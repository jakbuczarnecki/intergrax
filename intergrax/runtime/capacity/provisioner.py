# © Artur Czarnecki. All rights reserved.

"""Scaling provisioner backends (ECP-4 / ECP-5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from intergrax.runtime.capacity.action_gate import CapacityActionGate
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind
from intergrax.runtime.capacity.events import PublishFn, publish_scale_applied, publish_scale_failed
from intergrax.runtime.capacity.metrics import record_scale_action


class ProvisionerExecutionMode(StrEnum):
    """Execution posture for ScalingProvisioner side effects."""

    UNRESTRICTED = "unrestricted"
    GOVERNED_ONLY = "governed_only"


class KubernetesScaler(Protocol):
    def scale_workload(self, *, deployment: str, replicas: int) -> int: ...

    def get_replicas(self, *, deployment: str) -> int: ...


class CeleryScaler(Protocol):
    def scale_workers(self, *, delta: int) -> int: ...

    def get_worker_count(self) -> int: ...


class OrchestrationCeilingPatcher(Protocol):
    def raise_ceiling(self, *, delta: int) -> int: ...


class StaleCapacityStateError(Exception):
    """Authorized current state no longer matches provider state before apply."""

    def __init__(
        self,
        *,
        authorized_current: int,
        observed_current: int,
        deployment: str | None = None,
        pool_id: str | None = None,
    ) -> None:
        resource = deployment or pool_id or "unknown"
        message = (
            f"stale capacity state for {resource}: "
            f"authorized={authorized_current} observed={observed_current}"
        )
        super().__init__(message)
        self.authorized_current = authorized_current
        self.observed_current = observed_current
        self.deployment = deployment
        self.pool_id = pool_id


class GovernedExecutionRequiredError(Exception):
    """Production capacity backends require governed facade execution."""


class ScalingProvisioner:
    """Apply scaling actions to configured backends."""

    def __init__(
        self,
        *,
        kubernetes: KubernetesScaler | None = None,
        celery: CeleryScaler | None = None,
        ceiling_patcher: OrchestrationCeilingPatcher | None = None,
        action_gate: CapacityActionGate | None = None,
        publish: PublishFn | None = None,
        execution_mode: ProvisionerExecutionMode = ProvisionerExecutionMode.UNRESTRICTED,
    ) -> None:
        self._kubernetes = kubernetes
        self._celery = celery
        self._ceiling_patcher = ceiling_patcher
        self._action_gate = action_gate or CapacityActionGate()
        self._publish = publish
        self._execution_mode = execution_mode
        self.applied: list[ScalingAction] = []
        self.failures: list[str] = []

    def read_k8s_replicas(self, *, deployment: str) -> int:
        if self._kubernetes is None:
            raise RuntimeError("kubernetes backend not configured")
        return self._kubernetes.get_replicas(deployment=deployment)

    def read_celery_worker_count(self) -> int:
        if self._celery is None:
            raise RuntimeError("celery backend not configured")
        return self._celery.get_worker_count()

    def _apply_authorized_k8s_target(
        self,
        *,
        deployment: str,
        replicas: int,
        authorized_current: int,
    ) -> int:
        if self._kubernetes is None:
            raise RuntimeError("kubernetes backend not configured")
        observed_current = self._kubernetes.get_replicas(deployment=deployment)
        if observed_current != authorized_current:
            raise StaleCapacityStateError(
                authorized_current=authorized_current,
                observed_current=observed_current,
                deployment=deployment,
            )
        return self._kubernetes.scale_workload(deployment=deployment, replicas=replicas)

    def _apply_authorized_celery_target(
        self,
        *,
        target_workers: int,
        authorized_current: int,
    ) -> int:
        if self._celery is None:
            raise RuntimeError("celery backend not configured")
        observed_current = self._celery.get_worker_count()
        if observed_current != authorized_current:
            raise StaleCapacityStateError(
                authorized_current=authorized_current,
                observed_current=observed_current,
                pool_id="default",
            )
        delta = target_workers - authorized_current
        return self._celery.scale_workers(delta=delta)

    def apply(self, action: ScalingAction, *, deployment: str = "nexus-host") -> bool:
        if self._requires_governed_execution(action):
            reason = "governed_execution_required_for_production_capacity"
            self.failures.append(reason)
            if self._publish is not None:
                publish_scale_failed(self._publish, action, reason=reason)
            raise GovernedExecutionRequiredError(reason)
        if not self._action_gate.authorize(action):
            reason = "capacity_action_denied_by_policy"
            self.failures.append(reason)
            if self._publish is not None:
                publish_scale_failed(self._publish, action, reason=reason)
            return False
        try:
            if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
                if self._kubernetes is None:
                    reason = "kubernetes backend not configured"
                    self.failures.append(reason)
                    if self._publish is not None:
                        publish_scale_failed(self._publish, action, reason=reason)
                    return False
                current = self._kubernetes.get_replicas(deployment=deployment)
                self._kubernetes.scale_workload(
                    deployment=deployment,
                    replicas=max(0, current + action.delta),
                )
            elif action.kind is ScalingActionKind.SCALE_CELERY_WORKERS:
                if self._celery is None:
                    reason = "celery backend not configured"
                    self.failures.append(reason)
                    if self._publish is not None:
                        publish_scale_failed(self._publish, action, reason=reason)
                    return False
                self._celery.scale_workers(delta=action.delta)
            elif action.kind is ScalingActionKind.RAISE_ORCHESTRATION_CEILING:
                if self._ceiling_patcher is None:
                    reason = "orchestration ceiling patcher not configured"
                    self.failures.append(reason)
                    if self._publish is not None:
                        publish_scale_failed(self._publish, action, reason=reason)
                    return False
                self._ceiling_patcher.raise_ceiling(delta=action.delta)
            elif action.kind is ScalingActionKind.REQUEST_HITL:
                return False
            self.applied.append(action)
            record_scale_action(target=action.target.value)
            if self._publish is not None:
                publish_scale_applied(self._publish, action)
            return True
        except Exception as exc:  # noqa: BLE001 — provisioner records failure
            reason = str(exc)
            self.failures.append(reason)
            if self._publish is not None:
                publish_scale_failed(self._publish, action, reason=reason)
            return False

    def _requires_governed_execution(self, action: ScalingAction) -> bool:
        if self._execution_mode is not ProvisionerExecutionMode.GOVERNED_ONLY:
            return False
        return action.kind in (
            ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            ScalingActionKind.SCALE_CELERY_WORKERS,
        )
