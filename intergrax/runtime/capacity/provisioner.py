# © Artur Czarnecki. All rights reserved.

"""Scaling provisioner backends (ECP-4 / ECP-5)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.capacity.action_gate import CapacityActionGate
from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind
from intergrax.runtime.capacity.events import PublishFn, publish_scale_applied, publish_scale_failed
from intergrax.runtime.capacity.metrics import record_scale_action


class KubernetesScaler(Protocol):
  def scale_workload(self, *, deployment: str, replicas: int) -> int: ...
  def get_replicas(self, *, deployment: str) -> int: ...


class ScalingProvisioner:
    """Apply scaling actions to configured backends."""

    def __init__(
        self,
        *,
        kubernetes: KubernetesScaler | None = None,
        action_gate: CapacityActionGate | None = None,
        publish: PublishFn | None = None,
    ) -> None:
        self._kubernetes = kubernetes
        self._action_gate = action_gate or CapacityActionGate()
        self._publish = publish
        self.applied: list[ScalingAction] = []
        self.failures: list[str] = []

    def apply(self, action: ScalingAction, *, deployment: str = "nexus-host") -> bool:
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
                # ECP-5.1 stub — lab documents broker autoscale separately
                pass
            elif action.kind is ScalingActionKind.RAISE_ORCHESTRATION_CEILING:
                pass
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
