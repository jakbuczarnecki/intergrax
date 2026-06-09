# © Artur Czarnecki. All rights reserved.

"""Scaling provisioner backends (ECP-4 / ECP-5)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.capacity.contracts import ScalingAction, ScalingActionKind


class KubernetesScaler(Protocol):
  def scale_workload(self, *, deployment: str, replicas: int) -> int: ...
  def get_replicas(self, *, deployment: str) -> int: ...


class ScalingProvisioner:
    """Apply scaling actions to configured backends."""

    def __init__(
        self,
        *,
        kubernetes: KubernetesScaler | None = None,
    ) -> None:
        self._kubernetes = kubernetes
        self.applied: list[ScalingAction] = []
        self.failures: list[str] = []

    def apply(self, action: ScalingAction, *, deployment: str = "nexus-host") -> bool:
        try:
            if action.kind is ScalingActionKind.SCALE_K8S_DEPLOYMENT:
                if self._kubernetes is None:
                    self.failures.append("kubernetes backend not configured")
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
            return True
        except Exception as exc:  # noqa: BLE001 — provisioner records failure
            self.failures.append(str(exc))
            return False
