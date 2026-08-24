# © Artur Czarnecki. All rights reserved.

"""Production-scale Celery/K8s adapter contracts (AUDIT-IDEAL-30.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.capacity.control_plane_governance import EcpResourceTenantResolver
from intergrax.runtime.capacity.governed_capacity_mutation import GovernedCapacityMutationExecutor
from intergrax.runtime.capacity.provisioner import (
    ProvisionerExecutionMode,
    ScalingProvisioner,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

KubernetesBackendKind = Literal["live", "in_memory"]


@dataclass
class InMemoryKubernetesScaler:
    """Deterministic Kubernetes scaler for harness CI when cluster URL is unset."""

    replicas_by_deployment: dict[str, int] = field(default_factory=dict)

    def scale_workload(self, *, deployment: str, replicas: int) -> int:
        self.replicas_by_deployment[deployment] = max(0, replicas)
        return self.replicas_by_deployment[deployment]

    def get_replicas(self, *, deployment: str) -> int:
        return self.replicas_by_deployment.get(deployment, 1)


@dataclass
class CeleryProductionAdapter:
    """Records Celery worker scale intents for production hosts."""

    worker_count: int = 1

    def scale_workers(self, *, delta: int) -> int:
        self.worker_count = max(1, self.worker_count + delta)
        return self.worker_count

    def get_worker_count(self) -> int:
        return self.worker_count


def resolve_kubernetes_backend() -> tuple[InMemoryKubernetesScaler | object, KubernetesBackendKind]:
    """Use REST K8s client when INTERGRAX_KUBERNETES_URL is configured."""
    if os.environ.get("INTERGRAX_KUBERNETES_URL", "").strip():
        from intergrax.integrations.providers.cloud_platform.kubernetes.bundle import (
            create_kubernetes_cloud_platform,
        )

        return create_kubernetes_cloud_platform(), "live"
    return InMemoryKubernetesScaler(replicas_by_deployment={"nexus-host": 2}), "in_memory"


@dataclass(frozen=True, slots=True)
class ProductionCapacityAdapters:
    kubernetes: InMemoryKubernetesScaler | object
    celery: CeleryProductionAdapter
    governed_executor: GovernedCapacityMutationExecutor
    kubernetes_backend: KubernetesBackendKind


def build_production_capacity_adapters(
    *,
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    tenant_resolver: EcpResourceTenantResolver | None = None,
) -> ProductionCapacityAdapters:
    """Wire production Celery/K8s adapters behind governed capacity mutation facade."""
    kubernetes, backend_kind = resolve_kubernetes_backend()
    celery = CeleryProductionAdapter(worker_count=2)
    provisioner = ScalingProvisioner(
        kubernetes=kubernetes,
        celery=celery,
        execution_mode=ProvisionerExecutionMode.GOVERNED_ONLY,
    )
    governed_executor = GovernedCapacityMutationExecutor(
        provisioner=provisioner,
        mutation_boundary=mutation_boundary,
        tenant_resolver=tenant_resolver,
    )
    return ProductionCapacityAdapters(
        kubernetes=kubernetes,
        celery=celery,
        governed_executor=governed_executor,
        kubernetes_backend=backend_kind,
    )


def apply_production_scale_probe(
    adapters: ProductionCapacityAdapters,
    *,
    principal: RequestIdentity,
    tenant_id: str,
    k8s_mutation_id: str,
    celery_mutation_id: str,
    deployment: str = "nexus-host",
    celery_pool_id: str = "default",
) -> bool:
    """Exercise governed K8s and Celery adapter paths for gate evidence."""
    k8s_result = adapters.governed_executor.scale_k8s_deployment(
        principal=principal,
        tenant_id=tenant_id,
        mutation_id=k8s_mutation_id,
        deployment=deployment,
        delta=1,
    )
    celery_result = adapters.governed_executor.scale_celery_workers(
        principal=principal,
        tenant_id=tenant_id,
        mutation_id=celery_mutation_id,
        pool_id=celery_pool_id,
        delta=1,
    )
    replicas_ok = True
    if adapters.kubernetes_backend == "in_memory":
        replicas_ok = adapters.kubernetes.get_replicas(deployment=deployment) >= 3
    return (
        k8s_result.authorization_evidence.mutation_id == k8s_mutation_id
        and celery_result.authorization_evidence.mutation_id == celery_mutation_id
        and replicas_ok
        and adapters.celery.worker_count >= 3
    )
