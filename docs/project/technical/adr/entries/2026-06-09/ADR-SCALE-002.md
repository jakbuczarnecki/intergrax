# ADR-SCALE-002: Ingress controller vs nginx integration slug (ECP-6.1)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-09 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md) · ECP-6.1 |

## Context

ECP scaling actions target Kubernetes deployments and Celery pools. Product docs referenced a future `nginx` vs `ingress_controller` catalog slug for edge routing scale-out. The harness already exposes `KubernetesCloudPlatform.scale_workload` for replica changes.

## Decision

- **Defer** a dedicated `nginx` / `ingress_controller` integration slug until a Tier-3 product requires edge routing autoscale beyond K8s deployment replicas.
- **Canonical scale path (harness):** `ScalingProvisioner` + `KubernetesCloudPlatform` for `NEXUS_HOST` and workload deployments.
- **Edge routing:** document as operational concern outside ECP control plane v1 - operators scale ingress via existing K8s deployment targets using the same provisioner.

Rejected: parallel nginx-specific scaler duplicating K8s deployment API.

## Consequences

- ECP-6.2 integration scaffold remains optional/backlog.
- Lab reference policy (`HARNESS_ENVIRONMENT.md`) covers queue→worker rules without nginx slug.

## Compliance

- Tier-0 capacity module remains single control plane.
- Plan row ECP-6.1 closed as documentation decision.
