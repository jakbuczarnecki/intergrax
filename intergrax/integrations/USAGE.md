# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# Integrations catalog — usage notes

## Lab stack honesty (INT-MAINT-03)

Integration manifests expose ``requires_local_container`` on
:class:`~intergrax.integrations.core.manifest.IntegrationManifest` and catalog
rows. Use it when wiring Tier-3 hosts:

| Value | Meaning |
|-------|---------|
| ``False`` (default) | In-process, SaaS API, or embedded backend — no local Docker required |
| ``True`` | Realistic lab testing expects a local container or compose stack |

**SaaS-only slugs** (no local substitute) are indexed in
``intergrax.integrations._shared.saas_only_slugs.SAAS_ONLY_SLUGS``. These
providers require vendor credentials; do not treat them as offline lab defaults.

**Local container slugs** (Postgres extensions, Kafka, MinIO, Vault, Kubernetes)
are listed in ``LOCAL_CONTAINER_SLUGS`` — pair with compose profiles or
``intergrax doctor`` integration health checks before enabling on a host.

Per-provider notes live beside each bundle under
``intergrax/integrations/providers/<category>/<slug>/USAGE.md``.

## Ingress / nginx (INT-MAINT-04)

The **nginx / ingress controller** integration slug is owned by
**Elastic Capacity & Scaling (ECP)**, not the Integrations catalog. See
[`docs/plan/ELASTIC_CAPACITY_AND_SCALING.md`](../../docs/plan/ELASTIC_CAPACITY_AND_SCALING.md)
and ADR-SCALE-002 for the defer decision and Kubernetes-first deployment path.

Integrations documents the bridge only; capacity ingress wiring remains in ECP.
