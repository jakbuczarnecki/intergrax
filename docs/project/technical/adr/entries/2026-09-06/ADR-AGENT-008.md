# ADR-AGENT-008: Durable runtime projection rehydration (EA-03)

| Field | Value |
|-------|-------|
| **Status** | Accepted / Implemented |
| **Date** | 2026-09-06 |
| **Deciders** | Agent Platform / Harness architecture |
| **Related** | [AGENT_DISTRIBUTION.md](../../../../architecture/AGENT_DISTRIBUTION.md) §18 · [ADR-AGENT-004](../2026-08-12/ADR-AGENT-004.md) · [ADR-AGENT-005](../2026-08-17/ADR-AGENT-005.md) · [ADR-AGENT-006](../2026-09-01/ADR-AGENT-006.md) · [ADR-AGENT-007](../2026-09-02/ADR-AGENT-007.md) · EA-03 · [AGENT_PLATFORM_FINAL_CLOSURE.md](../../../../maintainers/audits/AGENT_PLATFORM_FINAL_CLOSURE.md) (Stage 18 historical closure) |

## Context

The frozen production invariant **startup-time reprojection is forbidden** assumed that MaterializedRegistryProjection would remain available in the runtime projection store across process boundaries.

Enterprise durable testing demonstrated a fail-closed gap.

### Context failure

```text
Process A dies
    ↓
durable lifecycle survives
    ↓
process-local projection disappears
    ↓
Process B cannot serve
```

RuntimeRevision, `traffic_serving_revision_id`, roster/lock/materialization authority, and activation CAS semantics are durable. MaterializedRegistryProjection, AgentRegistryRead, and instantiated agent runtime objects are process-local. Without an explicit rehydration contract, cold restart cannot restore traffic-serving execution.

**Gap closed:** EA-03 durable runtime projection rehydration.

## Decision

Separate durable runtime authority from process-local materialized runtime objects.

### Durable runtime authority

| Artifact / pointer | Role |
|--------------------|------|
| RuntimeRevision | Immutable revision identity and authority references |
| EffectiveRoster identity | Content-addressed roster authority for the revision |
| MaterializedRuntimeLock identity | Revision-bound lock authority |
| RuntimeMaterialization identity | Revision-bound materialization authority |
| `traffic_serving_revision_id` | Durable serving pointer |
| RuntimeRegistryProjectionDescriptor | Typed immutable reconstruction authority keyed by revision |

### Process-local runtime objects

| Object | Role |
|--------|------|
| MaterializedRegistryProjection | Process-local registry projection store payload |
| AgentRegistryRead | Execution-facing registry read surface |
| Agent runtime instances | Instantiated agents and factories |

```text
DURABLE RUNTIME AUTHORITY ≠ PROCESS-LOCAL MATERIALIZED RUNTIME OBJECT
```

### Authority split

```text
Durable authority
├── RuntimeRevision
├── EffectiveRoster
├── RuntimeLock
├── Materialization
├── ProjectionDescriptor
└── Serving Pointer

Process-local
├── MaterializedRegistryProjection
├── AgentRegistryRead
└── agent instances
```

## Rehydration rule

**Startup-time projection from current mutable desired state remains forbidden.**

Allowed:

```text
deterministic rehydration of the already traffic-serving,
revision-bound projection from durable immutable authority
```

Rehydration:

- is **not** install,
- is **not** bind,
- is **not** revision build,
- is **not** activation,
- is **not** routing decision,
- does **not** mutate serving pointer.

Rehydration reconstructs the process-local projection for the revision already selected by `traffic_serving_revision_id`. It does not derive authority from current installation, binding, or desired roster state.

## Why live projection is not persisted

Persisting MaterializedRegistryProjection directly was **rejected** because it:

- contains process-local runtime objects,
- may contain instantiated agents/factories,
- should not be serialized/pickled,
- would couple persistence to Python runtime representation,
- would break provider-neutral architecture.

Durable persistence stores only typed reconstruction authority (RuntimeRegistryProjectionDescriptor and referenced immutable lifecycle artifacts). Process B rebuilds the projection deterministically from that authority.

## Descriptor contract

RuntimeRegistryProjectionDescriptor is a typed, versioned durable artifact — not a generic JSON blob. Persistence fields must not use Any or untyped dict payloads.

```text
RuntimeRegistryProjectionDescriptor
├── typed ApplicationManifest
├── typed BuildContextDescriptorSnapshot
├── SkillProfile?
├── ToolProfile?
├── EnvironmentIdentitySnapshot?
├── revision IDs
├── roster identity
├── lock identity/digest
├── materialization locator/digest
└── schema/descriptor versions
```

No generic JSON blob. No Any persistence fields.

Implementation: [registry_projection_descriptor.py](../../../../../../intergrax/applications/_shared/registry_projection_descriptor.py).

## Activation invariant

```text
SERVING(N) ⇒ durable projection descriptor(N) exists
```

Ordering at activation:

```text
projection input validated
→ descriptor built
→ descriptor persisted
→ activation CAS commit
→ serving pointer N
```

Descriptor may exist without serving if activation fails. Serving must never exist without descriptor.

## Cold start flow

```text
Serving Pointer=N
      ↓
Descriptor(N)
      ↓
Authority validation
      ↓
Projection rehydration
      ↓
AgentRegistryRead
      ↓
Execution
```

Detailed cold restart:

```text
Process B starts
        ↓
read traffic_serving_revision_id = N
        ↓
load descriptor(N)
        ↓
resolve revision-bound authority
        ↓
validate roster/lock/materialization/release
        ↓
assemble canonical projection input
        ↓
build process-local MaterializedRegistryProjection
        ↓
AgentRegistryRead
        ↓
Execution
```

Implementation: [registry_projection_rehydrator.py](../../../../../../intergrax/applications/_shared/registry_projection_rehydrator.py), [durable_agent_platform_runtime.py](../../../../../../intergrax/applications/_shared/durable_agent_platform_runtime.py).

## Failure semantics

Fail-closed — process cannot become serving-ready when any of the following occur:

| Condition | Result |
|-----------|--------|
| missing descriptor | fail closed |
| corrupt descriptor | fail closed |
| schema mismatch | fail closed |
| revision mismatch | fail closed |
| roster mismatch | fail closed |
| lock mismatch | fail closed |
| artifact mismatch | fail closed |

**Forbidden fallbacks:**

- empty registry fallback,
- manifest fallback,
- desired roster fallback,
- automatic new activation.

## Artifact immutability

Non-authoritative runtime caches such as __pycache__ and .pyc are excluded from content identity digests. Authoritative source mutation must change digest. Tests prove both properties.

## Consequences

### Positive

- cold restart works without Process A objects,
- serving authority remains durable,
- runtime projection remains process-local,
- persistence is provider-neutral,
- future PostgreSQL/distributed adapter possible,
- replay/historical semantics remain revision-bound.

### Trade-offs

- descriptor is an additional durable artifact,
- startup includes deterministic rehydration,
- corrupt/missing descriptor intentionally prevents readiness.

## Evidence

### Integration / enterprise E2E

| Test | File |
|------|------|
| Durable lifecycle happy path + restart | [test_enterprise_agent_lifecycle_durable_e2e.py](../../../../../../tests/integration/agent_distribution/test_enterprise_agent_lifecycle_durable_e2e.py) — test_enterprise_durable_lifecycle_happy_path_and_restart, test_enterprise_restart_preserves_active_revision |
| Enterprise projection rehydration E2E | [test_enterprise_projection_rehydration_e2e.py](../../../../../../tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py) |
| Durable F3 — revoked install, serving unchanged after reopen | same — test_enterprise_durable_f3_revoked_install_rejected_serving_unchanged_after_reopen |
| Durable F4 — failed activation preserves serving after reopen | same — test_enterprise_durable_f4_failed_activation_preserves_serving_after_reopen |
| Durable F5 — emergency rollback rehydrates prior revision | same — test_enterprise_durable_f5_emergency_rollback_rehydrates_prior_revision |

### Unit / contract / architecture gates

| Test | File |
|------|------|
| Descriptor contract (typed round-trip, version mismatch, no Any) | [test_registry_projection_descriptor_contract.py](../../../../../../tests/unit/applications/test_registry_projection_descriptor_contract.py) |
| Descriptor corruption fail-closed | [test_registry_projection_descriptor_corruption.py](../../../../../../tests/unit/applications/test_registry_projection_descriptor_corruption.py) |
| SQLite bounded store + descriptor reopen | [test_sqlite_agent_distribution_bounded_store.py](../../../../../../tests/unit/agent_distribution/test_sqlite_agent_distribution_bounded_store.py) — test_sqlite_revision_serving_and_descriptor_reopen |
| Rehydration architecture gates | [test_registry_projection_rehydration_architecture_gates.py](../../../../../../tests/unit/applications/test_registry_projection_rehydration_architecture_gates.py) |
| Digest hardening (__pycache__ exclusion, authoritative mutation) | [test_directory_content_digest_hardening.py](../../../../../../tests/unit/agent_distribution/test_directory_content_digest_hardening.py) |

### Architecture documentation

| Document | Role |
|----------|------|
| [AGENT_DISTRIBUTION.md](../../../../architecture/AGENT_DISTRIBUTION.md) §18 | EA-03 frozen semantics and cold-restart diagram |
| [AGENT_PLATFORM_FINAL_CLOSURE.md](../../../../maintainers/audits/AGENT_PLATFORM_FINAL_CLOSURE.md) | Stage 18 historical architecture closure (pre-EA-03) |

### EA-03 implementation paths

| Module | Responsibility |
|--------|----------------|
| [registry_projection_descriptor.py](../../../../../../intergrax/applications/_shared/registry_projection_descriptor.py) | Typed durable descriptor contract |
| [registry_projection_rehydrator.py](../../../../../../intergrax/applications/_shared/registry_projection_rehydrator.py) | Deterministic process-local rehydration |
| [durable_agent_platform_runtime.py](../../../../../../intergrax/applications/_shared/durable_agent_platform_runtime.py) | Durable runtime composition and cold-start wiring |
| [reference_production_lifecycle.py](../../../../../../intergrax/applications/_shared/reference_production_lifecycle.py) | Activation-time descriptor persistence |
| [sqlite_stores.py](../../../../../../intergrax/agent_distribution/sqlite_stores.py) | SQLite bounded durable store adapter |

## Compliance

- Tier boundaries preserved: Tier-0 durable stores and descriptors; Tier-3 hosts consume rehydrated projections
- Supersedes implicit assumption that process-local projection survives restart
- Extends [ADR-AGENT-005](../2026-08-17/ADR-AGENT-005.md) store ownership with descriptor persistence
- Complements [ADR-AGENT-006](../2026-09-01/ADR-AGENT-006.md) materialization authority and [ADR-AGENT-007](../2026-09-02/ADR-AGENT-007.md) historical roster authority
- Aligns with [AGENT_DISTRIBUTION.md](../../../../architecture/AGENT_DISTRIBUTION.md) EA-03 frozen section
