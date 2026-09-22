# ADR-GR-12-VECTOR-ADMIN-CONTROL-PLANE-BOUNDARY

| Field | Value |
| ----- | ----- |
| **Status** | Accepted (GR-12-A4-R2) |
| **Date** | 2026-09-22 |
| **Task** | `GR-12-A4-R2 — Vector Administration Governance Architecture & CLA-04 Mapping Decision` |

---

## Context

GR-12 residual path `CP-VECTOR-INDEX-ADMIN` was classified `ARCHITECTURE_DECISION_REQUIRED` after catalog hot reload qualification closed. The platform already exposes a **provider-neutral** integration contract for vector/search index administration. Governance must not introduce a second vector admin port, provider-specific policy, or authorization inside provider adapters.

Control-plane qualification distinguishes **declarative bootstrap provisioning** from **live operator mutation** of a running system (see catalog `CP-BOOT-PLUGIN-REGISTER` vs `CP-PLUGIN-CATALOG-HOT-RELOAD`).

---

## Existing canonical port

| Item | Value |
| ---- | ----- |
| Module | `intergrax.integrations.contracts.vector_index_administration` |
| Contract | `VectorIndexAdministration` (runtime-checkable `Protocol`) |
| Identity | `VectorIndexIdentity(logical_name, tenant_id)` — both non-empty strings |
| Spec / description | `VectorIndexSpec`, `VectorIndexDescription`, `VectorIndexPrepareResult` |

### Public operations (code-evidenced)

| Operation | Semantics |
| --------- | --------- |
| `probe()` | Integration health via provider control-plane reachability |
| `describe_index(identity)` | Read persisted index projection (exists, dimensions, capabilities, point count) |
| `prepare_index(spec)` | Idempotent ensure: **create** collection/index when absent; **validate** when present (`CREATED` vs `ALREADY_COMPATIBLE`) |
| `close()` | Close local adapter/client |

**Destructive lifecycle** (drop, delete, recreate, reindex, migrate, replace, truncate) is **not** exposed on the neutral port.

`delete_collection` exists only on the Qdrant adapter's internal client protocol; it is **not** callable through `VectorIndexAdministration`.

---

## Production implementations (evidence only)

| Adapter | Notes |
| ------- | ----- |
| `QdrantVectorIndexAdministration` | Sole production `VectorIndexAdministration` implementation |

`prepare_index` on Qdrant: may call `create_collection` when index missing; when present runs `validate_spec_against_description` (dimension/capabilities) without mutating configuration. Does **not** alter metric/dimensions on existing indexes (raises `VectorIndexCompatibilityError`).

---

## Production callers (inventory)

| Caller | Layer | Operations | Classification |
| ------ | ----- | ------------ | -------------- |
| `PlatformSearchIndexBootstrapAdapter` | `platform_proofs` bootstrap | `probe`, `prepare_index`, `describe_index`, `close` | A — startup/bootstrap |
| `QdrantVectorStorageAdapter` | `platform_proofs` data-pack load | `prepare_index`, `describe_index` (via flow), `close` | A/B — bootstrap + runtime data-plane preparation |
| `QdrantVectorIndexIdentityResolver` | `platform_proofs` runtime identity | `describe_index`, `close` | B — runtime read |
| `qdrant_runtime` (qualification) | proof harness | `describe_index`, admin wiring | E — test/proof |

No `intergrax.applications` or operator HTTP/CLI/admin service invokes `VectorIndexAdministration` today.

### Side-channel finding (documented, out of R2 scope)

`QdrantRagStore` (data-plane `VectorStore`) may `create_collection` / `delete_collection` internally for legacy RAG paths. This is **not** a second public admin port but bypasses `VectorIndexAdministration` for collection lifecycle. Consolidation is a separate architecture concern; it does not block R2.

---

## Operation classification (control-plane)

| Operation | Class | Consequential CP mutation | Live operator GR-12 |
| --------- | ----- | ------------------------- | ------------------- |
| `probe` | READ_ONLY | No | No |
| `describe_index` | READ_ONLY | No | No |
| `prepare_index` | CONDITIONAL_MUTATION | Only when outcome is `CREATED` (or would fail compatibility) | Yes — when exposed via future operator service |
| `close` | LIFECYCLE_ONLY | No | No |

---

## Decision 1 — `prepare_index` governance

**Option B — conditional mutation; CLA-04 only on live operator path when mutation is consequential.**

- `prepare_index` is **not** always a control-plane mutation (`ALREADY_COMPATIBLE` is a no-op).
- Bootstrap/proof callers may continue to use the port **without** CLA-04 (same class as declarative bootstrap registry population).
- When a **live operator** initiates index preparation that may create or change authoritative configuration, that request must pass CLA-04 **before** calling `prepare_index`.

Rejected: Option A (always CP mutation — ignores no-op path); Option C as sole label (port remains the domain mutation owner, but operator orchestration is missing).

---

## Decision 2 — operator boundary

**LIVE OPERATOR SURFACE: NO**

There is no production API endpoint, admin service, or application operator surface that accepts `RequestIdentity` and performs vector index administration. CLA-04 adoption for vector requires **GR-12-A4-R2-R1** (bounded implementation).

---

## CLA-04 applicability

**APPLICABLE** for live operator-initiated `vector_index.prepare` when consequential.

**NOT** required for existing bootstrap-only `prepare_index` callers until/unless they are promoted to operator-governed entrypoints.

---

## Preferred architecture (Option A)

```text
operator request (RequestIdentity)
  → VectorIndexAdminService (applications / control-plane orchestration)
  → ControlPlaneMutationAuthorizationBoundary (CLA-04)
  → ControlPlaneMutationPolicyEvaluator (injectable)
  → VectorIndexAdministration (integrations domain port)
  → provider adapter (mechanics only)
```

### Rejected alternatives

| Alternative | Reason |
| ----------- | ------ |
| **B — `GovernedVectorIndexAdministration` wrapper as public authority** | Couples governance to integrations layer; duplicates operator context ownership |
| **C — provider-layer CLA-04 in Qdrant adapter** | Violates layer direction; provider credentials ≠ business permission |
| Second vector admin port / provider-specific governance contract | Forbidden by GR-12 boundary rules |

---

## CLA-04 resource mapping (provider-neutral)

| Field | Value |
| ----- | ----- |
| `mutation_type` | `vector_index.prepare` |
| `resource_type` | `vector_index` |
| `resource_id` | `{tenant_id}/{logical_name}` from `VectorIndexIdentity` (logical authority) |
| `resource_scope` | `vector_index.tenant/{tenant_id}` |
| `principal` | `RequestIdentity` on operator invocation (R2-R1) |

Provider physical names (e.g. Qdrant `logical__tenant__tenant_id`) are **diagnostics only**, not permission SSOT.

### Tenant semantics

`VectorIndexIdentity.tenant_id` is **required** (validated non-empty). Indexes are **tenant-scoped** at the logical identity layer; host-global vector indexes are not modeled on this contract.

---

## Revision / state semantics

There is **no** canonical `VectorIndexRevision` or generation SSOT today.

`describe_index` exposes shape (dimension, metric, capabilities, point count) but not a version token suitable for CLA-04 `current_revision` / `target_revision`.

**R2-R1 must introduce** a provider-neutral **configuration digest** function:

- **current**: digest projected from `VectorIndexDescription` + channel names
- **target**: digest from requested `VectorIndexSpec`

Until then, vector control-plane adoption cannot be qualified.

---

## TOCTOU / stale state

Providers do not expose CAS mutation on the neutral port. Strategy:

1. Operator service reads `describe_index` → builds `current_revision` digest.
2. CLA-04 authorizes against `current_revision` / `target_revision`.
3. Immediately before `prepare_index`, **re-read** `describe_index`; if digest ≠ authorized `current_revision`, **fail closed** (reauthorize or abort).
4. No fake CAS or provider-generation assumptions.

Idempotency of `prepare_index` does **not** replace authorization.

---

## Risk mapping (live operator path)

| Path | Risk |
| ---- | ---- |
| `describe_index` / `probe` | No CLA-04 |
| `prepare_index` → `ALREADY_COMPATIBLE` | Operator action; no resulting mutation — audit optional; no HIGH risk mutation |
| `prepare_index` → `CREATED` | HIGH (new authoritative index configuration) |
| Future drop/recreate on port | CRITICAL if ever added — out of scope for R2 |

---

## Evidence (governed mutation)

Emit `ControlPlaneMutationAuthorizationEvidence` (CLA-04) with at minimum:

- `mutation_id`, `mutation_type`, `principal`, `resource_scope`, `resource_type`, `resource_id`
- `current_revision`, `target_revision` (digests once implemented)
- policy decision + provider-neutral prepare outcome (`CREATED` / `ALREADY_COMPATIBLE`)

No provider-specific evidence contract.

---

## Pluginability

Inject `ControlPlaneMutationPolicyEvaluator` at application composition (same spine as catalog hot reload). No vector-specific policy plugin interface. Provider code unchanged.

---

## Layer ownership

| Concern | Owner |
| ------- | ----- |
| Operator orchestration | `applications` (future `VectorIndexAdminService`) |
| Governance spine | `intergrax.runtime.governance` + CLA-04 contracts |
| Domain mutation | `VectorIndexAdministration` + provider adapters |
| Provider mechanics | `intergrax.integrations.providers.vector_store.*` |

Composition root today: `open_qdrant_vector_index_administration` in `intergrax.integrations.providers.vector_store.qdrant.opens`.

---

## Failure semantics

- CLA-04 DENY → do not call `prepare_index`.
- Stale digest at execution → fail closed; no silent mutation.
- Provider credentials missing → integration error (capability), not policy ALLOW.

---

## Next implementation task

**GR-12-A4-R2-R1 — Governed Vector Index Operator Service & CLA-04 Enforcement**

Deliver: operator service, CLA-04 request builder, configuration digest contract, composition wiring, qualification tests. Do not modify `VectorIndexAdministration` public surface unless digest types are added as neutral contract types.

---

## Non-goals (R2)

- Memory governance (GR-12-A4-R3)
- Catalog / GR-10 / RequestIdentity / CLA-04 public contract changes
- Provider adapter authorization
- Exposing destructive lifecycle on the neutral port
- Qualifying bootstrap-only paths as CLA-04 governed
