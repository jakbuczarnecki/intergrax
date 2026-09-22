# ADR-GR-12-VECTOR-ADMIN-CONTROL-PLANE-BOUNDARY

| Field | Value |
| ----- | ----- |
| **Status** | Accepted — reconciled (GR-12-A4-R2-R0) |
| **Date** | 2026-09-22 |
| **Task** | `GR-12-A4-R2 / GR-12-A4-R2-R0 — Vector administration governance & revision semantics` |

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
| Identity | `VectorIndexIdentity(logical_name, tenant_id)` — **no intrinsic non-empty validation** on the dataclass; non-empty enforcement is on `VectorIndexSpec.__post_init__` and on the **live operator path** before CLA-04 mapping |
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
| `prepare_index` | CONDITIONAL_MUTATION | May create authoritative configuration (`CREATED`) or no-op (`ALREADY_COMPATIBLE`) | Yes — live operator path (governed **before** port invocation) |
| `close` | LIFECYCLE_ONLY | No | No |

---

## Decision 1 — `prepare_index` governance

**Option B — conditional mutation on the port; CLA-04 on every live operator `prepare_index` invocation.**

- `prepare_index` is **not** always a resulting state mutation (`ALREADY_COMPATIBLE` is a no-op), but the **live operator action may mutate** and must be governed **before** calling the port.
- Bootstrap/proof callers may continue to use the port **without** CLA-04 (same class as declarative bootstrap registry population).
- **Rejected timing:** authorize only after `prepare_index` or only when outcome is `CREATED` — mutation may already have occurred.

Rejected: Option A (always CP mutation — ignores no-op path); Option C as sole label (port remains the domain mutation owner, but operator orchestration is missing).

---

## Decision 2 — operator boundary

**LIVE OPERATOR SURFACE: NO**

There is no production API endpoint, admin service, or application operator surface that accepts `RequestIdentity` and performs vector index administration. CLA-04 adoption for vector requires **GR-12-A4-R2-R1** (bounded implementation).

---

## CLA-04 applicability

**APPLICABLE** for live operator-initiated `vector_index.prepare` (potentially consequential operation evaluated before invocation).

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
| `resource_id` | `{tenant_id}/{logical_name}` **after** live-operator validation of both fields |
| `resource_scope` | `vector_index.tenant/{tenant_id}` |
| `principal` | `RequestIdentity` on operator invocation (R2-R1) |

Provider physical names (e.g. Qdrant `logical__tenant__tenant_id`) are **diagnostics only**, not permission SSOT.

### Identity validation

1. `VectorIndexIdentity` is a frozen dataclass and **does not** validate non-empty `logical_name` / `tenant_id`.
2. `VectorIndexSpec.__post_init__` validates non-empty identity when constructing a spec (bootstrap/spec paths).
3. **Live operator path (R2-R1):** validate `logical_name` and `tenant_id` non-empty **before** CLA-04 resource identity construction — fail closed before policy evaluation.
4. **Forbidden:** synthetic tenant placeholders (`platform`, `default`, `profile_id` as tenant) without real upstream identity.

Indexes remain **tenant-scoped** at the logical identity layer.

---

## Canonical configuration projection

Qualification SSOT type: **`VectorIndexConfigurationProjection`** (provider-neutral logical schema; R2-R1 may promote to integrations contract types).

| Field | Included in digest? | Notes |
| ----- | ------------------- | ----- |
| `logical_name` | Yes | logical identity |
| `tenant_id` | Yes | authority scope |
| `dense_dimension` | Yes | configuration |
| `dense_metric` | Yes | configuration |
| `dense_channel_name` | Yes | configuration |
| `required_capabilities` | Yes | exact capability set |
| `sparse_lexical_channel_name` | Conditional | when sparse capability required |
| `point_count` | **No** | runtime data |
| `reachable` | **No** | health/runtime |
| physical provider id / host / credentials | **No** | provider diagnostics |

**Same schema** for:

- **target:** `VectorIndexSpec` → projection → digest
- **current:** `VectorIndexDescription` → projection → digest (when `exists=True`)

Do **not** use Qdrant enums, collection payloads, or provider config objects in the canonical projection.

### Revision digest semantics

- Deterministic canonical serialization + cryptographic digest (algorithm chosen in R2-R1; not provider-specific).
- Equal logical configuration → equal digest across providers.
- **`VectorIndexPrepareOutcome.ALREADY_COMPATIBLE` ≠ guaranteed exact configuration equality** with the requested spec. Port helper `validate_spec_against_description` checks existence, dense dimension, and required capabilities — **not** dense metric, dense/sparse channel names, or full spec equality. Digest equality is defined only via the canonical projection, not via prepare outcome.

### Absent index state

When `describe_index` reports `exists=False`, **`current_revision`** uses semantic state **`ABSENT`** (or a canonical absent digest derived from that state). Target revision is the projection digest of the requested `VectorIndexSpec`. CLA-04 must represent mutation intent: **create vector index** (`ABSENT` → target).

### Incompatible existing index

If the index exists but persisted shape is incompatible with the requested spec, the neutral port **does not** rebuild or recreate. **Fail closed** (`VectorIndexCompatibilityError` / compatibility error). No automatic drop/recreate/migrate in R2/R2-R0.

---

## No CAS guarantee

The neutral `VectorIndexAdministration` port does **not** expose compare-and-swap or provider generation tokens. Stale-state handling is **optimistic re-read + invalidation of prior authorization**, not CAS.

---

## TOCTOU invalidation

Providers do not expose CAS on the neutral port (see **No CAS guarantee** above).

1. Read `describe_index` → `current_revision` digest (or `ABSENT`).
2. Build `target_revision` from requested spec projection.
3. CLA-04 **authorize** against fresh `current_revision` / `target_revision` (and evidence fields).
4. Immediately before `prepare_index`, **re-read** `describe_index` and recompute current digest.
5. If authorized current digest **A** ≠ re-read digest **B**: **previous authorization is invalid**; `prepare_index` **must not** run under the stale authorization.
6. **Allowed:** fail closed / abort with typed stale result (**preferred**), or **one bounded** fresh CLA-04 evaluation against the newly observed state — never rebuild the request and continue without `authorize()`.
7. **No unbounded** `while stale: reauthorize` loops.

**Race window:** re-read → `prepare_index` still has a tiny window without provider CAS; ADR does not claim strict serializable CAS.

Idempotency of `prepare_index` does **not** replace authorization timing.

---

## Live prepare authorization timing

Live operator `prepare_index` is a **governed potentially-consequential operation**. CLA-04 occurs **before** `VectorIndexAdministration.prepare_index` invocation, even when the final provider outcome may be `ALREADY_COMPATIBLE` (authorized operator action, no resulting mutation).

---

## Risk mapping (live operator path)

| Path | Risk |
| ---- | ---- |
| `describe_index` / `probe` | No CLA-04 |
| Live `prepare_index` (authorized before call) → `ALREADY_COMPATIBLE` | Governed operator action; no resulting configuration mutation |
| Live `prepare_index` → `CREATED` | HIGH (new authoritative index configuration) |
| Stale digest at execution | Abort or fresh CLA-04 — no silent mutation |
| Future drop/recreate on port | CRITICAL if ever added — out of scope |

---

## Evidence (governed mutation)

Emit `ControlPlaneMutationAuthorizationEvidence` (CLA-04) with at minimum:

- `mutation_id`, `mutation_type`, `principal`, `resource_scope`, `resource_type`, `resource_id`
- `current_revision`, `target_revision` (configuration digests or `ABSENT`)
- `mutation_id`, policy decision
- provider prepare outcome (`CREATED` / `ALREADY_COMPATIBLE` / compatibility failure) is **execution result**, not authority identity

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

## Non-goals (R2 / R2-R0)

- Memory governance (GR-12-A4-R3)
- Catalog / GR-10 / RequestIdentity / CLA-04 public contract changes
- Provider adapter authorization or production digest implementation (R2-R1)
- Changing public `VectorIndexIdentity` / `VectorIndexAdministration` surface in R2-R0
- Drop, recreate, migrate, delete, reindex on the neutral port
- Qualifying bootstrap-only paths as CLA-04 governed
- Treating `QdrantRagStore.delete_collection` as evidence against the neutral admin port (separate architecture concern)
