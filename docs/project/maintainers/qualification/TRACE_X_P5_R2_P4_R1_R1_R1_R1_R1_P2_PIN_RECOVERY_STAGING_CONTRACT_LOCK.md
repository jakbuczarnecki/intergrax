# TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1 — P2 Pin Recovery Staging Contract Lock

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** |
| **Production delta** | **0** |
| **FINAL_COMMIT** | `903465560563b13fb33cefeeffb52afe9f426955` |
| **Parent** | [`TRACE_X_P5_R2_P4_R1_R1_R1_R1_DURABLE_REQUIREMENT_FACT_RECOVERY_LOCK.md`](TRACE_X_P5_R2_P4_R1_R1_R1_R1_DURABLE_REQUIREMENT_FACT_RECOVERY_LOCK.md) |
| **Rejected design baseline** | `2c1fccf4b611afcdd7407e37c4ba33e76f81ce9a` — parent lock assumed `requirement_recovery_staging.v1` inside the pin envelope **without** a canonical typed `ExecutionIntegrationConfigurationPinningStore` write/read contract |
| **Blocker** | `R2-P4-P2-PIN-RECOVERY-STAGING-CONTRACT-30` — **RESOLVED IN DESIGN** (this artifact) |
| **P4 / R1 chain** | **BLOCKED** on P4 implementation after audit |
| **FRZ-TRC-11** | **OPEN** |
| **P5 / CERT** | **NOT ENTERED** |

## 1. Reconciliation boundary

Parent [`TRACE-X-P5-R2-P4-R1-R1-R1-R1`](TRACE_X_P5_R2_P4_R1_R1_R1_R1_DURABLE_REQUIREMENT_FACT_RECOVERY_LOCK.md) locks **what** durable fields Case C needs and **that** they must live in the **same** P2 durable row as provenance. It does **not** define the sanctioned **typed** store API for atomic write/read of recovery staging with the pin.

This artifact closes only the P2 pin-record + recovery-staging **contract** (write, read, idempotency, schema, adapter parity). It does **not** reopen accepted decisions from R1 through R1-R1-R1-R1 (requirement event identity, timestamp canon, Case C/D algorithms, pin→spine→I/O sequencing, no second store).

## 2. Blocker disposition

| Blocker | Disposition |
|---|---|
| `R2-P4-P2-PIN-RECOVERY-STAGING-CONTRACT-30` | **RESOLVED IN DESIGN** — one canonical pin record model; extended `pin()` + `read_pin_records()` on `ExecutionIntegrationConfigurationPinningStore`; optional additive envelope field under existing provenance schema v1 |

## 3. Closed-world: one canonical durable row model

Exactly **one** typed durable pin row — no parallel configured-specific row type, no shadow recovery document, no second key.

**Canonical type (Integrations contract locus):**

```text
ExecutionIntegrationConfigurationPinRecord
    subject: IntegrationConfigurationSubject
    provenance: ExecutionIntegrationConfigurationProvenance
    requirement_recovery_staging: ExecutionIntegrationConfigurationRequirementRecoveryStaging | None
```

`tenant_id` and `execution_id` remain authoritative on `provenance` (unchanged). The row is the unit of CAS / `put_if_absent` for both production adapters and the in-memory reference store.

**Authority counts (unchanged intent):**

```text
semantic pin store owner = 1
durable row = 1
recovery staging store = 0
shadow index = 0
```

The existing execution-level subject index (`integration_config_provenance_index` / document partition enumeration) may remain **only** as already-sanctioned P2 implementation detail — not a recovery index.

## 4. Recovery staging type

**Canonical type (neutral, immutable):**

```text
ExecutionIntegrationConfigurationRequirementRecoveryStaging
```

Minimum fields (aligned with parent R1-R1-R1-R1 §4 — field names normalized to a **flat** staging object; semantic equivalent to parent’s nested `requirement_recovery_staging.v1`):

| Field | Required | Notes |
|---|---|---|
| `requirement_boundary_recorded_at` | Yes | timezone-aware UTC; pin-success instant for Case C timestamp canon |
| `task_id` | Yes | |
| `run_id` | Yes | |
| `attempt_id` | Yes | |
| `node_id` | No | omit or `None` when unknown |
| `agent_id` | No | |
| `step_id` | No | |
| `correlation_id` | No | |
| `traceparent` | No | |
| `tracestate` | No | |

All fields immutable after construction. Validation lives in Integrations/contracts (or shared contracts imported by Integrations only — **no** Runtime → KV/DocumentStore).

**Policy (orchestration, not adapter):**

- For obligations that require Case C recovery (`CONFIGURED_ADOPTED` on the corrected P4 path), the **first** successful `pin()` for that subject must supply **non-`None`** staging.
- `EFFECTIVE_ONLY` pins: `requirement_recovery_staging=None` unless a future lock explicitly requires otherwise (out of scope here).

## 5. Semantic ownership

| Concern | Owner |
|---|---|
| Pin record persistence semantics | Integrations — `ExecutionIntegrationConfigurationPinningStore` + pin-record codec |
| Construction of correlation / boundary timestamp / trace snapshot | Runtime (or application orchestration at pin boundary) via **typed** staging built **before** `pin()` |
| Serialize / deserialize durable bytes | Applications persistence module (`encode_*` / `decode_*`) — **no** active-runtime context |
| Case C rebuild | Reconstruction / repair orchestration consumes `read_pin_records()` — **no** private adapter JSON parsing in Runtime |

**Forbidden inside** `KvExecutionIntegrationConfigurationPinningStore`, `DocumentStoreExecutionIntegrationConfigurationPinningStore`, in-memory reference store, and pin envelope codec:

- `peek_active_execution_*`, `require_active_execution_identity`, `peek_active_execution_evidence_context`, or any other active-context lookup.

## 6. Atomicity

The **first** successful durable write for `(tenant_id, execution_id, subject)` commits in **one** immutable value:

```text
tenant_id          (on provenance)
execution_id       (on provenance)
subject            (explicit on pin record + pin_subject in envelope)
provenance
requirement_recovery_staging | absent/None in legacy rows
```

No second write, no second KV key, no recovery-side shadow document. Idempotent replay re-reads the same bytes and accepts without rewrite.

## 7. Canonical store API (write + read)

**Extended write** — sole sanctioned API that may persist staging with provenance:

```text
ExecutionIntegrationConfigurationPinningStore.pin(
    *,
    subject: IntegrationConfigurationSubject,
    provenance: ExecutionIntegrationConfigurationProvenance,
    requirement_recovery_staging: ExecutionIntegrationConfigurationRequirementRecoveryStaging | None = None,
) -> None
```

**Read model — Option A (selected):** extend the **same** canonical store with typed pin-record read; keep provenance-only projection for ordinary reconstruction.

```text
read_pin_records(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
) -> tuple[ExecutionIntegrationConfigurationPinRecord, ...]
```

`read_all(tenant_id, execution_id) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]` **remains** for backward-compatible reconstruction projection (provenance-only). It does **not** expose subject or staging; callers must not infer subject from provenance alone.

**Justification for Option A over Option B:** recovery staging is part of the **same** durable row and the **same** P2 owner; a second read-only port would duplicate ownership surface without a second persistence implementation and would invite adapter-specific readers. One store, two read projections: provenance-only vs full pin record.

**Exit-criterion answer:**

> **Write:** `pin(..., requirement_recovery_staging=...)` atomically persists staging in the pin envelope on first CAS/`put_if_absent`.
>
> **Read:** `read_pin_records(...)` returns `ExecutionIntegrationConfigurationPinRecord` with `subject`, `provenance`, and `requirement_recovery_staging` (possibly `None` on legacy rows).

## 8. Idempotency and conflict semantics

Let `R = (subject, provenance, staging_normalized)` where `staging_normalized` is canonical equality on `ExecutionIntegrationConfigurationRequirementRecoveryStaging` or both sides `None`.

| Situation | Result |
|---|---|
| Row absent | Write complete pin record → **success** |
| Row present; `R` equal to stored | **Idempotent success** (no overwrite) |
| Row present; same `subject` + same `provenance`; staging differs (including `None` vs non-`None`) | **`CONFLICT`** |
| Row present; same staging; `provenance` differs | **`CONFLICT`** |
| Row present; same staging; `subject` differs | **`CONFLICT`** (distinct row keys per subject) |

Do **not** silently accept changed staging or provenance. Do **not** backfill staging on idempotent retry.

### 8.1 Legacy rows without staging

Rows written before P4 implementation lack `requirement_recovery_staging` in the envelope.

| Decode | `requirement_recovery_staging` on `ExecutionIntegrationConfigurationPinRecord` |
|---|---|
| Field absent or JSON `null` | `None` |

| Consumer path | Behavior |
|---|---|
| Ordinary reconstruction via `read_all()` | Unchanged — provenance only |
| Case C repair for configured-adopted obligation **requiring** staging | **Fail closed** — typed unavailable (e.g. `RECOVERY_STAGING_UNAVAILABLE` or repair-classification equivalent); **do not** manufacture staging from active context or checkpoint |
| Exact retry after partial implementation | If first pin omitted staging but retry supplies staging while row exists with `None` | **`CONFLICT`** (treat as staging change) |

## 9. Subject recovery

Recovery and repair must use **`read_pin_records()`** (or decode via the sanctioned pin-record codec used by the store — not ad hoc JSON in Runtime).

- `IntegrationConfigurationSubject` is **explicit** on every `ExecutionIntegrationConfigurationPinRecord`.
- Multiple subjects ⇒ multiple independent rows/obligations; never collapse.

**Not claimed:** `read_all() -> (subject, provenance)` — current production contract does not do that and will not be reinterpreted.

## 10. Encoder / schema evolution

Current envelope: `_PROVENANCE_SCHEMA_VERSION = 1`; `encode_integration_configuration_provenance` / `decode_integration_configuration_provenance` @ `intergrax/applications/_shared/integrations/persistence.py`.

**Decision:** **additive optional field** under **schema v1** — do **not** bump to v2 for staging alone.

```text
record.requirement_recovery_staging   # optional object; omitted on legacy rows
```

Renamed from design sketch `requirement_recovery_staging.v1` to a versioned **staging payload** inside the optional object if needed (`staging_schema_version: 1`), or flat fields with codec-owned validation — implementation chooses one encoder layout; **decode** must accept legacy rows without the key.

| Actor | Behavior |
|---|---|
| Old writer | No staging key → new reader sets staging `None` |
| New writer | Writes staging object when `pin(..., staging=...)` |
| Old reader (provenance-only code paths) | Ignores unknown record keys if using JSON parser that preserves only known fields; `read_all` path uses decode that tolerates absent staging |
| Corrupt / invalid staging shape | `CORRUPT_RECORD` |
| Wrong top-level `schema_version` | `UNSUPPORTED_SCHEMA_VERSION` (unchanged) |

No silent schema drift: encoder and decoder updated together in the same P4 change set; qualification gates prove round-trip and legacy read.

## 11. KV, DocumentStore, and in-memory parity

Same semantics on:

- `KvExecutionIntegrationConfigurationPinningStore`
- `DocumentStoreExecutionIntegrationConfigurationPinningStore`
- `InMemoryExecutionIntegrationConfigurationPinningStore` (reference; `non_durable_reference_only` classification unchanged)

Including: atomic first write; exact idempotent retry; staging conflict; provenance conflict; `read_pin_records` tenant/execution integrity; restart durability (KV + DocumentStore gates in §14).

## 12. Layering proof (design)

```text
intergrax/integrations/contracts/     ExecutionIntegrationConfigurationPinRecord
                                      ExecutionIntegrationConfigurationRequirementRecoveryStaging
                                      ExecutionIntegrationConfigurationPinningStore (extended Protocol)

intergrax/runtime/...                 builds staging; calls pin() via orchestration seam

applications/_shared/integrations/    encode/decode envelope; KV/DocumentStore/in-memory adapters

reconstruction / repair                 read_pin_records only; no Integrations → Runtime impl import
                                      no Runtime → concrete KV/DocumentStore import
```

## 13. Case C flow (after contract implementation)

**Happy path:**

```text
active configured execution
→ runtime builds ExecutionIntegrationConfigurationRequirementRecoveryStaging (typed)
→ materialize_validate_and_pin
→ pin(subject, provenance, requirement_recovery_staging=staging)   # single atomic row
→ runtime requirement event commit (spine)
→ business I/O
```

**After crash before spine:**

```text
retry with tenant_id + execution_id
→ read_pin_records(tenant_id, execution_id)
→ per CONFIGURED_ADOPTED subject: subject + provenance + staging from pin record
→ if staging None where required → fail closed
→ rebuild immutable ExecutionIntegrationConfigurationProvenanceRequirementFact
→ deterministic RuntimeEvent
→ commit
→ business I/O
```

No private adapter JSON parsing by runtime; no active-context lookup in storage.

## 14. Case D

If spine event already exists: full stored `RuntimeEvent` remains equality source. P2 staging may corroborate obligation discovery / integrity only; **do not** override spine fields from staging.

## 15. Corrected P4 tests (post-implementation)

1. Complete pin record written atomically (provenance + subject + staging in one durable value).
2. Exact retry idempotent (same subject, provenance, staging).
3. Changed staging → `CONFLICT`.
4. Changed provenance → `CONFLICT`.
5. KV restart preserves staging.
6. DocumentStore restart preserves staging.
7. `read_pin_records` returns subject + provenance + staging.
8. Legacy row without staging handled per §8.1.
9. Static/gate proof: no active-context calls in adapter/codec modules.
10. Docker Case C rebuild uses typed pin record, not raw JSON in runtime.

## 16. Tracker status

| ID | Status after this docs-only reconciliation |
|---|---|
| `TRACE-X-P5-R2-P4` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1-R1` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1-R1** (P4 implementation) |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1-R1** (staging contract superseded by child closure; durable field inventory remains authoritative) |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1` | **READY FOR AUDIT** |
| `FRZ-TRC-11` | **OPEN** |
| `P5` | **NOT ENTERED** |
| `CERT` | **NOT ENTERED** |

## 17. Independent audit notice

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
