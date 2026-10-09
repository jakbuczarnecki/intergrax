# TRACE-X-P5-R2-P4-R1-R1-R1 — Requirement Event Canonical Retry Identity

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** |
| **Production delta** | **0** |
| **Rejected R1-R1 baseline** | `7093bea7f65553f1f1f1cfc56dba75664a99a7b31c` |
| **Parent** | [`TRACE_X_P5_R2_P4_R1_R1_REQUIREMENT_EVIDENCE_EMISSION_BOUNDARY_DUAL_WRITE_RECONCILIATION.md`](TRACE_X_P5_R2_P4_R1_R1_REQUIREMENT_EVIDENCE_EMISSION_BOUNDARY_DUAL_WRITE_RECONCILIATION.md) |
| **Grandparent** | [`TRACE_X_P5_R2_P4_R1_RECONSTRUCTION_REQUIREMENT_AUTHORITY_AS_OF_RECONCILIATION.md`](TRACE_X_P5_R2_P4_R1_RECONSTRUCTION_REQUIREMENT_AUTHORITY_AS_OF_RECONCILIATION.md) |
| **P4** | **BLOCKED** pending corrected implementation |
| **FRZ-TRC-11** | **OPEN** |
| **P5 / CERT** | **NOT ENTERED** |

## 1. Reconciliation boundary

This artifact corrects only the retry-identity gap in R1-R1. It does not reopen
accepted decisions:

- typed requirement spine fact;
- magic payload authority = `0`;
- Option B for `execution_as_of`;
- runtime-owned physical emitter;
- Integrations as the pin-only semantic source;
- commit port;
- `RuntimeEventBus.record`;
- deterministic logical evidence identity;
- `pin → spine durable → business I/O`;
- orphan-pin repair;
- Docker E2E qualification.

The rejected R1-R1 assumption was:

> A deterministic `EventId` is sufficient to make a retry idempotent.

That is false for the canonical persistence contract. The existing
`reconcile_idempotent_event_acceptance` accepts a duplicate only when:

```text
accepted.positioned.event == incoming RuntimeEvent
```

The same `EventId` with any different canonical field is an integrity conflict.
Therefore, deterministic `EventId` is necessary but not sufficient; the complete
`RuntimeEvent` must be rebuilt from one immutable requirement fact.

## 2. RuntimeEvent equality inventory

The closed-world inventory is the complete `RuntimeEvent` model, including
derived fields and metadata. Equality is model equality, not equality of only
the logical evidence key.

| RuntimeEvent field | Canonical source | Stable across retry? | Derivation |
|---|---|---:|---|
| `event_id` | Immutable logical requirement fact identity | Yes | Deterministic hash/ID of `tenant_id + execution_id + subject + schema/event kind`; never UUID/random/current state |
| `tenant_id` | Typed requirement fact from the successful P2 pin handoff | Yes | Copied unchanged; persistence routing tenant must match |
| `task_id` | Original active execution identity | Yes | Captured in the immutable fact/request; never read from a later active context |
| `run_id` | Original active execution identity | Yes | Captured in the immutable fact/request; never rebound |
| `attempt_id` | Original active execution identity | Yes | Captured in the immutable fact/request; a later retry attempt cannot replace it |
| `execution_id` | P2 provenance and logical requirement fact | Yes | Copied unchanged; it is part of the logical evidence key |
| `node_id` | Immutable fact metadata, or canonical `None` for this requirement event | Yes | Never regenerated from the retry worker; use `None` unless the original fact carried a stable value |
| `agent_id` | Immutable fact metadata, or canonical `None` for this requirement event | Yes | Never regenerated from the retry worker; use `None` unless the original fact carried a stable value |
| `step_id` | Immutable fact metadata, or canonical `None` for this requirement event | Yes | Never regenerated from the retry worker; use `None` unless the original fact carried a stable value |
| `event_type` | Typed requirement event contract: `INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED` / `integration_configuration_provenance_requirement_committed` | Yes | Fixed new `RuntimeEventType` value for the configured-provenance requirement |
| `event_kind` | `event_type` | Yes | Let `RuntimeEvent` derive the enum value; do not supply a mutable/current kind |
| `event_category` | Runtime event/spine metadata registry | Yes | Derive from the fixed `event_type`; registry entry is mandatory |
| `ops_hint` | Runtime event/spine metadata registry | Yes | Derive from the fixed `event_type`; registry entry is mandatory |
| `phase` | Immutable requirement fact/request | Yes | Fixed at the original requirement boundary; never use the retry worker's current phase |
| `severity` | Immutable requirement fact/request | Yes | Fixed by the requirement event contract; never use a current policy/configuration lookup |
| `payload` | Typed requirement payload DTO | Yes | Serialize the same typed fields with canonical key ordering; no mutable metadata or current configuration lookup |
| `timestamp` | Immutable factual timestamp captured at the original pin/execution boundary | Yes | Carry the original timestamp in the typed fact/request; never call `datetime.now(...)` during retry |
| `correlation_id` | Original execution correlation metadata in the fact, or canonical empty value | Yes | Copy unchanged; do not mint a new correlation value |
| `parent_event_id` | Original fact metadata, or canonical `None` | Yes | Copy unchanged; this requirement fact has no implicit parent |
| `traceparent` | Original W3C trace metadata in the fact, or canonical `None` | Yes | Copy unchanged; never create a new child trace on retry |
| `tracestate` | Original W3C trace metadata in the fact, or canonical `None` | Yes | Copy unchanged; never refresh from the retry process |
| `schema_version` | Runtime event schema contract | Yes | Fixed canonical value for the event schema; no per-retry version lookup |

No field in this inventory may be regenerated nondeterministically during
retry. In particular, `RuntimeEvent` defaults for `event_id` and `timestamp`
are forbidden on the requirement recorder path.

## 3. Immutable requirement fact and identity model

Corrected P4 must introduce one immutable typed fact/request at the commit-port
boundary. It must contain, or deterministically derive, every field in §2:

```text
ExecutionIntegrationConfigurationProvenanceRequirementFact
  tenant_id
  execution_id
  IntegrationConfigurationSubject
  schema/event kind
  task_id
  run_id
  attempt_id
  factual_timestamp
  phase
  severity
  node_id / agent_id / step_id
  correlation_id
  parent_event_id
  traceparent / tracestate
  typed payload fields
  schema_version
```

The fact is created from the successful pin handoff and is immutable for the
obligation. The runtime recorder maps this fact to `RuntimeEvent` and calls the
existing `RuntimeEventBus.record`; it must not reconstruct the fact from current
configuration, catalog, adoption binding, or retry-worker state.

Two identities must remain distinct:

```text
logical fact identity
  tenant_id + ExecutionId + IntegrationConfigurationSubject + schema/event kind

full RuntimeEvent canonical identity
  equality of every RuntimeEvent field in §2
```

The logical identity remains the exactly-once semantic key. The full canonical
identity is the persistence integrity contract.

## 4. Timestamp decision

The event timestamp is locked to the factual timestamp captured once at the
original successful pin/execution boundary. The timestamp is carried in the
immutable typed commit request and reused for all retries of that obligation.

The corrected P4 implementation must make that value recoverable by the
existing retry/obligation path after a process crash. It must not substitute a
new wall-clock value when reconstructing the request. A retry for which the
original factual timestamp cannot be recovered is not idempotent and must fail
closed; it must not claim canonical duplicate acceptance.

Forbidden:

```python
datetime.now(...)
```

on any retry reconstruction path.

No synthetic timestamp derived from a hash is permitted because it would
misrepresent temporal semantics. If the selected production seam cannot carry
and recover the original factual timestamp, corrected P4 stops with:

```text
STOP — ARCHITECTURE DECISION REQUIRED
```

## 5. Deterministic EventId and payload

`EventId` remains deterministic from immutable fact identity. The existing
stable hash/ID precedent is reused; mutable/current state and random UUIDs are
forbidden.

The typed requirement payload contains only the canonical requirement fact
dimensions already selected by R1-R1:

```text
tenant_id
execution_id
IntegrationConfigurationSubject
schema_version
```

The payload serializer must use canonical serialization with deterministic
field names and ordering. Equivalent typed facts must serialize to equivalent
payloads on every retry. The payload must not include:

- dict ordering as an implicit identity input;
- current configuration state;
- mutable worker metadata;
- fresh timestamps, UUIDs, or trace values;
- the full P2 configuration payload.

An altered payload under an already accepted `EventId` is an integrity
conflict, not a new duplicate acceptance.

## 6. Execution and attempt retry semantics

The requirement evidence remains tied to the original active execution
identity:

```text
task_id + run_id + attempt_id + execution_id
```

If a retry runs in a different worker or process, it reuses the original
immutable fact and therefore the original execution correlation. It must not
silently bind the event to the retry worker's current `attempt_id`, `run_id`,
or `execution_id`.

If the system creates a genuinely new `ExecutionId`, that is a new logical
obligation and receives a new requirement fact/event. It is not a retry of the
old obligation. A different attempt within the same logical execution does
not create a new requirement obligation; it repairs or re-accepts the original
fact using the original correlation identity.

No new deduplication store is introduced. The only mechanisms are:

- deterministic `EventId`;
- existing canonical `RuntimeEvent` persistence;
- `reconcile_idempotent_event_acceptance`.

## 7. Crash cases

### Case C — append failed before acceptance

```text
pin OK
→ requirement event append fails before acceptance
→ business I/O is blocked
→ retry reuses the same immutable fact and canonical RuntimeEvent
→ append accepts it for the first time
→ business I/O may continue
```

There is no ambiguity between first append and duplicate acceptance: no
accepted positioned event exists before the successful append.

### Case D — append accepted, crash before business I/O

```text
pin OK
→ requirement event persisted
→ crash before business I/O
→ retry same obligation
→ rebuild EXACT same RuntimeEvent
→ same EventId
→ reconcile_idempotent_event_acceptance()
→ original positioned event returned
→ no duplicate semantic fact
→ business I/O may continue
```

This is mechanically true only when every field in §2, especially timestamp,
payload, execution correlation, and metadata, comes from the same immutable
fact.

## 8. Mandatory durability classification

Corrected P4 must add the new requirement `RuntimeEventType` to every canonical
event metadata and payload registry used by production persistence:

```text
INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED
  = "integration_configuration_provenance_requirement_committed"
```

| Gate | Required condition |
|---|---|
| Event type catalog | New configured-provenance requirement type is present and has one canonical kind |
| Spine metadata registry | `event_category` and `ops_hint` resolve from the new type |
| Payload schema registry | Typed requirement payload schema is registered and validated |
| Persistence policy | The type is persistence-enabled |
| Retention | The effective retention is not `DEBUG` |
| Effective requirement | `evidence_persistence_requirement(event) == MANDATORY` exactly |

The corrected P4 test gate must assert the exact expression above for a
canonical requirement event. A mandatory persistence failure must fail closed
before business I/O.

## 9. Failure recorder distinction

`RuntimeEventExecutionFailureEvidenceRecorder` is a structural precedent only:

```text
typed request → RuntimeEventBus.record
```

It must not be copied as a fresh-event implementation. A failure recorder may
legitimately create a new event for each fresh failure occurrence. The
requirement recorder has a stronger contract: it must rebuild the same
canonical event for the same obligation, including timestamp, payload,
execution correlation, and all metadata.

## 10. Layering and ownership

The accepted ownership model is unchanged:

- Integrations owns materialize, validate, and P2 pin only.
- Runtime owns the physical recorder and persistence sequencing.
- The commit port is the sole handoff.
- P2 remains the pinning store.
- ExecutionReconstructor remains the reconstruction owner.
- `RuntimeEventBus.record` remains the shared runtime emission path.

No Integrations → runtime event implementation edge, second emitter, or new
deduplication/state mechanism is authorized by this reconciliation.

## 11. Exact corrected-P4 qualification tests

Corrected P4 must implement and pass all of the following:

1. **Same obligation, exact event equality** — build the requirement event,
   retry the same obligation, and assert `first_event == retry_event` across
   every field in §2, not only `EventId`.
2. **Same EventId, altered payload** — force the same `EventId` with a changed
   payload and assert `RuntimeEventPersistenceIntegrityError`.
3. **Same EventId, altered timestamp** — force the same `EventId` with a changed
   timestamp and assert `RuntimeEventPersistenceIntegrityError`.
4. **Same EventId, altered tenant/execution/subject** — force the same
   `EventId` while changing each of `tenant_id`, `execution_id`, or
   `IntegrationConfigurationSubject`, and assert integrity failure.
5. **Duplicate append returns original position** — append the canonical event,
   append the exact same event again, and assert the second call returns the
   original positioned event/position without a second semantic fact.
6. **Mandatory persistence fail-closed** — force requirement-event persistence
   failure, assert `evidence_persistence_requirement(event) == MANDATORY`,
   and assert business I/O count remains zero.
7. **Docker restart durability** — persist the requirement event, restart the
   Docker-backed runtime, retry the same obligation, and assert reconstruction
   retains exactly one semantic requirement fact and the original position.
8. **Crash Case C** — fail append before acceptance, assert no business I/O,
   retry the same immutable fact, and assert first successful append.
9. **Crash Case D** — accept append, crash before business I/O, retry, and
   assert exact event equality, original-position reconciliation, one semantic
   fact, then allowed business I/O.
10. **Registry coverage** — assert catalog entry, payload schema entry,
    persistence enabled, non-`DEBUG` retention, and exactly
    `evidence_persistence_requirement(event) == MANDATORY`.

Tests must use the production persistence path and Docker-backed restart where
specified; an in-memory-only proof is insufficient.

## 12. Tracker status

| ID | Status after this docs-only reconciliation |
|---|---|
| `TRACE-X-P5-R2-P4` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1` | **BLOCKED ON R1-R1** |
| `TRACE-X-P5-R2-P4-R1-R1` | **BLOCKED ON R1-R1-R1** |
| `TRACE-X-P5-R2-P4-R1-R1-R1` | **READY FOR AUDIT** |
| `FRZ-TRC-11` | **OPEN** |
| `P5` | **NOT ENTERED** |
| `CERT` | **NOT ENTERED** |

## 13. Independent audit notice

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
