# Vendor Knowledge — Reconciliation Finalization Architecture

**Task:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1`  
**Status:** `READY_FOR_REVIEW`  
**Branch:** `development`  
**Plan:** [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**Architecture:** [`../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)

---

## 1. Objective

Freeze a **provider-neutral** durable reconciliation-finalization architecture that can detect items **absent from a completed synchronized source inventory** without producing premature, non-deterministic, or globally misleading deletions.

This document is architecture only. No production code is introduced here.

---

## 2. Current defect (baseline on HEAD)

The one-page `VendorKnowledgeSyncCoordinator` applies durable effects in this order:

```text
1. sink batch
2. remote-item state batch
3. checkpoint commit
```

The completed sync checkpoint stores only:

- provider cursor;
- binding identity (`tenant_id`, `binding_id`, `binding_configuration_version`).

The remote-item state repository does **not** own a durable reconciliation-run lifecycle.

Therefore a naive implementation — *load active IDs, subtract seen IDs, emit missing tombstones on the final page* — is **unsafe**.

### Failure window (must be eliminated)

```text
1. final reconciliation page is read
2. missing-item tombstones are computed
3. sink accepts the batch
4. item states are updated (missing items → DELETED)
5. checkpoint commit fails
6. caller retries with restart=True
```

A fresh scan of active item states now sees a **different baseline** because step 4 already mutated state. The retry can emit a **different tombstone set** and a **different delivery batch**.

The architecture below eliminates this failure mode by separating reconciliation-run durability from the ordinary completed sync checkpoint and by freezing page intent before any sink or item-state side effect.

---

## 3. Frozen architecture decision

Introduce a **provider-neutral durable reconciliation-run state machine** that is separate from the ordinary completed sync checkpoint.

The coordinator continues to process **at most one provider page per call**. The source lease still protects **one coordinator call**, not the entire multi-page traversal. The durable run state protects **continuity between calls**.

### 3.1 Phases

| Phase | Meaning |
|---|---|
| `COLLECTING` | Run exists; provider pages may be read; candidate inventory may be loaded once at run start; no prepared page side effects. |
| `PAGE_PREPARED` | Exact page intent is durably frozen **before** sink or remote-item state mutation. |
| `FINALIZING` | Final page sink and item-state application succeeded; only completed-checkpoint CAS and run completion remain. |
| `COMPLETED` | Ordinary completed sync checkpoint committed successfully. |

#### `COLLECTING`

The run owns:

- `tenant_id`;
- `binding_id`;
- `binding_configuration_version`;
- `provider_id` and `source_kind` (source identity);
- opaque `run_id`;
- base completed checkpoint **or** its exact CAS identity (`expected_base_checkpoint_cas`);
- current provider input cursor (nullable at run start);
- bounded remaining candidate remote IDs (see §5);
- **no** prepared page side effects.

Entry:

- authorized new reconciliation when no active run exists; or
- resume of an existing active run in `COLLECTING`.

#### `PAGE_PREPARED`

Before calling the sink, the coordinator durably prepares the exact page intent.

**Prepared intent — minimum required fields:**

| Field | Purpose |
|---|---|
| `run_id` | binds intent to one reconciliation run |
| `input_cursor_fingerprint` | identifies provider input cursor without leaking opaque payload |
| `provider_page_fingerprint` | canonical fingerprint of the fetched provider page |
| `proposed_checkpoint_fingerprint` | resulting checkpoint identity for this page |
| `next_cursor_fingerprint` | continuation identity when `has_more=true` |
| `has_more` | whether more provider pages remain |
| `delivery_id` | frozen deterministic delivery identity for this page |
| `remaining_candidate_remote_ids` | candidate IDs after subtracting all remote IDs present on this provider page |
| `synthetic_tombstone_remote_ids` | exact **sorted** synthetic tombstone remote IDs; **empty** for non-final pages |
| `binding_configuration_version` | configuration guard |

**Safety constraints for prepared intent storage:**

- no credentials, tokens, or secret material;
- no raw provider continuation URLs outside existing opaque cursor objects already stored in checkpoint/run state;
- no full fetched content bodies or attachment bytes.

**Final page rule:** a page with `has_more=false` **must** reach `PAGE_PREPARED` before sink or remote-item state side effects occur. Synthetic tombstones are computed and stored in the prepared intent on that page only.

**Retry in `PAGE_PREPARED` must:**

- load the same stored run and input cursor;
- reproduce and verify the same `provider_page_fingerprint` (re-read provider page only to verify; see failure matrix);
- reuse the exact stored `synthetic_tombstone_remote_ids`;
- reuse the same `delivery_id`;
- **fail closed** when the provider page no longer matches the prepared intent.

#### `FINALIZING`

Entered only after **all** of the following:

- sink accepted the prepared batch;
- remote-item state repository idempotently accepted the same `delivery_id`;
- the prepared page had `has_more=false` (final provider page).

`FINALIZING` retains:

- exact final provider checkpoint cursor value;
- `expected_previous_completed_checkpoint_cas` (the checkpoint CAS identity observed at final page preparation);
- final `delivery_id`;
- `run_id`.

**Retry in `FINALIZING` must:**

- **not** read the provider again;
- **not** resend a different tombstone batch;
- only finish completed-checkpoint CAS and run-state transition to `COMPLETED`.

#### `COMPLETED`

The ordinary completed sync checkpoint was committed successfully with CAS against `expected_previous_completed_checkpoint_cas`.

### 3.2 Active-run identity

Exactly **one active reconciliation run** per:

```text
(tenant_id, binding_id, binding_configuration_version)
```

Active means phase ∈ `{COLLECTING, PAGE_PREPARED, FINALIZING}`.

The active run is stored in a dedicated reconciliation-run repository slot distinct from:

- `KnowledgeSyncCheckpoint` (completed sync cursor only);
- per-item `KnowledgeRemoteItemState`;
- per-delivery markers.

`run_id` is an opaque stable identifier assigned at run creation and never reused across superseded runs.

### 3.3 Cleanup and retention policy (frozen)

**Policy:** single active-slot replacement with supersession pointer.

- One durable active-slot document per `(tenant_id, binding_id, binding_configuration_version)`.
- On transition to `COMPLETED`, the slot remains in `COMPLETED` until the next authorized reconciliation start.
- Starting a genuinely new reconciliation CAS-replaces the slot with a new `COLLECTING` run. The replacement record carries `superseded_run_id` pointing at the prior `run_id` for audit correlation only.
- `PAGE_PREPARED` payload and `FINALIZING` payload are cleared from the active slot on `COMPLETED` transition. They are not retained in the active slot after completion.
- Historical prepared intents are not archived in v1. Idempotent replay relies on delivery markers and the completed checkpoint, not long-term prepared-intent retention.
- Delivery markers follow the existing remote-item state repository retention semantics (idempotent replay window); reconciliation-run implementation must not shorten that guarantee.

### 3.4 Restart and interleaving semantics (frozen)

| Rule | Behavior |
|---|---|
| Source lease | Protects one coordinator call only. |
| Durable run state | Protects continuity across calls for one reconciliation traversal. |
| Active run present | Incremental `sync_once` for the same binding **must not** interleave. |
| Active run present | Another reconciliation invocation **must resume** the active run (`restart=False` semantics internally when active run exists). |
| `restart=True` with active run | **Fail closed** when the active run may already have side effects (`PAGE_PREPARED` or `FINALIZING`). Allowed only when active run is absent or implementation proves zero side effects (not the case after sink/state mutation). |
| New run start | Allowed only when no active run exists, or previous run is `COMPLETED` and is replaced through the CAS supersession rule in §3.3. |
| Binding configuration version mismatch | **Fail closed.** Do not resume or finalize against a different `binding_configuration_version`. |

### 3.5 Tombstone semantics (frozen)

Reconciliation-generated synthetic tombstones **do not** prove global provider deletion.

They mean only:

```text
absent_from_completed_synchronized_source_inventory
```

Provider-originated tombstones (explicit provider deletion signals, e.g. Graph `deletedDateTime`, mail/drive `REMOVED` delta entries) and reconciliation-generated synthetic tombstones **must remain distinguishable** in coordinator envelopes:

- reconciliation synthetic tombstones use a dedicated change kind or equivalent source-scoped marker defined in implementation task `1B`;
- provider tombstones retain existing provider-scoped semantics already frozen per adapter family.

Both classes share the platform rule: tombstones are **source-inventory scoped**, never global provider deletion proof unless the adapter contract explicitly documents provider-global semantics (primary-calendar delta removed entries remain view-scoped).

### 3.6 Final completed-checkpoint CAS rule

Completed checkpoint commit occurs **only** in `FINALIZING` after sink and item-state success for the final page.

CAS inputs:

- `new_checkpoint` with final provider cursor;
- `expected_previous=expected_previous_completed_checkpoint_cas` captured when the final page entered `PAGE_PREPARED`.

Non-final pages **must not** advance the completed sync checkpoint. They may update only run state (`remaining_candidate_remote_ids`, run cursor, phase).

---

## 4. Bounded reconciliation candidate inventory

`DocumentStore.query(partition_key, row_key_prefix=..., limit=...)` supports prefix scans with a limit but exposes **no generic continuation contract**. The architecture therefore defines a **bounded reconciliation candidate inventory** rather than an unlimited or hidden backend scan.

### Frozen rules

| Rule | Requirement |
|---|---|
| Maximum scan | Explicit configurable `max_reconciliation_candidate_count` per platform default and/or binding policy. |
| Ordering | Deterministic: `remote_id` ascending (UTF-8 lexicographic). |
| Filter | `tenant_id`, `binding_id`, `binding_configuration_version`, `status=ACTIVE` only. |
| Over-limit | **Fail closed** when repository holds more ACTIVE records than the configured bound. |
| Partial baseline | **Forbidden.** Either the full bounded inventory loads or the run fails before any provider page read. |
| Provider read after limit failure | **Forbidden.** |
| Future extension | Architecture reserves a `candidate_inventory_continuation_token` field on the run for a later paged inventory task; field is unused in v1 and must remain null. |

Candidate inventory is loaded **once** when entering `COLLECTING` at run start (or restored from run state on resume). Each provider page subtracts seen remote IDs from `remaining_candidate_remote_ids`. On the final page, `synthetic_tombstone_remote_ids = sorted(remaining_candidate_remote_ids)` after subtraction.

---

## 5. Provider-page fingerprint

Canonical SHA-256 over JSON with sorted keys:

```text
input_cursor_fingerprint
has_more
proposed_checkpoint_fingerprint
next_cursor_fingerprint
ordered_changes: [
  { remote_id, change_kind, revision_fingerprint }
]
```

Uses the same revision fingerprint approach as the existing delivery-id builder. No raw cursor payloads, URLs, tokens, or content in the fingerprint input.

---

## 6. Delivery-ID contract (frozen)

The reconciliation delivery ID is `SHA-256(canonical_json)` and **must** bind to:

| Input | Included |
|---|---|
| `tenant_id` | yes |
| `binding_id` | yes |
| `binding_configuration_version` | yes |
| `mode` | `reconciliation` |
| `run_id` | yes |
| `input_cursor_fingerprint` | yes |
| `provider_page_fingerprint` | yes |
| `ordered_provider_changes` | exact order from provider page |
| `synthetic_tombstone_remote_ids` | exact sorted list (empty when not final) |
| `proposed_checkpoint_fingerprint` | yes |
| `next_cursor_fingerprint` | yes |

**Must not appear** in delivery-id inputs or public prepared-intent objects:

- raw provider URLs;
- credentials, access tokens, refresh tokens;
- signed URLs;
- attachment bytes;
- event/message body content.

The delivery ID is assigned in `PAGE_PREPARED` and is immutable for that page intent.

---

## 7. Page and failure matrix

Legend:

- **Phase** — durable reconciliation-run phase at retry entry.
- **Side effects** — durable mutations that may already exist.
- **Retry reads** — durable state consulted on retry.
- **Provider** — whether the provider is read again.
- **Sink** — whether sink may be invoked again.
- **Delivery ID** — same / new / none.
- **Checkpoint** — whether completed checkpoint may advance.
- **Error** — expected safe error category.

| # | Case | Phase | Side effects may exist | Retry reads | Provider | Sink | Delivery ID | Checkpoint | Error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | First page, more pages remain | `COLLECTING` → `PAGE_PREPARED` | none before prepare | run, checkpoint, candidates | yes (verify if prepared) | only after prepare | same once prepared | no | — |
| 2 | Intermediate page | `COLLECTING` / `PAGE_PREPARED` | prior pages committed checkpoint only via final page; run cursor updated | run, prepared intent if stuck | yes in `COLLECTING`; verify in `PAGE_PREPARED` | after prepare | same once prepared | no | — |
| 3 | Final page, no missing items | `PAGE_PREPARED` → `FINALIZING` | sink + state for final page | run, prepared intent, `FINALIZING` | no after `FINALIZING` | idempotent in `PAGE_PREPARED` retry | same | yes in `FINALIZING` | — |
| 4 | Final page with missing items | `PAGE_PREPARED` → `FINALIZING` | sink + state including synthetic tombstones | run, prepared intent, frozen tombstone list | no after `FINALIZING` | idempotent in `PAGE_PREPARED` retry | same | yes in `FINALIZING` | — |
| 5 | Provider page read failure | `COLLECTING` | none | run | yes on retry | no | none | no | `DEPENDENCY_UNAVAILABLE`, retryable |
| 6 | Content materialization failure | `COLLECTING` | none | run | yes on retry | no | none | no | `DEPENDENCY_UNAVAILABLE`, retryable |
| 7 | `PAGE_PREPARED` CAS failure | `COLLECTING` | none | run | yes | no | none | no | `DEPENDENCY_UNAVAILABLE`, retryable |
| 8 | Sink failure | `PAGE_PREPARED` | prepared intent only | prepared intent | verify only | yes retry | same | no | `DEPENDENCY_UNAVAILABLE`, retryable |
| 9 | Partial / completed item-state then failure | `PAGE_PREPARED` | sink ok; state partial or complete | prepared intent, delivery marker | verify only | idempotent | same | no | `DEPENDENCY_UNAVAILABLE`, retryable |
| 10 | Transition from `PAGE_PREPARED` failure | `PAGE_PREPARED` | prepared intent | prepared intent | verify; fail closed on mismatch | after verify | same | no | mismatch → `INVALID_PROVIDER_RESPONSE`, not retryable |
| 11 | Final completed-checkpoint CAS failure | `FINALIZING` | sink + state complete | `FINALIZING` payload | **no** | **no** | same | retry CAS only | `DEPENDENCY_UNAVAILABLE`, retryable |
| 12 | Process crash after durable boundary | any | up to last committed boundary | phase-appropriate | per phase | per phase | per phase | only in `FINALIZING` | resume by phase |
| 13 | Retry with `restart=True` | active run with side effects | sink/state possible | active run | **fail closed** | **no** | — | no | `INVALID_CURSOR`, not retryable |
| 14 | Retry with `restart=False` | active phase | per phase | active run | per phase | per phase | same when prepared | per phase | — |
| 15 | Stale binding configuration | any | — | binding + run | no | no | — | no | `INVALID_CURSOR`, not retryable |
| 16 | Concurrent call, lease busy | — | — | lease | no | no | none | no | `LEASE_BUSY` |
| 17 | Incremental sync while reconciliation active | active run | — | active run guard | no | no | none | no | `INVALID_CURSOR`, not retryable |
| 18 | Candidate inventory limit exceeded | `COLLECTING` | none | candidates | **no** | no | none | no | `INVALID_PROVIDER_RESPONSE`, not retryable |

### Crash boundaries

| Last committed boundary | Resume phase | Behavior |
|---|---|---|
| run created in `COLLECTING` | `COLLECTING` | continue from stored cursor and candidates |
| `PAGE_PREPARED` written | `PAGE_PREPARED` | verify provider page; replay sink/state with same delivery ID |
| sink accepted | `PAGE_PREPARED` | replay state if needed; same delivery ID |
| item-state accepted | `PAGE_PREPARED` or `FINALIZING` | if final page → enter/gcontinue `FINALIZING`; else advance run to `COLLECTING` with updated candidates |
| `FINALIZING` written | `FINALIZING` | checkpoint CAS only |
| checkpoint committed | `COMPLETED` | ordinary incremental sync allowed |

---

## 8. Implementation decomposition (later tasks)

Architecture only — implementation is split as follows.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`

Contracts and durable repositories:

- reconciliation-run models (`COLLECTING`, `PAGE_PREPARED`, `FINALIZING`, `COMPLETED`);
- repository protocol and DocumentStore implementation;
- bounded active candidate inventory loader;
- CAS transitions and corruption/configuration error boundaries;
- repository and model tests.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B`

Coordinator integration:

- start/resume rules and `restart` semantics;
- `PAGE_PREPARED` intent write before sink;
- deterministic delivery ID per §6;
- synthetic source-scoped tombstones on final page only;
- `FINALIZING` recovery without provider re-read;
- incremental-sync blocking while active run exists;
- failure-window tests covering §7.

### `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1`

Calendar acceptance proof after shared finalization lands:

- non-primary Calendar missing-item detection;
- no tombstones before the final snapshot page;
- retry after sink / state / checkpoint failures with deterministic batches;
- same integration instance;
- no raw continuation leakage;
- attachment inventory completeness flags;
- Calendar status restored to `READY_FOR_REVIEW` only after proof passes.

---

## 9. Roadmap linkage

| Item | Status after this task |
|---|---|
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1` | `READY_FOR_REVIEW` |
| `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` | `CHANGES_REQUIRED` — blocked by shared reconciliation-finalization implementation |
| Microsoft Graph adapter family (`MSGRAPH-KNOWLEDGE-ADAPTERS-1`) | `IN_PROGRESS` |
| Google Workspace knowledge workstream | independent; does not gate Calendar finalization |

**Next task after architecture acceptance:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`.

Missing-item detection for non-primary Calendar reconciliation is **not** implemented on HEAD. Calendar adapter code exists but durable finalization semantics required for safe missing-item tombstones are **not** present until `1A` + `1B` land.

---

## 10. Non-goals (this architecture task)

- production coordinator changes;
- Calendar adapter changes;
- Google Workspace, LKW Conversation Context, or unrelated provider work;
- paged candidate inventory implementation (extension point only);
- claiming reconciliation tombstones prove global provider deletion.
