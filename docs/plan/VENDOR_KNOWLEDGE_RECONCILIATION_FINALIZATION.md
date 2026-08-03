# Vendor Knowledge — Reconciliation Finalization Architecture

**Task:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1`  
**Status:** `CHANGES_REQUIRED` — correction under review
**Review fix:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1`
**Review-fix status:** `READY_FOR_REVIEW`
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

#### Successful path

| Phase | Meaning |
|---|---|
| `COLLECTING` | Run exists; provider pages may be read; candidate inventory may be loaded once at run start; no prepared page side effects. |
| `PAGE_PREPARED` | Exact page intent is durably frozen **before** sink or remote-item state mutation. |
| `FINALIZING` | Final page sink and item-state application succeeded; only completed-checkpoint CAS and run completion remain. |
| `COMPLETED` | Ordinary completed sync checkpoint committed successfully. |

#### Recovery and terminal states (not on the successful path)

| Phase | Meaning |
|---|---|
| `RECOVERY_REQUIRED` | Fail-closed state when automatic deterministic continuation cannot be proven safe. |
| `ABORTED` | Durable terminal state proving the run was safely abandoned. |

#### `COLLECTING`

The run owns:

- `tenant_id`;
- `binding_id`;
- `binding_configuration_version` (immutable field inside the run);
- `provider_id` and `source_kind` (source identity);
- opaque `run_id`;
- base completed checkpoint **or** its exact CAS identity (`expected_base_checkpoint_cas`);
- current provider input cursor (nullable at run start);
- bounded remaining candidate remote IDs (see §4);
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
| `prepared_batch_fingerprint` | SHA-256 of exact materialized sink batch manifest (see §5) |
| `prepared_state_mutations` | bounded exact item-state mutations (see §6) |
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
- no full fetched content bodies or attachment bytes;
- no event bodies, message bodies, or provider payloads in `prepared_state_mutations`.

**Final page rule:** a page with `has_more=false` **must** reach `PAGE_PREPARED` before sink or remote-item state side effects occur. Synthetic tombstones are computed and stored in the prepared intent on that page only.

**Retry in `PAGE_PREPARED`** follows the frozen receipt-driven decision table in §10. Provider re-read is **not** required on every retry.

#### `FINALIZING`

Entered only after **all** of the following:

- sink accepted the prepared batch;
- remote-item state repository idempotently accepted the same `delivery_id`;
- the prepared page had `has_more=false` (final provider page).

`FINALIZING` retains:

- exact final provider checkpoint cursor value;
- `expected_previous_completed_checkpoint` (complete canonical checkpoint identity observed at final page preparation);
- `intended_final_completed_checkpoint` (complete canonical checkpoint identity to commit);
- `intended_final_checkpoint_fingerprint`;
- final `delivery_id`;
- `prepared_batch_fingerprint`;
- `run_id`.

**Retry in `FINALIZING`** follows §12. Checkpoint commit and run completion are **separate durable boundaries**.

#### `COMPLETED`

The ordinary completed sync checkpoint was committed successfully with CAS against `expected_previous_completed_checkpoint`.

#### `RECOVERY_REQUIRED`

A durable fail-closed state used when automatic deterministic continuation cannot be proven safe.

Examples:

- prepared provider page no longer matches;
- prepared batch no longer matches;
- delivery receipt reports a conflicting fingerprint;
- completed checkpoint is neither the expected previous checkpoint nor the exact intended final checkpoint;
- binding configuration changed while an active run exists;
- durable run state or delivery state is internally inconsistent.

While in `RECOVERY_REQUIRED`:

- incremental synchronization is blocked;
- normal reconciliation continuation is blocked;
- a new run is blocked;
- the run must not be silently deleted or replaced;
- only an explicit recovery operation with exact run identity and CAS version may act (see §13).

#### `ABORTED`

A durable terminal state proving that the run was safely abandoned.

A run may become `ABORTED` only when one of these conditions is proven:

1. the run is still `COLLECTING` and no prepared page exists;
2. the run is `PAGE_PREPARED`, the item-state delivery marker is absent, the sink delivery receipt is `ABSENT`, and no sink or state side effect exists;
3. an explicit recovery procedure has repaired or compensated all effects and recorded an auditable resolution receipt.

A plain force-delete is forbidden.

Starting a new reconciliation run is allowed after `ABORTED` through CAS replacement.

### 3.2 Active-run identity

Exactly **one active reconciliation run** per:

```text
(tenant_id, binding_id)
```

`binding_configuration_version` remains a required immutable field **inside** the run record. It is **not** part of the active-slot key.

Active means phase ∈ `{COLLECTING, PAGE_PREPARED, FINALIZING, RECOVERY_REQUIRED}`.

The active run is stored in a dedicated reconciliation-run repository slot distinct from:

- `KnowledgeSyncCheckpoint` (completed sync cursor only);
- per-item `KnowledgeRemoteItemState`;
- per-delivery markers.

`run_id` is an opaque stable identifier assigned at run creation and never reused across superseded runs.

**Configuration version behavior:**

- exactly one active reconciliation run across all configuration versions of one binding;
- a configuration change does not create a parallel slot;
- when an active run contains an older configuration version than the current binding, normal processing **fails closed** into `RECOVERY_REQUIRED`;
- a `COMPLETED` or safely `ABORTED` run may be CAS-replaced by a new run using the current configuration version;
- incremental synchronization for the same binding remains blocked while the slot is in `COLLECTING`, `PAGE_PREPARED`, `FINALIZING`, or `RECOVERY_REQUIRED`;
- `COMPLETED` and `ABORTED` may be superseded through the frozen CAS rule in §3.3.

### 3.3 Cleanup and retention policy (frozen)

**Policy:** single active-slot replacement with supersession pointer.

- One durable active-slot document per `(tenant_id, binding_id)`.
- Terminal `COMPLETED` and `ABORTED` records remain until CAS-superseded by a new run.
- `RECOVERY_REQUIRED` is **never** automatically superseded.
- On transition to `COMPLETED` or proven-safe `ABORTED`, the slot remains in that terminal phase until the next authorized reconciliation start.
- Starting a genuinely new reconciliation CAS-replaces the slot with a new `COLLECTING` run. The replacement record carries `superseded_run_id` pointing at the prior `run_id` for audit correlation only.
- `PAGE_PREPARED` and `FINALIZING` payloads may be cleared only after `COMPLETED` or proven-safe `ABORTED`.
- Delivery receipts must outlive any active run that references them.
- A new run must not shorten the existing sink or item-state idempotency window.
- Historical prepared intents are not archived in v1. Idempotent replay relies on delivery receipts and the completed checkpoint, not long-term prepared-intent retention.

### 3.4 Restart and interleaving semantics (frozen)

| Rule | Behavior |
|---|---|
| Source lease | Protects one coordinator call only. |
| Durable run state | Protects continuity across calls for one reconciliation traversal. |
| Active run present | Incremental `sync_once` for the same binding **must not** interleave. |
| Active run present | Another reconciliation invocation **must resume** the active run (`restart=False` semantics internally when active run exists). |
| `restart=True` with active run | **Fail closed** when the active run may already have side effects (`PAGE_PREPARED`, `FINALIZING`, or `RECOVERY_REQUIRED`). Allowed only when active run is absent or implementation proves zero side effects. |
| New run start | Allowed only when no active run exists, or previous run is `COMPLETED` or `ABORTED` and is replaced through the CAS supersession rule in §3.3. |
| Binding configuration version mismatch on active run | **Fail closed** → `RECOVERY_REQUIRED`. Do not resume or finalize against a different `binding_configuration_version`. |

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

Synthetic tombstones are represented as envelopes in the `prepared_batch_fingerprint` manifest with exact sorted synthetic tombstone IDs and their source-scoped semantic marker.

### 3.6 Final completed-checkpoint CAS rule

Completed checkpoint commit occurs **only** in `FINALIZING` after sink and item-state success for the final page.

CAS inputs:

- `new_checkpoint` with final provider cursor;
- `expected_previous=expected_previous_completed_checkpoint` captured when the final page entered `PAGE_PREPARED`.

Non-final pages **must not** advance the completed sync checkpoint. They may update only run state (`remaining_candidate_remote_ids`, run cursor, phase).

Checkpoint commit success and run transition to `COMPLETED` are **separate durable boundaries**. A successful checkpoint commit does **not** automatically imply the run is `COMPLETED`.

---

## 4. Bounded reconciliation candidate inventory

`DocumentStore.query(partition_key, row_key_prefix=..., limit=...)` supports prefix scans with a limit but exposes **no generic continuation contract**. The architecture therefore defines a **bounded reconciliation candidate inventory** rather than an unlimited or hidden backend scan.

### Frozen rules

| Rule | Requirement |
|---|---|
| Maximum count | Explicit configurable `max_reconciliation_candidate_count` per platform default and/or binding policy. |
| Maximum payload bytes | Explicit configurable `max_reconciliation_candidate_payload_bytes` per platform default and/or binding policy. |
| Maximum remote-ID bytes | Explicit configurable `max_reconciliation_remote_id_bytes` per platform default and/or binding policy. |
| Remote-ID validation | Each UTF-8 encoded remote ID validated before run creation; must be within `max_reconciliation_remote_id_bytes`. |
| Ordering | Deterministic: `remote_id` ascending (UTF-8 lexicographic). |
| Filter | `tenant_id`, `binding_id`, `binding_configuration_version`, `status=ACTIVE` only. |
| Over-limit detection | **Fail closed** when repository holds more ACTIVE records than the configured count bound. Use a count-plus-one or equivalent explicit over-limit probe rather than assuming the backend limit is complete. |
| Payload bound | Canonical serialized candidate inventory must fit within `max_reconciliation_candidate_payload_bytes`. |
| Partial baseline | **Forbidden.** Either the full bounded inventory loads or the run fails before any provider page read. |
| Provider read after limit failure | **Forbidden.** |
| Candidate mutation bound | The same byte bound applies after each candidate-list mutation written to the run slot. |
| Future extension | Architecture reserves a `candidate_inventory_continuation_token` field on the run for a later paged inventory task; field is unused in v1 and must remain null. |

Count overflow, payload byte overflow, or single remote-ID byte overflow fails **before** run creation, provider page read, sink call, or item-state mutation.

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

`provider_page_fingerprint` proves the provider page was read consistently. It does **not** prove the exact materialized sink batch.

---

## 6. Prepared batch fingerprint

`prepared_batch_fingerprint` is a SHA-256 digest of canonical JSON containing the exact logical batch manifest.

The manifest must include:

```text
tenant_id
binding_id
binding_configuration_version
mode
run_id
delivery_id
source identity
has_more
ordered envelopes
prepared_state_mutations fingerprint
```

Each ordered envelope fingerprint entry must include:

```text
change_kind
remote_id
descriptor fingerprint or null
content fingerprint or null
permissions fingerprint or null
```

### Descriptor fingerprint

Fingerprint canonical public descriptor data required by the sink:

```text
identity
revision
item_type
title
content mode
content availability
metadata
provenance
safe locator
web URL when present in an adapter contract
```

Do not include Python object representations.

### Content fingerprint

Bind to:

```text
mode
mime type
content_hash
structured schema identity when present
```

When the canonical content model contains no trusted `content_hash`, the implementation must hash the canonical content payload before preparation.

### Permissions fingerprint

Bind to the canonical permissions model or its canonical ACL hash.

### Synthetic tombstones

The exact sorted synthetic tombstone IDs and their source-scoped semantic marker must be represented as envelopes in the batch manifest.

---

## 7. Prepared state mutations

`PAGE_PREPARED` must store a bounded, safe and exact `prepared_state_mutations`.

Each mutation must contain only data required to finish the existing remote-item state application:

```text
remote_id
resulting status
revision or null
delivery_id
binding configuration version
```

Do not store event bodies, message bodies, attachment bytes, credentials or provider payloads.

The fingerprint of `prepared_state_mutations` must be included in `prepared_batch_fingerprint` and `delivery_id`. This allows item-state application to resume without re-reading provider content after the sink has already accepted the delivery.

---

## 8. Delivery-ID contract (frozen)

The reconciliation delivery ID is `SHA-256(canonical_json)` and **must** bind to:

| Input | Included |
|---|---|
| `tenant_id` | yes |
| `binding_id` | yes |
| `binding_configuration_version` | yes |
| `mode` | `reconciliation` |
| `run_id` | yes |
| `prepared_batch_fingerprint` | yes |
| `prepared_state_mutations` fingerprint | yes |
| `input_cursor_fingerprint` | yes |
| `provider_page_fingerprint` | yes |
| `ordered_provider_changes` | exact order from provider page |
| `synthetic_tombstone_remote_ids` | exact sorted list (empty when not final) |
| `proposed_checkpoint_fingerprint` | yes |
| `next_cursor_fingerprint` | yes |

The delivery ID must not be computed from a weaker representation than the batch sent to the sink.

**Must not appear** in delivery-id inputs or public prepared-intent objects:

- raw provider URLs;
- credentials, access tokens, refresh tokens;
- signed URLs;
- attachment bytes;
- event/message body content.

The delivery ID is assigned in `PAGE_PREPARED` and is immutable for that page intent.

---

## 9. Delivery receipt inspection

The architecture requires inspectable idempotency receipts at both durable boundaries.

### Sink delivery receipt

States:

```text
ABSENT
APPLIED
CONFLICT
UNKNOWN
```

Inspection input:

```text
tenant_id
binding_id
delivery_id
prepared_batch_fingerprint
```

Semantics:

- `ABSENT`: the sink proves the delivery has not been applied;
- `APPLIED`: the sink proves the exact fingerprint was accepted;
- `CONFLICT`: the same delivery ID exists with a different fingerprint;
- `UNKNOWN`: the sink cannot establish a safe result.

`CONFLICT` and `UNKNOWN` transition the run to `RECOVERY_REQUIRED`.

### Item-state delivery receipt

States:

```text
ABSENT
APPLYING
COMPLETED
CONFLICT
```

The receipt must bind to:

```text
delivery_id
prepared state-mutation fingerprint
```

The architecture may extend the existing delivery-marker repository contract in implementation task `1A`.

Do not require a provider re-read when item-state receipt is `APPLYING` or `COMPLETED`.

---

## 10. `PAGE_PREPARED` retry decision table (frozen)

### Item-state receipt `COMPLETED`

The sink necessarily succeeded earlier because sink application precedes item-state application.

Therefore:

- do not read the provider;
- do not call the sink;
- use the frozen prepared intent;
- for a non-final page, CAS the run back to `COLLECTING`;
- for a final page, CAS the run to `FINALIZING`.

### Item-state receipt `APPLYING`

- do not read the provider;
- do not call the sink again;
- replay the exact stored `prepared_state_mutations`;
- reuse the same delivery ID;
- then perform the phase transition.

### Item-state receipt `ABSENT`

Inspect the sink receipt.

#### Sink receipt `APPLIED`

- do not read the provider;
- do not resend the sink batch;
- apply the exact stored state mutations;
- continue the phase transition.

#### Sink receipt `ABSENT`

No durable side effect exists.

The coordinator may re-read and rematerialize the provider page using the frozen input cursor.

It must verify both:

```text
provider_page_fingerprint
prepared_batch_fingerprint
```

If both match:

- call the sink with the same delivery ID;
- apply the frozen state mutations;
- continue.

If either differs:

- do not call the sink;
- transition to `RECOVERY_REQUIRED`.

#### Sink receipt `CONFLICT` or `UNKNOWN`

Transition to `RECOVERY_REQUIRED`.

### Item-state receipt `CONFLICT`

Transition to `RECOVERY_REQUIRED`.

---

## 11. `FINALIZING` checkpoint idempotency (frozen)

On retry, read the current completed checkpoint. Compare the **complete canonical checkpoint identity**, not cursor version alone.

### Case A — current checkpoint equals expected previous

Perform CAS:

```text
expected previous → intended final
```

Then CAS the run to `COMPLETED`.

### Case B — current checkpoint equals intended final

The checkpoint commit succeeded before the process crashed.

Treat the checkpoint operation as already complete.

Do not:

- read the provider;
- call the sink;
- apply item states;
- attempt the old CAS again.

Only CAS the run from `FINALIZING` to `COMPLETED`.

### Case C — current checkpoint matches neither value

Transition to `RECOVERY_REQUIRED`.

Do not overwrite the current checkpoint.

---

## 12. Operator recovery (frozen)

A narrow provider-neutral recovery interface for future implementation.

Every recovery command must require:

```text
tenant_id
binding_id
expected_run_id
expected_run_record_version
expected_phase
operator reason code
```

### Allowed v1 recovery actions

#### `RESUME_EXACT`

Allowed when stored receipts and fingerprints prove that the existing prepared intent remains reproducible.

Resumes the existing run only. Must not create a new run ID.

#### `FINALIZE_ALREADY_COMMITTED`

Allowed when:

```text
phase == FINALIZING
current completed checkpoint == intended final checkpoint
```

Only marks the run `COMPLETED`.

#### `ABORT_PRISTINE`

Allowed only when receipt inspection proves:

```text
no sink delivery
no item-state delivery
no completed checkpoint mutation belonging to the run
```

Transitions the run to `ABORTED`.

#### `REPAIR_REQUIRED`

Not an automatic mutation action.

Records that manual data repair or compensation is required.

The binding remains blocked until a separately reviewed repair procedure proves consistency.

### Forbidden recovery operations

```text
force-delete active run
ignore delivery conflict
replace checkpoint unconditionally
start new run while side effects are unknown
mark COMPLETED without checkpoint proof
mark ABORTED without pristine-side-effect proof
```

Fail-closed blocking is preferable to silently unlocking an inconsistent binding.

---

## 13. Error categories (frozen)

| Condition | Category | Retryable | Run transition |
|---|---|---|---|
| lease busy | `LEASE_BUSY` | yes | — |
| provider dependency failure before preparation | `DEPENDENCY_UNAVAILABLE` | yes | — |
| CAS conflict with no contradictory durable state | `DEPENDENCY_UNAVAILABLE` | yes | — |
| prepared page or batch mismatch | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` |
| stale binding configuration | `INVALID_CURSOR` | no | → `RECOVERY_REQUIRED` |
| candidate count or payload policy exceeded | `CONFIGURATION_ERROR` | no | — |
| delivery fingerprint conflict | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` |
| unknown delivery outcome | `DEPENDENCY_UNAVAILABLE` or dedicated recovery mapping | — | remains `RECOVERY_REQUIRED` |
| completed checkpoint differs from both expected and intended | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` |

Administrator-configured candidate limits are **not** classified as malformed provider responses.

---

## 14. Page and failure matrix

Legend:

- **Phase** — durable reconciliation-run phase at retry entry.
- **Receipts** — durable delivery receipts that may exist.
- **Provider** — whether the provider is read again.
- **Sink** — whether sink may be invoked again.
- **State** — whether item-state mutations may be replayed.
- **Delivery ID** — same / new / none.
- **Checkpoint** — whether completed checkpoint may advance.
- **Next** — next phase after successful retry.
- **Error** — expected safe error category.

| # | Case | Phase | Receipts | Provider | Sink | State | Delivery ID | Checkpoint | Next | Error |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | First page, more pages remain | `COLLECTING` → `PAGE_PREPARED` | none | yes | only after prepare | after sink | same once prepared | no | `COLLECTING` or `PAGE_PREPARED` | — |
| 2 | Intermediate page | `COLLECTING` / `PAGE_PREPARED` | per §10 | per §10 | per §10 | per §10 | same once prepared | no | per §10 | per §10 |
| 3 | Final page, no missing items | `PAGE_PREPARED` → `FINALIZING` | sink APPLIED + state COMPLETED | no | no | no | same | yes in `FINALIZING` | `FINALIZING` → `COMPLETED` | — |
| 4 | Final page with missing items | `PAGE_PREPARED` → `FINALIZING` | sink APPLIED + state COMPLETED | no | no | no | same | yes in `FINALIZING` | `FINALIZING` → `COMPLETED` | — |
| 5 | Provider page read failure | `COLLECTING` | none | yes on retry | no | no | none | no | `COLLECTING` | `DEPENDENCY_UNAVAILABLE`, retryable |
| 6 | Content materialization failure | `COLLECTING` | none | yes on retry | no | no | none | no | `COLLECTING` | `DEPENDENCY_UNAVAILABLE`, retryable |
| 7 | `PAGE_PREPARED` CAS failure | `COLLECTING` | none | yes | no | no | none | no | `COLLECTING` | `DEPENDENCY_UNAVAILABLE`, retryable |
| 8 | Sink failure | `PAGE_PREPARED` | sink ABSENT | per §10 | per §10 | per §10 | same | no | per §10 | `DEPENDENCY_UNAVAILABLE`, retryable |
| 9 | Item-state partial then failure | `PAGE_PREPARED` | sink APPLIED; state APPLYING | no | no | replay | same | no | per §10 | `DEPENDENCY_UNAVAILABLE`, retryable |
| 10 | Item-state APPLYING recovery | `PAGE_PREPARED` | sink APPLIED; state APPLYING | no | no | replay | same | no | per §10 | — |
| 11 | Item-state COMPLETED but run still PAGE_PREPARED | `PAGE_PREPARED` | state COMPLETED | no | no | no | same | no | `COLLECTING` or `FINALIZING` | — |
| 12 | Prepared page no longer reproducible, no side effects | `PAGE_PREPARED` | both ABSENT | verify; mismatch | no | no | same | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE`, not retryable |
| 13 | Prepared page no longer reproducible after sink APPLIED | `PAGE_PREPARED` | sink APPLIED | no | no | per §10 | same | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE`, not retryable |
| 14 | Sink receipt CONFLICT | `PAGE_PREPARED` | sink CONFLICT | no | no | no | same | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE`, not retryable |
| 15 | Sink receipt UNKNOWN | `PAGE_PREPARED` | sink UNKNOWN | no | no | no | same | no | `RECOVERY_REQUIRED` | recovery mapping |
| 16 | Item-state receipt CONFLICT | `PAGE_PREPARED` | state CONFLICT | no | no | no | same | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE`, not retryable |
| 17 | Checkpoint commit succeeded, run completion failed | `FINALIZING` | all complete | no | no | no | same | already committed | `COMPLETED` (Case B) | — |
| 18 | Final checkpoint CAS failure | `FINALIZING` | all complete | no | no | no | same | retry CAS (Case A) | `COMPLETED` | `DEPENDENCY_UNAVAILABLE`, retryable |
| 19 | Checkpoint differs from both expected and intended | `FINALIZING` | all complete | no | no | no | same | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE`, not retryable |
| 20 | Retry with `restart=True` | active run with side effects | per phase | **fail closed** | **no** | **no** | — | no | blocked | `INVALID_CURSOR`, not retryable |
| 21 | Stale binding configuration | any active | — | no | no | no | — | no | `RECOVERY_REQUIRED` | `INVALID_CURSOR`, not retryable |
| 22 | Configuration changed while active slot exists | active | — | no | no | no | — | no | `RECOVERY_REQUIRED` | `INVALID_CURSOR`, not retryable |
| 23 | Concurrent call, lease busy | — | — | no | no | no | none | no | — | `LEASE_BUSY` |
| 24 | Incremental sync while reconciliation active | active run | — | no | no | no | none | no | blocked | `INVALID_CURSOR`, not retryable |
| 25 | Candidate count limit exceeded | before `COLLECTING` | none | **no** | no | no | none | no | blocked | `CONFIGURATION_ERROR`, not retryable |
| 26 | Candidate byte limit exceeded | before `COLLECTING` | none | **no** | no | no | none | no | blocked | `CONFIGURATION_ERROR`, not retryable |
| 27 | Single remote ID byte limit exceeded | before `COLLECTING` | none | **no** | no | no | none | no | blocked | `CONFIGURATION_ERROR`, not retryable |
| 28 | Operator ABORT_PRISTINE | any with pristine proof | both ABSENT | no | no | no | — | no | `ABORTED` | — |
| 29 | Operator FINALIZE_ALREADY_COMMITTED | `FINALIZING` | checkpoint = intended final | no | no | no | same | already committed | `COMPLETED` | — |

### Crash boundaries

| Last committed boundary | Resume phase | Behavior |
|---|---|---|
| run created in `COLLECTING` | `COLLECTING` | continue from stored cursor and candidates |
| `PAGE_PREPARED` written | `PAGE_PREPARED` | receipt-driven retry per §10 |
| sink accepted | `PAGE_PREPARED` | receipt-driven retry per §10 |
| item-state accepted | `PAGE_PREPARED` or `FINALIZING` | per §10 |
| `FINALIZING` written | `FINALIZING` | checkpoint idempotency per §11 |
| checkpoint committed, run not `COMPLETED` | `FINALIZING` | Case B: CAS run to `COMPLETED` only |
| run `COMPLETED` | — | ordinary incremental sync allowed |

---

## 15. Implementation decomposition (later tasks)

Architecture only — implementation is split as follows.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`

**Status:** `PLANNED` — blocked pending architecture correction acceptance.

Contracts and durable repositories:

- reconciliation-run models and phases (`COLLECTING`, `PAGE_PREPARED`, `FINALIZING`, `COMPLETED`, `RECOVERY_REQUIRED`, `ABORTED`);
- single active slot keyed by `(tenant_id, binding_id)`;
- run repository and CAS transitions;
- candidate count and byte limits;
- sink delivery receipt protocol;
- item-state delivery receipt protocol;
- prepared state mutations;
- recovery command contracts;
- DocumentStore implementation;
- model/repository tests.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B`

**Status:** `PLANNED`.

Coordinator integration:

- start/resume rules and `restart` semantics;
- materialized batch fingerprinting;
- `PAGE_PREPARED` creation;
- receipt-driven retry decision table (§10);
- synthetic source-scoped tombstones on final page only;
- incremental-sync blocking while active run exists;
- `FINALIZING` idempotency (§11);
- `RECOVERY_REQUIRED` transitions;
- operator-safe recovery orchestration (§12);
- failure-window tests covering §14.

### `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1`

Calendar acceptance proof only — does not own generic reconciliation lifecycle code:

- non-primary Calendar missing-item detection;
- no tombstones before the final snapshot page;
- retry after sink / state / checkpoint failures with deterministic batches;
- same integration instance;
- no raw continuation leakage;
- attachment inventory completeness flags;
- Calendar status restored to `READY_FOR_REVIEW` only after proof passes.

---

## 16. Roadmap linkage

| Item | Status after this task |
|---|---|
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1` | `CHANGES_REQUIRED` — correction under review |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1` | `READY_FOR_REVIEW` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A` | `PLANNED` — blocked pending architecture correction acceptance |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B` | `PLANNED` |
| `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` | `CHANGES_REQUIRED` |
| Microsoft Graph adapter family (`MSGRAPH-KNOWLEDGE-ADAPTERS-1`) | `IN_PROGRESS` |
| Google Workspace knowledge workstream | independent; does not gate reconciliation finalization or Microsoft Calendar acceptance |

**Next task after architecture acceptance:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`.

Missing-item detection for non-primary Calendar reconciliation is **not** implemented on HEAD. Calendar adapter code exists but durable finalization semantics required for safe missing-item tombstones are **not** present until `1A` + `1B` land.

---

## 17. Non-goals (this architecture task)

- production coordinator changes;
- Calendar adapter changes;
- Google Workspace, LKW Conversation Context, or unrelated provider work;
- paged candidate inventory implementation (extension point only);
- claiming reconciliation tombstones prove global provider deletion.
