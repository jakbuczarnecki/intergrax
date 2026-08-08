# Vendor Knowledge — Reconciliation Finalization Architecture

**Task:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1`
**Status:** `CHANGES_REQUIRED` — correction under review
**Review fix:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1`
**Review-fix-1 status:** `CHANGES_REQUIRED`
**Review fix 2:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2`
**Review-fix-2 status:** `ACCEPTED`
**Branch:** `development`
**Plan:** [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**Architecture:** [`../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)

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
| `COLLECTING` | Run exists; provider pages may be read; candidate inventory may be loaded once at run start; no prepared page side effects for the current page. Earlier pages may already have been applied. |
| `PAGE_PREPARED` | Exact page intent is durably frozen **before** sink or remote-item state mutation for the current page. |
| `FINALIZING` | Final page sink and item-state application succeeded; only completed-checkpoint CAS and run completion remain. |
| `COMPLETED` | Ordinary completed sync checkpoint committed successfully. |

#### Recovery and terminal states (not on the successful path)

| Phase | Meaning |
|---|---|
| `RECOVERY_REQUIRED` | Fail-closed state when automatic deterministic continuation cannot be proven safe. |
| `ABORTED` | Durable terminal state proving the run was safely abandoned. |

#### Run-wide applied-page evidence (immutable or monotonic)

Every active and terminal run stores:

| Field | Semantics |
|---|---|
| `applied_page_count` | Starts at `0`. Authoritative count of prepared pages that completed sink and item-state application. Increases exactly once when a `PAGE_PREPARED` page with item-state receipt `COMPLETED` is durably transitioned to `COLLECTING` (non-final page) or `FINALIZING` (final page). Retrying the same transition must not increment twice. |
| `last_applied_delivery_id` | `null` while `applied_page_count == 0`. After the first applied page, contains the most recently completed page `delivery_id`. Changes only in the same CAS transition that increments `applied_page_count`. |

Derived invariant:

```text
effects_started := applied_page_count > 0
```

Do **not** store a separately mutable boolean that can disagree with `applied_page_count`.

When the run remains in `PAGE_PREPARED` and a receipt is `APPLYING` or `COMPLETED`, side effects may already exist even if `applied_page_count` has not yet advanced. Recovery and abort guards must inspect both `applied_page_count` and current prepared delivery receipts.

`COLLECTING` does **not** imply a pristine run. A run may return to `COLLECTING` with `applied_page_count > 0` after earlier pages were applied.

#### `COLLECTING`

The run owns:

- `tenant_id`;
- `binding_id`;
- `binding_configuration_version` (immutable field inside the run);
- `provider_id` and `source_kind` (source identity);
- opaque `run_id`;
- base completed checkpoint **or** its exact CAS identity (`expected_base_checkpoint_cas`);
- `applied_page_count` and `last_applied_delivery_id`;
- exact private cursor state:
  - `current_input_cursor: KnowledgeCursor | null` — at run start `null`; on resume the exact opaque cursor required by the adapter for the next provider page;
  - `current_input_cursor_fingerprint` — must correspond to `current_input_cursor`;
- bounded remaining candidate remote IDs (see §4);
- **no** prepared page side effects for the current page.

Cursor objects are private durable state validated through normal `KnowledgeCursor` contracts. They must be excluded from public safe output, logs and ordinary repr. Encoded raw continuation URLs may remain inside private opaque cursor values when already required by an adapter; those values must never appear in public prepared-intent views or error messages.

Fingerprints are used for comparison only; a cursor fingerprint is **not** sufficient to call the provider.

Entry:

- authorized new reconciliation when no active run exists; or
- resume of an existing active run in `COLLECTING`.

#### `PAGE_PREPARED`

Before calling the sink, the coordinator durably prepares the exact page intent.

**Prepared intent — minimum required fields:**

| Field | Purpose |
|---|---|
| `run_id` | binds intent to one reconciliation run |
| `prepared_input_cursor` | exact private `KnowledgeCursor | null` used for this page |
| `prepared_input_cursor_fingerprint` | fingerprint corresponding to `prepared_input_cursor` |
| `provider_page_fingerprint` | canonical fingerprint of the fetched provider page |
| `prepared_batch_payload_fingerprint` | SHA-256 of exact materialized sink payload manifest excluding `delivery_id` (see §5) |
| `prepared_state_mutation_templates` | bounded exact item-state mutation templates (see §6) |
| `prepared_state_mutations_fingerprint` | SHA-256 of canonical JSON over ordered templates |
| `prepared_proposed_checkpoint` | exact private `KnowledgeCursor | null` for the resulting checkpoint |
| `prepared_proposed_checkpoint_fingerprint` | fingerprint corresponding to `prepared_proposed_checkpoint` |
| `prepared_next_cursor` | exact private `KnowledgeCursor | null` for continuation when `has_more=true` |
| `prepared_next_cursor_fingerprint` | fingerprint corresponding to `prepared_next_cursor` |
| `has_more` | whether more provider pages remain |
| `delivery_id` | frozen deterministic delivery identity for this page (see §7) |
| `remaining_candidate_remote_ids` | candidate IDs after subtracting all remote IDs present on this provider page |
| `synthetic_tombstone_remote_ids` | exact **sorted** synthetic tombstone remote IDs; **empty** for non-final pages |
| `binding_configuration_version` | configuration guard |

Exact cursor objects and their fingerprints must correspond. Cursor objects are private durable state excluded from public prepared-intent views.

**Prepared-intent bounds (frozen before CAS):**

| Limit | Semantics |
|---|---|
| `max_reconciliation_prepared_intent_payload_bytes` | Applies to the canonical serialized private `PAGE_PREPARED` durable payload, including all identities, exact private cursor objects, fingerprints, provider-page fingerprint, prepared batch payload fingerprint, prepared state-mutation templates, prepared state-mutations fingerprint, synthetic tombstone IDs and semantics, remaining candidate IDs, delivery ID, `has_more`, record schema and phase-specific fields. |
| `max_reconciliation_prepared_state_mutation_count` | Explicit count bound compatible with configured provider page size plus possible final synthetic tombstones. |

Required behavior when either limit is exceeded:

1. construct and validate the complete candidate `PAGE_PREPARED` record in memory;
2. serialize it canonically as UTF-8;
3. check its exact byte length;
4. if the limit is exceeded:
   - do **not** CAS `PAGE_PREPARED`;
   - do **not** call the sink;
   - do **not** apply item states;
   - do **not** advance any cursor;
   - keep the run in `COLLECTING`;
   - return `CONFIGURATION_ERROR`, not retryable.

Count or payload overflow must fail before any side effect. v1 uses one bounded record and fails closed when it does not fit.

**Safety constraints for prepared intent storage:**

- no credentials, tokens, or secret material;
- no raw provider continuation URLs in public views (private opaque cursor objects may retain adapter-required values);
- no full fetched content bodies or attachment bytes;
- no event bodies, message bodies, or provider payloads in `prepared_state_mutation_templates`.

**Final page rule:** a page with `has_more=false` **must** reach `PAGE_PREPARED` before sink or remote-item state side effects occur. Synthetic tombstones are computed and stored in the prepared intent on that page only.

**Retry in `PAGE_PREPARED`** follows the frozen receipt-driven decision table in §10. Provider re-read is allowed **only** when both receipts are `ABSENT`.

After successful non-final page application, the CAS transition to `COLLECTING` sets `current_input_cursor = prepared_next_cursor`, clears prepared cursor fields, increments `applied_page_count` exactly once, and sets `last_applied_delivery_id` to the page `delivery_id`.

#### `FINALIZING`

Entered only after **all** of the following:

- sink accepted the prepared batch;
- remote-item state repository idempotently accepted the same `delivery_id`;
- the prepared page had `has_more=false` (final provider page);
- `applied_page_count` was incremented for the final page in the same transition.

`FINALIZING` retains:

- `intended_final_completed_checkpoint` — exact private `KnowledgeCursor` derived from `prepared_proposed_checkpoint`;
- `intended_final_checkpoint_fingerprint`;
- `expected_previous_completed_checkpoint` (complete canonical checkpoint identity observed at final page preparation);
- final `delivery_id`;
- `prepared_batch_payload_fingerprint`;
- `run_id`.

**Retry in `FINALIZING`** follows §11. Checkpoint commit and run completion are **separate durable boundaries**.

#### `COMPLETED`

The ordinary completed sync checkpoint was committed successfully with CAS against `expected_previous_completed_checkpoint`.

#### `RECOVERY_REQUIRED`

A durable fail-closed state used when automatic deterministic continuation cannot be proven safe.

Examples:

- both receipts `ABSENT` but provider page or payload fingerprints no longer match;
- delivery receipt reports a conflicting fingerprint;
- stored prepared intent or receipt binding is corrupt after sink `APPLIED`;
- completed checkpoint is neither the expected previous checkpoint nor the exact intended final checkpoint;
- binding configuration changed while an active run exists;
- cursor object/fingerprint mismatch;
- durable run state or delivery state is internally inconsistent;
- sink receipt `UNKNOWN`.

While in `RECOVERY_REQUIRED`:

- incremental synchronization is blocked;
- normal reconciliation continuation is blocked;
- a new run is blocked;
- the run must not be silently deleted or replaced;
- only an explicit recovery operation with exact run identity and CAS version may act (see §12).

#### `ABORTED`

A durable terminal state proving that the run was safely abandoned.

A run may become `ABORTED` only through `ABORT_PRISTINE` (see §12) or after an explicit recovery procedure has repaired or compensated all effects and recorded an auditable resolution receipt.

`ABORT_PRISTINE` is **forbidden** when:

- `applied_page_count > 0`;
- any sink receipt is `APPLIED`, `CONFLICT` or `UNKNOWN`;
- any item-state receipt is `APPLYING`, `COMPLETED` or `CONFLICT`;
- the completed checkpoint differs from the expected base checkpoint;
- durable evidence is missing or inconclusive.

A partially applied reconciliation must enter or remain in `RECOVERY_REQUIRED`. It cannot be converted to `ABORTED` without a separately reviewed repair or compensation procedure and an auditable resolution receipt.

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
| `restart=True` with active run | **Fail closed** when `effects_started` or when phase is `PAGE_PREPARED`, `FINALIZING`, or `RECOVERY_REQUIRED`. Allowed only when active run is absent or `ABORT_PRISTINE` proof succeeds. |
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

Synthetic tombstones are represented as envelopes in the `prepared_batch_payload_fingerprint` manifest with exact sorted synthetic tombstone IDs and their source-scoped semantic marker.

### 3.6 Final completed-checkpoint CAS rule

Completed checkpoint commit occurs **only** in `FINALIZING` after sink and item-state success for the final page.

CAS inputs:

- `new_checkpoint` with final provider cursor;
- `expected_previous=expected_previous_completed_checkpoint` captured when the final page entered `PAGE_PREPARED`.

Non-final pages **must not** advance the completed sync checkpoint. They may update only run state (`remaining_candidate_remote_ids`, run cursor, phase, `applied_page_count`, `last_applied_delivery_id`).

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

## 6. Prepared batch payload fingerprint (non-circular)

`prepared_batch_payload_fingerprint` is SHA-256 over canonical JSON of the exact logical sink payload manifest **excluding `delivery_id`**.

The manifest must include:

```text
tenant_id
binding_id
binding_configuration_version
mode
run_id
source identity
has_more
ordered envelope payload fingerprints
prepared_state_mutations_fingerprint
provider_page_fingerprint
input_cursor_fingerprint
proposed_checkpoint_fingerprint
next_cursor_fingerprint
```

Each ordered envelope fingerprint entry must include:

```text
change_kind
remote_id
descriptor fingerprint or null
content fingerprint or null
permissions fingerprint or null
source-scoped tombstone semantics when applicable
```

The payload fingerprint must **not** include:

- `delivery_id`;
- any object that itself contains `delivery_id`;
- receipt state;
- mutable timestamps generated after preparation.

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

## 7. Prepared state-mutation templates and delivery-ID contract (frozen)

### Step 1 — prepared state-mutation templates

`PAGE_PREPARED` stores bounded, safe and exact `prepared_state_mutation_templates`.

Each template contains only:

```text
remote_id
resulting_status
revision_or_null
binding_configuration_version
```

Templates must **not** contain `delivery_id` because the delivery ID has not yet been computed. The delivery ID is injected into the actual `KnowledgeRemoteItemState.last_delivery_id` only when the frozen templates are applied.

Compute:

```text
prepared_state_mutations_fingerprint
```

as SHA-256 of canonical JSON over the exact ordered templates.

Required ordering: `remote_id` ascending, UTF-8 lexicographic.

Synthetic tombstone templates follow the same ordering and carry the source-scoped semantic identity:

```text
absent_from_completed_synchronized_source_inventory
```

Do not store event bodies, message bodies, attachment bytes, credentials or provider payloads.

### Step 2 — batch payload fingerprint

See §6. `prepared_batch_payload_fingerprint` excludes `delivery_id`.

### Step 3 — delivery ID

Compute:

```text
delivery_id = SHA256(canonical_json({
  tenant_id,
  binding_id,
  binding_configuration_version,
  mode,
  run_id,
  provider_page_fingerprint,
  prepared_batch_payload_fingerprint,
  prepared_state_mutations_fingerprint,
  input_cursor_fingerprint,
  proposed_checkpoint_fingerprint,
  next_cursor_fingerprint
}))
```

The delivery ID depends on the payload fingerprint, but the payload fingerprint never depends on the delivery ID.

### Step 4 — delivery receipt binding

Sink receipt inspection binds to:

```text
tenant_id
binding_id
delivery_id
prepared_batch_payload_fingerprint
```

Item-state receipt inspection binds to:

```text
tenant_id
binding_id
delivery_id
prepared_state_mutations_fingerprint
```

The architecture may define a convenience `prepared_delivery_fingerprint` over both the delivery ID and payload fingerprint, but it must **not** be used as an input for computing either of them.

**Must not appear** in delivery-id inputs or public prepared-intent objects:

- raw provider URLs;
- credentials, access tokens, refresh tokens;
- signed URLs;
- attachment bytes;
- event/message body content.

The delivery ID is assigned in `PAGE_PREPARED` and is immutable for that page intent.

---

## 8. Delivery receipt inspection

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
prepared_batch_payload_fingerprint
```

Semantics:

- `ABSENT`: the sink proves the delivery has not been applied;
- `APPLIED`: the sink proves the exact fingerprint was accepted;
- `CONFLICT`: the same delivery ID exists with a different fingerprint;
- `UNKNOWN`: the sink was reached but cannot prove whether the delivery was absent or applied.

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
tenant_id
binding_id
delivery_id
prepared_state_mutations_fingerprint
```

The architecture may extend the existing delivery-marker repository contract in implementation task `1A`.

Do not require a provider re-read when item-state receipt is `APPLYING` or `COMPLETED`.

---

## 9. `PAGE_PREPARED` retry decision table (frozen)

### Item-state receipt `COMPLETED`

The sink necessarily succeeded earlier because sink application precedes item-state application.

Therefore:

- do not read the provider;
- do not call the sink;
- do not replay state mutations;
- use the frozen prepared intent;
- for a non-final page, CAS the run back to `COLLECTING` and increment `applied_page_count` exactly once;
- for a final page, CAS the run to `FINALIZING` and increment `applied_page_count` exactly once.

### Item-state receipt `APPLYING`

- do not read the provider;
- do not call the sink again;
- replay the exact stored `prepared_state_mutation_templates` with the same delivery ID;
- then perform the phase transition and increment `applied_page_count` exactly once.

### Item-state receipt `ABSENT`

Inspect the sink receipt.

#### Sink receipt `APPLIED`

- do not read the provider;
- do not resend the sink batch;
- apply the exact stored state-mutation templates;
- then perform the phase transition and increment `applied_page_count` exactly once.

#### Sink receipt `ABSENT`

No durable side effect exists for this page.

**Only in this case** may the provider page be reread and rematerialized using `prepared_input_cursor`.

It must reproduce:

```text
provider_page_fingerprint
prepared_batch_payload_fingerprint
prepared_state_mutations_fingerprint
```

If all three match:

- call the sink with the same delivery ID;
- apply the frozen state-mutation templates;
- continue.

If any differs:

- do not call the sink;
- transition to `RECOVERY_REQUIRED`.

#### Sink receipt `CONFLICT` or `UNKNOWN`

Transition to `RECOVERY_REQUIRED`.

### Item-state receipt `CONFLICT`

Transition to `RECOVERY_REQUIRED`.

### Stored prepared intent corrupt after sink `APPLIED`

When sink receipt is `APPLIED` but stored prepared intent or receipt binding is corrupt (missing fields, cursor/fingerprint mismatch, unreadable templates):

- do not read the provider;
- do not replay the sink;
- do not apply state mutations based on untrusted intent;
- transition to `RECOVERY_REQUIRED`;
- return the frozen non-retryable corruption error (`INVALID_PROVIDER_RESPONSE`).

Provider reproducibility is irrelevant after the sink proves the exact delivery was already applied.

---

## 10. `FINALIZING` checkpoint idempotency (frozen)

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

## 11. Operator recovery (frozen)

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

Allowed when stored receipts and fingerprints prove that the existing prepared intent remains valid and recoverable per §9.

Resumes the existing run only. Must not create a new run ID.

#### `FINALIZE_ALREADY_COMMITTED`

Allowed when:

```text
phase == FINALIZING
current completed checkpoint == intended final checkpoint
```

Only marks the run `COMPLETED`.

#### `ABORT_PRISTINE`

Allowed only when **all** of the following are true:

```text
applied_page_count == 0
last_applied_delivery_id is null
current completed checkpoint == expected base completed checkpoint
```

and additionally one of these phase-specific proofs holds:

**Run in `COLLECTING`:**

- no prepared intent exists;
- no delivery ID has ever been prepared for this run;
- no sink receipt exists for this run;
- no item-state delivery receipt exists for this run.

**Run in `PAGE_PREPARED`:**

- sink receipt is `ABSENT`;
- item-state receipt is `ABSENT`;
- no checkpoint mutation belongs to the run;
- prepared intent is internally valid.

`ABORT_PRISTINE` is forbidden when `applied_page_count > 0`, any sink receipt is `APPLIED`/`CONFLICT`/`UNKNOWN`, any item-state receipt is `APPLYING`/`COMPLETED`/`CONFLICT`, the completed checkpoint differs from the expected base checkpoint, or durable evidence is missing or inconclusive.

A partially applied reconciliation must enter or remain in `RECOVERY_REQUIRED`.

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
mark ABORTED without ABORT_PRISTINE proof
abort COLLECTING run with applied_page_count > 0
```

Fail-closed blocking is preferable to silently unlocking an inconsistent binding.

---

## 12. Error categories (frozen)

| Condition | Category | Retryable | Run transition | Safe message |
|---|---|---|---|---|
| lease busy | `LEASE_BUSY` | yes | — | — |
| provider dependency failure before preparation | `DEPENDENCY_UNAVAILABLE` | yes | — | — |
| CAS conflict with no contradictory durable state | `DEPENDENCY_UNAVAILABLE` | yes | — | — |
| receipt inspection operation failed (backend unavailable before returning state) | `DEPENDENCY_UNAVAILABLE` | yes | phase remains `PAGE_PREPARED` | — |
| prepared page or batch mismatch (both receipts `ABSENT`) | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` | — |
| stored prepared intent corrupt after sink `APPLIED` | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` | — |
| stale binding configuration | `INVALID_CURSOR` | no | → `RECOVERY_REQUIRED` | — |
| candidate, prepared-intent, or state-mutation policy exceeded | `CONFIGURATION_ERROR` | no | — | — |
| delivery fingerprint conflict | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` | — |
| sink receipt returned `UNKNOWN` | `DEPENDENCY_UNAVAILABLE` | **no** | → `RECOVERY_REQUIRED` | `Knowledge delivery outcome requires recovery` |
| completed checkpoint differs from both expected and intended | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` | — |
| cursor object/fingerprint mismatch | `INVALID_PROVIDER_RESPONSE` | no | → `RECOVERY_REQUIRED` | — |

For sink receipt `UNKNOWN`, do not expose delivery IDs, provider IDs, cursor values, raw URLs, or sink implementation details.

Administrator-configured limits are **not** classified as malformed provider responses.

---

## 13. Page and failure matrix

Legend:

- **Phase** — durable reconciliation-run phase at retry entry.
- **Applied** — `applied_page_count` at entry.
- **Receipts** — durable delivery receipts that may exist.
- **Provider** — whether the provider is read again.
- **Sink** — whether sink may be invoked again.
- **State** — whether item-state mutations may be replayed.
- **Delivery ID** — same / new / none.
- **Cursor** — cursor transition on success.
- **Checkpoint** — whether completed checkpoint may advance.
- **Next** — next phase after successful retry.
- **Error** — expected safe error category.
- **Retry** — retryable.

| # | Case | Phase | Applied | Receipts | Provider | Sink | State | Delivery ID | Cursor | Checkpoint | Next | Error | Retry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | First provider page applied | `PAGE_PREPARED` | 0→1 | sink APPLIED + state COMPLETED | no | no | no | same | `prepared_next_cursor` → `current_input_cursor` | no | `COLLECTING` | — | — |
| 2 | Later provider page applied | `PAGE_PREPARED` | n→n+1 | sink APPLIED + state COMPLETED | no | no | no | same | `prepared_next_cursor` → `current_input_cursor` | no | `COLLECTING` | — | — |
| 3 | Run returned to `COLLECTING` with prior pages applied | `COLLECTING` | >0 | none for current page | yes (next page) | only after prepare | after sink | new on prepare | advance on each applied page | no | `PAGE_PREPARED` or `COLLECTING` | — | — |
| 4 | `ABORT_PRISTINE` before any page effects | `COLLECTING` or `PAGE_PREPARED` | 0 | both ABSENT; no prepared delivery | no | no | no | none | unchanged | no | `ABORTED` | — | — |
| 5 | `ABORT_PRISTINE` after earlier page applied | any | >0 | any | no | no | no | — | — | no | blocked | — | — |
| 6 | `PAGE_PREPARED` exceeds payload-byte limit | `COLLECTING` | unchanged | none | no | no | no | none | unchanged | no | `COLLECTING` | `CONFIGURATION_ERROR` | no |
| 7 | Prepared state-mutation count exceeds limit | `COLLECTING` | unchanged | none | no | no | no | none | unchanged | no | `COLLECTING` | `CONFIGURATION_ERROR` | no |
| 8 | Both receipts `ABSENT`, provider and payload reproduce exactly | `PAGE_PREPARED` | unchanged | both ABSENT | yes (verify) | yes | yes | same | per success path | no | `COLLECTING` or `FINALIZING` | — | — |
| 9 | Both receipts `ABSENT`, provider page fingerprint differs | `PAGE_PREPARED` | unchanged | both ABSENT | yes (mismatch) | no | no | same | unchanged | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE` | no |
| 10 | Both receipts `ABSENT`, batch payload fingerprint differs | `PAGE_PREPARED` | unchanged | both ABSENT | yes (mismatch) | no | no | same | unchanged | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE` | no |
| 11 | Sink `APPLIED`, state `ABSENT` | `PAGE_PREPARED` | n→n+1 | sink APPLIED; state ABSENT | no | no | replay templates | same | per success path | no | `COLLECTING` or `FINALIZING` | — | — |
| 12 | State `APPLYING` | `PAGE_PREPARED` | n→n+1 | sink APPLIED; state APPLYING | no | no | replay templates | same | per success path | no | `COLLECTING` or `FINALIZING` | — | — |
| 13 | State `COMPLETED`, phase still `PAGE_PREPARED` | `PAGE_PREPARED` | n→n+1 | state COMPLETED | no | no | no | same | per success path | no | `COLLECTING` or `FINALIZING` | — | — |
| 14 | Stored prepared intent corrupt after sink `APPLIED` | `PAGE_PREPARED` | unchanged | sink APPLIED; corrupt intent | no | no | no | same | unchanged | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE` | no |
| 15 | Receipt inspection call dependency failure | `PAGE_PREPARED` | unchanged | inspection failed | no | no | no | same | unchanged | no | `PAGE_PREPARED` | `DEPENDENCY_UNAVAILABLE` | yes |
| 16 | Sink receipt returned `UNKNOWN` | `PAGE_PREPARED` | unchanged | sink UNKNOWN | no | no | no | same | unchanged | no | `RECOVERY_REQUIRED` | `DEPENDENCY_UNAVAILABLE` | no |
| 17 | Checkpoint committed, run still `FINALIZING` | `FINALIZING` | final | all complete | no | no | no | same | unchanged | already committed | `COMPLETED` (Case B) | — | — |
| 18 | Cursor object/fingerprint mismatch | any active | any | — | no | no | no | — | — | no | `RECOVERY_REQUIRED` | `INVALID_PROVIDER_RESPONSE` | no |
| 19 | Configuration changed while active run exists | any active | any | — | no | no | no | — | — | no | `RECOVERY_REQUIRED` | `INVALID_CURSOR` | no |

Additional standard cases (provider read failure, sink failure before APPLIED, candidate limits, lease busy, incremental sync blocked, `restart=True` with side effects) follow §3.4, §4 and §12 with the same column semantics.

No row may equate `COLLECTING == pristine`. No row may reread the provider after sink `APPLIED` or state `APPLYING`/`COMPLETED`.

### Crash boundaries

| Last committed boundary | Resume phase | Behavior |
|---|---|---|
| run created in `COLLECTING` | `COLLECTING` | continue from `current_input_cursor` and candidates |
| earlier page applied, run in `COLLECTING` | `COLLECTING` | `applied_page_count > 0`; continue next page |
| `PAGE_PREPARED` written | `PAGE_PREPARED` | receipt-driven retry per §9 |
| sink accepted | `PAGE_PREPARED` | receipt-driven retry per §9 |
| item-state accepted | `PAGE_PREPARED` or `FINALIZING` | per §9; increment `applied_page_count` on transition |
| `FINALIZING` written | `FINALIZING` | checkpoint idempotency per §10 |
| checkpoint committed, run not `COMPLETED` | `FINALIZING` | Case B: CAS run to `COMPLETED` only |
| run `COMPLETED` | — | ordinary incremental sync allowed |

---

## 14. Implementation decomposition (later tasks)

Architecture only — implementation is split as follows.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`

**Status:** `CHANGES_REQUIRED`

Contracts and durable repositories:

- reconciliation-run models and all phases (`COLLECTING`, `PAGE_PREPARED`, `FINALIZING`, `COMPLETED`, `RECOVERY_REQUIRED`, `ABORTED`);
- single active slot keyed by `(tenant_id, binding_id)`;
- exact private cursor fields (`current_input_cursor`, `prepared_input_cursor`, `prepared_next_cursor`, `prepared_proposed_checkpoint` and fingerprints);
- run repository and CAS transitions;
- `applied_page_count` and `last_applied_delivery_id`;
- candidate count and byte limits;
- prepared-intent count and byte limits (`max_reconciliation_prepared_intent_payload_bytes`, `max_reconciliation_prepared_state_mutation_count`);
- prepared state-mutation templates;
- sink delivery receipt inspection protocol;
- item-state delivery receipt inspection protocol;
- recovery command contracts;
- DocumentStore implementation;
- model/repository tests.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1`

**Status:** `CHANGES_REQUIRED`

Review-fix contracts for durable base checkpoint retention, exact recovery evidence, real v2 item-state receipt binding, remote-ID byte limits, ordered mutation templates and full durable prepared-record measurement.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2`

**Status:** `ACCEPTED`

Frozen phase immutability, exact successful-path transition binding, final-page tombstone completeness, recovery evidence self-validation, generic recovery exit blocking, configurable remote-ID policy separation, strict v1/v2 marker parsing and single durable wrapper builder.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-3`

**Status:** `ACCEPTED`

Exact delivery-marker payload identity binding, final synthetic tombstone template closure, immutable supersession audit pointer, and FINALIZING/COMPLETED configuration-version self-validation.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B`

**Status:** `ACCEPTED` through `REVIEW-FIX-5-REVIEW-CORRECTION-1`

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-1`

**Status:** `CHANGES_REQUIRED`

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-2`

**Status:** `CHANGES_REQUIRED`

Durable continuation lineage, fail-closed configuration, truthful active-index completeness, proof-bound recovery, exact error mapping, binding/source validation, binary fingerprinting, truthful result counts, and failure-window proofs.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-3`

**Status:** `CHANGES_REQUIRED`

Initial job identity and replay, truthful terminal replay, single PAGE_PREPARED recovery receipt decision, corrupt receipt and checkpoint boundaries, and shared strict active-index parsing.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-4`

**Status:** `CHANGES_REQUIRED`

Strict caller-supplied job identity without durable-state inference, normalized run and candidate-inventory repository error boundaries, and zero-effect rejection for malformed continuation identity.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5`

**Status:** `CHANGES_REQUIRED`

Normalized successful candidate-inventory repository output before `KnowledgeReconciliationRunCollecting` construction, exact structural versus policy error mapping with zero downstream effects, and provider-suite regression attribution with stale caller continuation-identity fixes.

### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5-REVIEW-CORRECTION-1-STATUS-TRUTH-AND-NONSEQUENCE-PROOF`

**Status:** `ACCEPTED`

Explicit non-sequence candidate-inventory rejection proof through `reconcile_once`, and truthful non-contradictory roadmap statuses for Review Fix 5 closeout.

### `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1`

**Status:** `CHANGES_REQUIRED`

Calendar acceptance proof only — does not own generic reconciliation lifecycle code:

- non-primary Calendar missing-item detection;
- no tombstones before the final snapshot page;
- retry after sink / state / checkpoint failures with deterministic batches;
- same integration instance;
- no raw continuation leakage;
- attachment inventory completeness flags;
- Calendar status restored to `READY_FOR_REVIEW` only after proof passes.

### `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1-REVIEW-CORRECTION-1-NO-PROVIDER-REREAD-AND-STATUS-HISTORY`

**Status:** `ACCEPTED`

The correction proves receipt-driven Calendar retry after durable sink acceptance without a provider reread, preserves same-instance integration identity, and restores truthful historical reconciliation and Calendar review-fix statuses.

---

## 15. Roadmap linkage

| Item | Status after this task |
|---|---|
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2` | `ACCEPTED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A` | `ACCEPTED` through `REVIEW-FIX-3` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-3` | `ACCEPTED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B` | `ACCEPTED` through `REVIEW-FIX-5-REVIEW-CORRECTION-1` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-1` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-2` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-3` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-4` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5` | `CHANGES_REQUIRED` |
| `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5-REVIEW-CORRECTION-1-STATUS-TRUTH-AND-NONSEQUENCE-PROOF` | `ACCEPTED` |
| `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` | `ACCEPTED` through `REVIEW-FIX-1-REVIEW-CORRECTION-1` |
| `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1` | `CHANGES_REQUIRED` |
| `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1-REVIEW-CORRECTION-1-NO-PROVIDER-REREAD-AND-STATUS-HISTORY` | `ACCEPTED` |
| Microsoft Graph adapter family (`MSGRAPH-KNOWLEDGE-ADAPTERS-1`) | `READY_FOR_REVIEW` |
| `MSGRAPH-KNOWLEDGE-ADAPTERS-1-FAMILY-CLOSEOUT` | `READY_FOR_REVIEW` |
| Google Workspace knowledge workstream | independent; does not gate reconciliation finalization or Microsoft Calendar acceptance |

**Next task after architecture acceptance:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A`.

Non-primary Calendar missing-item detection is implemented and covered by the
accepted Calendar proof; the prior Calendar review fix remains
`CHANGES_REQUIRED`, while its review correction is accepted.

**Next documented Vendor Knowledge task after this family closeout:**
`VENDOR-KNOWLEDGE-ADAPTER-FAMILY-AUDIT-1`.

---

## 16. Non-goals (this architecture task)

- production coordinator changes;
- Calendar adapter changes;
- Google Workspace, LKW Conversation Context, or unrelated provider work;
- paged candidate inventory implementation (extension point only);
- claiming reconciliation tombstones prove global provider deletion.
