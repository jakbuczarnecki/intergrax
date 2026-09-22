# ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY: Durable suspended operation & Execution work re-entry

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture) |
| **Date** | 2026-09-22 |
| **Task** | UCA-6C-ADR2 · **UCA-6C-ADR2-R1** (reconciliation) |
| **Audit baseline** | `1e6341d2c7a4b9c2029a23ccbab05f8cf8f9fa2c` (gap proof) · `e9ea0bdf20f38becce45d811bf09030c8088792f` (ADR2 partial pass) |
| **Reconciliation HEAD** | `8874c5c0af931bd798c90e77d387be9b26081352` (`development`) |
| **Deciders** | Execution Engine / UCA-6C architecture |
| **Related** | [ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION](ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md) · [ADR-GR-5-001](../2026-09-15/ADR-GR-5-001.md) · [ADR-HARNESS-003](ADR-HARNESS-003.md) · [ADR-PLATFORM-PLUGIN-001](../2026-09-20/ADR-PLATFORM-PLUGIN-001.md) |

## Revision — UCA-6C-ADR2-R1 (claim, ordering, codec)

Independent audit **PARTIAL PASS** at `e9ea0bdf…` accepted P4 / D2 / `ExecutionSuspendedWorkReentryPort` but left open:

1. canonical pause / continuation correlation ordering
2. crash-safe claim lifecycle (distinct from `CONSUMED`)
3. typed / versioned payload codec (no semantic `dict` / `Any`)

This revision closes all three **in architecture only** (no production code). Frozen ADR2 core unchanged: continuation port = lifecycle; suspended store = durable work; reentry port = reconstruction + ToolRuntime.

## 1. Context

[ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION](ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md) (ADR1) accepts **Option B1**: continuation-aware integration stays **Execution-owned** and **L3-internal** behind frozen L2 `ExecutionBoundCatalogToolInvoker`, reusing the canonical declarative HITL bridge (`raise_hitl_pause_from_tool_invocation` / `catalog_dispatch` semantics) and `ExecutionContinuationPort` for same-Execution pause/resume.

Independent audit at `1e6341d2c7a4b9c2029a23ccbab05f8cf8f9fa2c` confirmed:

```text
ExecutionContinuationPort = canonical lifecycle continuation owner
ExecutionContinuationPort.resume() != blocked work re-entry
```

Today `ExecutionContinuationStateStore` persists **continuation lifecycle** and exact **four-ID identity** (`PendingExecutionContinuation`, `ExecutionContinuationIdentity`) but **not** the blocked catalog/tool operation payload required to reconstruct an exact `ToolExecutionRequest` after process restart.

The qualified CodeCraft execution path (`QualifiedCapabilityExecutionDispatchService` → `ExecutionRuntime` → `QualifiedCapabilityExecutionRuntimeDelegate` → CodeCraft handler → L2 catalog invoke → L3 `NexusExecutionBoundCatalogToolInvoker` → `RuntimeToolInvoker`) has **no** durable operation descriptor, blocked-work checkpoint, work-reentry API, or replay owner inside the Execution Engine.

This ADR closes the **durable materialization + canonical work re-entry** gap without changing frozen identity, continuation lifecycle semantics, Governance, or ToolRuntime core.

## 2. Proven gap

| Layer | Present today | Missing for QCE HITL resume |
|-------|---------------|-----------------------------|
| Lifecycle | `ExecutionContinuationPort` + `ExecutionContinuationStateStore` | — |
| HITL scope | `GovernedContinuationCorrelation.operation_id`, bridge `invocation_scope_id` (`dhr_*`) | Durable **invocation intent** beyond correlation ids |
| Tool path | In-process `ExecutionBoundCatalogToolInvokeRequest` / `ToolExecutionRequest` | Restart-safe **descriptor** + **re-entry** |
| Code source | `CodeCraftSessionManager` (process-local session) | Pinned **exact** `code.exec` input in durable descriptor |

`PendingExecutionContinuation` does **not** carry catalog input, `tool_id`, or typed tool payload. `GovernedContinuationCorrelation.operation_id` identifies **exact governed HITL resolution scope** (for UCA-6C declarative path: **`invocation_scope_id`**), not a durable replay primary key.

## 3. Identifier ownership (required triad)

| Identifier | Owner | Mint point | Purpose |
| ---------- | ----- | ---------- | ------- |
| `ExecutionId` (+ TaskId, RunId, AttemptId) | Execution Engine / `ExecutionIdentityAuthority` | root admission | execution identity (frozen) |
| `continuation_id` | `ExecutionContinuationPort` episode | **`GovernedContinuationRequest.continuation_request_id`** at request construction; persisted as `PendingExecutionContinuation.continuation_id` at **`request_pause`** | lifecycle episode (frozen semantics) |
| `invocation_scope_id` | declarative HITL bridge | **`generate_invocation_scope_id()`** in `declarative_policy_hitl_bridge.py` at **`raise_hitl_pause_from_tool_invocation`** (`dhr_*`) | exact declarative tool approval scope (one-shot grant binding) |
| `suspended_operation_id` | suspended-work store / domain | **`prepare()`** on descriptor insert | local durable entity key only |

**Correlation triad (three meanings — never substitute):**

```text
continuation_id          ↔ lifecycle episode (ExecutionContinuationPort)
suspended_operation_id   ↔ durable materialization row key (store)
invocation_scope_id      ↔ exact HITL grant scope (declarative bridge)
```

**Correlation diagram (semantic linkage, not identity hierarchy):**

```text
ExecutionId (four-ID bundle)
   |
   +-- continuation_id ............... lifecycle owner: ExecutionContinuationPort
   |       |
   |       +-- suspended_operation_id ... materialization owner: SuspendedExecutionOperationStore
   |
   +-- invocation_scope_id ......... HITL scope owner: declarative bridge (stored on descriptor for verify)
```

**Forbidden:** `uca6c-scope:*`, `uca6c-continuation:*`, `uca6c-replay:*`, `SuspendedExecutionId`, `ReplayExecutionId`, `UcaContinuationId`.

### 3.1 `continuation_id` source (code evidence)

| Step | Contract / symbol |
| ---- | ----------------- |
| Mint **value** | `GovernedContinuationRequest.continuation_request_id` — default `gcr_{uuid4().hex[:12]}` (`intergrax/contracts/governed_continuation.py`) |
| Map to lifecycle id | `apply_governed_continuation_pause` passes `continuation_id=request.continuation_request_id` into `establish_canonical_hitl_pause` → `ExecutionPauseRequest.continuation_id` → `PendingExecutionContinuation.continuation_id` (`governed_continuation_bridge.py`, `internal_continuation_orchestration.py`) |
| Equality rule | `execution_continuation.py` documents: **`continuation_id` equals governed `continuation_request_id`** when pause is HITL-governed; enforced by `assert_governed_correlation_matches_continuation` |

**UCA-6C R6 requirement:** L3 materializer **must** compose **`GovernedContinuationRequest`** (same pattern as `raise_mse_governed_continuation_hitl_pause` / MSE) with **`operation_id = signal.invocation_scope_id`**, then pause via **`establish_canonical_hitl_pause`**. Do **not** rely on graph-only fallback `gcr_hr_{human_request.request_id}` from `resolve_continuation_id_for_execution` without governed correlation.

### 3.2 `operation_id` vs `invocation_scope_id`

| Field | Semantics |
| ----- | --------- |
| `invocation_scope_id` | Bridge-owned **`dhr_*`**; appears on `DeclarativePolicyHitlSignal`, `DeclarativeHitlPendingApproval`, grant |
| `GovernedContinuationCorrelation.operation_id` | Exact governed operation scope for continuation resolution commands (`execution_continuation.py` scope match rules) |

For declarative catalog HITL, **`operation_id` on governed correlation MUST equal `invocation_scope_id`**. Descriptor stores **`invocation_scope_id`** (and verifies **`governed_correlation.operation_id`** match at `BLOCKED` transition) but does **not** mint either id.

## 4. Existing owners (frozen)

| Concern | Owner | ADR2 touch |
|---------|-------|------------|
| Continuation lifecycle | `ExecutionContinuationPort` | **No semantic change** |
| Continuation snapshots | `ExecutionContinuationStateStore` | **No payload extension** |
| Four-ID identity | `ExecutionIdentityAuthority` / admission | **No change** |
| REQUIRE_HUMAN / grant | Declarative HITL bridge + Governance | Reuse; fresh eval on re-entry |
| Physical invoke | `RuntimeToolInvoker` (ToolRuntime) | Mandatory boundary |
| Task checkpoint | `TaskCheckpointPersistence` | Task/runner scope — **not** catalog op replay |
| Agent run checkpoint | `AgentRunCheckpoint` (ACP) | Agent-step scope — **not** EE catalog op |
| Idempotency (effect) | `IdempotencyStore` + `IdempotencyPreEffectCoordinator` | Reuse keys from descriptor — **separate layer from work claim** |
| External op attempt lifecycle | `ExternalOperationAttempt` | **Not** reusable for suspended-work claim (different aggregate); no fencing reuse proven |
| Work claim lease shape | **`LeaseOwnership`** (`owner_id`, `lease_expires_at`, `fence`) + **`StaleClaimError`** | **Reuse field semantics** for suspended-operation claim record (not idempotency claim) |
| UCA / AW / CodeCraft | Coordination & L2 invoke only | **No** replay ownership |

## 5. Non-goals

- Production code in ADR2 / ADR2-R1 (R6 implements).
- Second continuation lifecycle, second HITL owner, or Worker/AW redispatch on HITL approve.
- `RootExecutionLaunch` / new Execution on resume.
- Extending `ExecutionContinuationStateStore` into an operation payload database (default **reject**).
- Public Nexus types or consumer-facing replay ports for UCA/CodeCraft.
- Whole-handler / whole-QCE replay (`dispatch_once` from scratch).
- TIGAE removal (R6 scope per ADR1).
- Global distributed lock / consensus service.

## 6. Primary decision (unchanged direction)

**Durable artifact:** **`SuspendedExecutionOperationDescriptor`** + **`SuspendedExecutionOperationStore`** (P4).

**Re-entry:** **`ExecutionSuspendedWorkReentryPort`** after lifecycle **`RESUMED`** (immediately after successful `resume()` CAS) + governance allows effect.

`ExecutionContinuationPort.resume()` remains **lifecycle-only**.

## 7. Descriptor scope (D2 — unchanged)

**Chosen level:** execution-bound catalog operation (D2). D2 verdict unchanged from ADR2 options matrix.

### 7.1 Descriptor shape (R6 contracts)

Location: `intergrax/contracts/execution/suspended_operation.py` (names may adjust).

- **`SuspendedExecutionOperationDescriptor`**
  - `schema_version` — descriptor envelope version (e.g. `suspended_execution_operation.v1`)
  - `suspended_operation_id` — store primary key (not identity authority)
  - `operation_kind` — `SuspendedOperationKind` enum; initial: `execution_bound_catalog_tool`
  - `identity` — `ExecutionContinuationIdentity`
  - `continuation_id` — correlates lifecycle episode
  - **`invocation_scope_id`** — exact declarative HITL scope (`dhr_*`) for grant correlation verify
  - **`materialization_state`** — `PREPARED` | `BLOCKED` | `CLAIMED` | `CONSUMED` | `ABANDONED` (materialization only)
  - **`materialization_revision`** — monotonic CAS for all transitions (including claim / consume / abandon)
  - **`claim_ownership`** — present only in `CLAIMED`; typed **`SuspendedOperationClaimOwnership`** mirroring **`LeaseOwnership`** (`owner_id`, `lease_expires_at`, `fence`); **internal persistence — not consumer API**
  - **`payload_digest`** — digest over **canonical serialized semantic payload** (integrity only)
  - **`payload`** — discriminated **`SuspendedOperationPayload`** (see §12), never `dict[str, Any]` / `Any` / semantic `object`

**No** grant, wiring resolver, or Nexus types on descriptor.

### 7.2 Code & input durability (unchanged intent)

Pinned typed catalog input in payload; `CodeCraftSessionManager` not replay authority; `tool_input.code` authoritative for UCA-6C until artifact ref exists.

## 8. Canonical pause ordering (O3 — reuse existing orchestration)

**Decision:** Reuse the **MSE / governed continuation** pattern already implemented: mint HITL scope → compose **`GovernedContinuationRequest`** → durable **`PREPARED`** → **`establish_canonical_hitl_pause`** → lifecycle to **`WAITING_FOR_HUMAN`** → **`PREPARED` → `BLOCKED`** with fail-closed validation.

**Exact sequence (UCA-6C L3 on `REQUIRE_HITL`):**

```text
1. RuntimeToolInvoker → DeclarativePolicyHitlRequiredError
2. raise_hitl_pause_from_tool_invocation
      → invocation_scope_id = generate_invocation_scope_id()  [dhr_*]
      → DeclarativePolicyHitlSignal / pending approval artifacts
3. L3 materializer composes GovernedContinuationRequest
      → continuation_request_id minted at construct (gcr_*)
      → operation_id = invocation_scope_id
      → to_correlation() for governed_correlation on pause
4. SuspendedExecutionOperationStore.prepare()
      → PREPARED descriptor (continuation_id = continuation_request_id, invocation_scope_id, payload)
5. establish_canonical_hitl_pause / request_pause
      → PAUSE_REQUESTED → PAUSED → WAITING_FOR_HUMAN  [ExecutionContinuationStateStore]
6. store.block() CAS PREPARED → BLOCKED iff:
      - continuation exists; continuation_id matches
      - four-ID identity matches
      - governed_correlation.operation_id == descriptor.invocation_scope_id
      - lifecycle ∈ {WAITING_FOR_HUMAN, RESUME_AUTHORIZED} and non-terminal
      - payload_digest / operation_kind valid
```

**Answer (ordering gate):** **`continuation_id` value** may exist on **`GovernedContinuationRequest`** **before** `prepare(PREPARED)` and **before** `request_pause` (`GovernedContinuationRequest` mint). **`PAUSE_REQUESTED`** in continuation store is **not** required before **`PREPARED`**, but **`PREPARED` without ever reaching durable pause** is a **transient fault** → reconciler **`ABANDONED`**. **`BLOCKED`** **requires** durable continuation at least **`WAITING_FOR_HUMAN`** with exact correlation — **not** `PREPARED` alone.

**Work reentry timing:** only after **`resume()`** → **`RESUMED`** (preferred: canonical `resume()` CAS first, then reentry coordinator). **`RESUME_AUTHORIZED`** alone is insufficient for ToolRuntime invoke.

**Grant:** one-shot at actual resumed invocation; **claim does not consume grant**; do not remove grant before `ToolExecutionRequest` materialization.

## 9. Descriptor state machine & claim semantics

### 9.1 States

```text
                    ABANDONED ← PREPARED (reconcile: no continuation / terminal mismatch)
                        ↑
PREPARED ──block()──→ BLOCKED ──claim()──→ CLAIMED ──mark_consumed()──→ CONSUMED
                        ↑                      |
                        └── reclaim() ─────────┘ (stale claim: expired lease / higher fence)
```

- **`CONSUMED`:** terminal **successful accounting** of re-entry (backend success, or **`ClaimOutcome.REPLAY_COMPLETED`** via idempotency, or explicit non-effect terminal defined by reentry strategy). **Not** “worker claimed work”.
- **`CLAIMED`:** exclusive **reentry processing ownership** for one host/worker attempt. **Does not authorize** tool execution (Governance still owns WHETHER).
- **`ABANDONED`:** terminal non-executable materialization (`reason_code` typed enum, e.g. `governance_denied`, `continuation_terminal`, `corrupt_correlation`, `orphan_prepared`).

No mirror of `WAITING_FOR_HUMAN` / `RESUMED` inside descriptor states.

### 9.2 Claim model (concrete)

Reuse **`LeaseOwnership`** semantics + **`StaleClaimError`** pattern from `intergrax/contracts/lease_claim.py`:

| Mechanism | Rule |
| --------- | ---- |
| CAS | All mutating store ops take **`expected_materialization_revision`** |
| Claim | `claim(expected_revision, claim_owner_id)` → **`CLAIMED`** with new **`fence`** (monotonic int) + **`lease_expires_at`** |
| Fencing | `mark_consumed` / `abandon` / reclaim require matching **`fence`** + **`owner_id`**; stale host → typed **`STALE_CLAIM`** — fail closed |
| Reclaim | If lease expired before ToolRuntime effect: worker **`reclaim`** with **`fence + 1`**, returning **`BLOCKED`** then allowing new **`claim`** |
| Multi-host | At most one **`CLAIMED`** with valid lease per active descriptor; concurrent claim → **`ALREADY_CLAIMED`** |

**Do not** reuse **`IdempotencyStore.claim`** / **`ClaimOutcome`** for work ownership — effect layer only (§14).

### 9.3 `SuspendedOperationClaimResult` (typed)

Minimum outcomes: **`CLAIMED`**, **`STALE_REVISION`**, **`ALREADY_CLAIMED`**, **`STALE_CLAIM`**, **`TERMINAL`**, **`NOT_FOUND`**, **`INVALID_STATE`**, **`CONTINUATION_MISMATCH`**, **`IDENTITY_MISMATCH`**.

## 10. Re-entry owner (corrected flow)

```text
RESUMED continuation + internal grant evidence
  → load_active_for_continuation(continuation_id)  [fail if ≠1 active BLOCKED/CLAIMED]
  → validate four-ID, invocation_scope_id, payload_digest, operation_kind
  → claim() → CLAIMED  (NOT CONSUMED)
  → reconstruct ExecutionBoundCatalogToolInvokeRequest + ToolExecutionRequest
  → RuntimeToolInvoker (fresh governance + MSE + idempotency_key)
  → on terminal success or REPLAY_COMPLETED → mark_consumed() → CONSUMED
```

- **`SuspendedOperationReentryStrategy`** registry by **`SuspendedOperationKind`** (no reflection / pickle / dotted-path routing).
- Reentry coordinator lives under `intergrax/runtime/execution/`; L3 materialization may live under `intergrax/runtime/nexus/` (internal only).

## 11. Separation of concerns (canonical triple)

```text
ExecutionContinuationPort           = lifecycle (frozen)
SuspendedExecutionOperationStore    = durable work description + materialization claim
ExecutionSuspendedWorkReentryPort   = reconstruction + ToolRuntime invoke
```

## 12. Typed payload codec model

**Ban (contracts):** semantic `dict[str, Any]`, `Any`, `object`, Pydantic `extra="allow"`, reflection codecs, pickle, class-name-in-DB routing.

**Architecture:**

```text
operation_kind + payload_schema_version
  → registered SuspendedOperationPayloadCodec
  → typed SuspendedOperationPayload (discriminated union)
  → canonical serialized envelope (JSON/blob) in provider
  → decode fail-closed on unknown kind/version
```

### 12.1 Codec matrix (minimum)

| operation_kind | payload type | payload_schema_version | codec owner | runtime materialization |
| -------------- | ------------ | ---------------------- | ----------- | ----------------------- |
| `execution_bound_catalog_tool` | `ExecutionBoundCatalogToolOperationPayload` | e.g. `execution_bound_catalog_tool_payload.v1` | Execution contract + domain L3 encoder for `CodeExecInput` | L3 host builds `ToolExecutionRequest` via wiring registry |

**`ExecutionBoundCatalogToolOperationPayload` fields (typed models):**

- `tool_id`, `tool_input_schema_id` (stable platform id, not Python class name)
- **`tool_input`** — typed model matching schema (UCA-6C: explicit **`CodeExecInput`** codec)
- catalog scope: `tenant_id`, `task_id`, `run_id`, `agent_id`, `step_id`
- `idempotency_key`, optional `correlation_request_id`

**Store** persists **`SerializedSuspendedOperationEnvelope`** only; **codec registry** (coordinator) resolves encode/decode — **no store service locator**.

## 13. Store contract semantics (conceptual API)

| Operation | Purpose |
| --------- | ------- |
| `prepare` | Insert **`PREPARED`** |
| `block` | CAS **`PREPARED` → `BLOCKED`** with continuation + HITL correlation validation |
| `load` | By `suspended_operation_id` |
| `load_active_for_continuation` | Exactly **0 or 1** active (`BLOCKED` or `CLAIMED`); **>1 → fail closed** |
| `claim` | CAS **`BLOCKED` → `CLAIMED`** with lease + fence |
| `reclaim` | Stale **`CLAIMED` → `BLOCKED`** with higher fence |
| `mark_consumed` | CAS **`CLAIMED` → `CONSUMED`** with fence + owner + revision |
| `abandon` | Terminal **`ABANDONED`** with typed reason |

No DB transaction leakage in contract surface. Providers: in-memory (`is_durable=False`) for tests; durable required for production restart qualification.

## 14. ToolRuntime, idempotency & two safety layers

| Layer | Owner | Purpose |
| ----- | ----- | ------- |
| Work claim | `SuspendedExecutionOperationStore` | one active reentry worker |
| Effect idempotency | `IdempotencyPreEffectCoordinator` / `IdempotencyStore` | no duplicated backend effect |

After-effect crash: reclaim → invoke same **`idempotency_key`** → **`REPLAY_COMPLETED`** → then **`mark_consumed`**. Before-effect crash: reclaim → normal invoke.

## 15. Cross-store consistency (reconciler owner: Execution Engine)

Reconciler **repairs materialization only** — **never** governance decisions.

| State A | State B | Recovery |
| ------- | ------- | -------- |
| `PREPARED` | no continuation | **`ABANDONED`** (orphan_prepared) |
| `PAUSE_REQUESTED` | no descriptor | deny resume / reentry; pause reconciliation via continuation path |
| `WAITING_FOR_HUMAN` | `PREPARED` | complete **`block()`** if exact match |
| `WAITING_FOR_HUMAN` | no descriptor | **deny** reentry |
| `RESUMED` | `BLOCKED` | eligible for **`claim`** |
| `RESUMED` | `CLAIMED` stale | **`reclaim`** |
| continuation terminal (REJECTED / ESCALATED / CANCELLED) | `BLOCKED` / `CLAIMED` | **`ABANDONED`** |
| `CONSUMED` | resumed retry | idempotent no-op |

**PAUSE_REQUESTED** without descriptor: allowed **transiently** during ordering step 4→5; reconciler must **fail closed** on resume without **`BLOCKED`** descriptor.

Correctness **must not** rely solely on a background daemon — **`load` / resume / reentry paths** enforce fail-closed.

## 16. Crash matrix

| # | Crash point | Restart outcome | Duplicate risk | Loss risk | Fail-closed |
| - | ----------- | --------------- | -------------- | --------- | ----------- |
| 1 | after `PREPARED` | reconcile abandon or complete pause+block | low | none if abandon | yes |
| 2 | after pause request | continuation reconciles; descriptor PREPARED→block or abandon | low | deny if no block | yes |
| 3 | after `WAITING` | block if PREPARED matches | low | deny reentry without BLOCKED | yes |
| 4 | after `BLOCKED` | wait for human + resume | none | none | yes |
| 5 | after `CLAIMED` | reclaim if lease stale | medium without fence | none | fence + idempotency |
| 6 | before ToolRuntime | reclaim → invoke | fenced stale owner | none | yes |
| 7 | after idempotency claim | effect may be in flight → UNCERTAIN path | idempotency layer | none | yes |
| 8 | after backend effect | reclaim → REPLAY_COMPLETED → CONSUMED | prevented by idempotency | none | yes |
| 9 | before `CONSUMED` | same as 8 | prevented | none | yes |
| 10 | after `CONSUMED` | no-op | none | none | yes |

## 17. Restart & multi-host

**Process restart:**

```text
Process A: REQUIRE_HITL → dhr scope → GovernedContinuationRequest (gcr_*) → PREPARED
         → establish_canonical_hitl_pause → BLOCKED → exit
Process B: load continuation → approve → RESUME_AUTHORIZED → resume() → RESUMED
         → load_active_for_continuation → claim (fence) → ToolRuntime → CONSUMED
```

**Multi-host:** **YES** — durable store + revision CAS + claim lease/fence + effect idempotency.

## 18. Observability (conceptual)

Events: descriptor prepared, blocked, claim acquired, reclaim, stale claim, reentry started, idempotent replay, consumed, abandoned. **No raw code** in telemetry — refs/digests only.

## 19. R6 phases (updated)

1. Contracts: descriptor, claim ownership, store port, claim result enums, payload union + codec registry
2. In-memory + durable providers
3. Codec registry + `execution_bound_catalog_tool` + `CodeExecInput` encoder
4. L3 pause materialization (ordering §8) + TIGAE removal
5. Reentry coordinator + claim recovery
6. Restart / multi-host / stale-fence tests + architecture gates

## 20. R6 readiness (ADR2-R1 gates)

| Criterion | R1 |
| --------- | -- |
| `continuation_id` source proven | **YES** |
| Pause ordering proven | **YES** (O3) |
| `invocation_scope_id` correlation proven | **YES** |
| Descriptor state machine crash-safe | **YES** |
| Claim distinct from `CONSUMED` | **YES** |
| Claim reclaimable + stale host fenced | **YES** |
| Typed codec fixed (no semantic Any/dict) | **YES** |
| No new lifecycle / identity | **YES** |
| **UCA-6C-R6** | **UNBLOCKED** (architecture) |

**R6 forbidden scope (unchanged):** identity authority, continuation lifecycle semantics, Governance/ToolRuntime core, public Nexus, Worker replay, CodeCraft replay ownership, universal L2 invoker, global lock framework.

## 21. Permanent architecture gates

```text
AW/UCA must not own suspended execution work
CodeCraft must not own platform replay
Suspended operation contracts must be Nexus-free
Suspended operation store must not authorize execution
ExecutionContinuationPort remains lifecycle-only
Work reentry must preserve four-ID identity
Work reentry must cross ToolRuntime
No replay through RootExecutionLaunch
No Worker redispatch for HITL resume
No runtime object/callable in durable descriptor
Claim must not consume HITL grant
TIGAE removed from final UCA production path (R6)
```

## 22. Architecture gates (summary)

| Check | Result |
|-------|--------|
| Single canonical fulfillment flow preserved | **PASS** |
| Second continuation lifecycle | **NO** |
| Second HITL | **NO** |
| `ExecutionContinuationPort` unchanged | **YES** |
| EE owns durable work + reentry | **YES** |

---

**UCA-6C-ADR2 — Durable Suspended Operation & Execution Work Re-entry**
**UCA-6C-ADR2-R1 — Suspended Operation Claim & Pause Correlation Hardening**
