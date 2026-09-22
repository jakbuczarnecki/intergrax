# ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY: Durable suspended operation & Execution work re-entry

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture) |
| **Date** | 2026-09-22 |
| **Task** | UCA-6C-ADR2 |
| **Audit baseline** | `1e6341d2c7a4b9c2029a23ccbab05f8cf8f9fa2c` (independently audited gap proof) |
| **Session HEAD** | `eabce6bc5cf9879fc928ac7cbdcf5b4d805f7038` (`development`) |
| **Deciders** | Execution Engine / UCA-6C architecture |
| **Related** | [ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION](ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md) · [ADR-GR-5-001](../2026-09-15/ADR-GR-5-001.md) · [ADR-HARNESS-003](ADR-HARNESS-003.md) · [ADR-PLATFORM-PLUGIN-001](../2026-09-20/ADR-PLATFORM-PLUGIN-001.md) |

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

`PendingExecutionContinuation` does **not** carry catalog input, `tool_id`, or typed tool payload. `operation_id` on governed correlation identifies **HITL resolution scope**, not a durable replay record.

## 3. Existing owners (frozen)

| Concern | Owner | ADR2 touch |
|---------|-------|------------|
| Continuation lifecycle | `ExecutionContinuationPort` | **No semantic change** |
| Continuation snapshots | `ExecutionContinuationStateStore` | **No payload extension** |
| Four-ID identity | `ExecutionIdentityAuthority` / admission | **No change** |
| REQUIRE_HUMAN / grant | Declarative HITL bridge + Governance | Reuse; fresh eval on re-entry |
| Physical invoke | `RuntimeToolInvoker` (ToolRuntime) | Mandatory boundary |
| Task checkpoint | `TaskCheckpointPersistence` | Task/runner scope — **not** catalog op replay |
| Agent run checkpoint | `AgentRunCheckpoint` (ACP) | Agent-step scope — **not** EE catalog op |
| Idempotency | `IdempotencyStore` + `IdempotencyPreEffectCoordinator` on `RuntimeToolInvoker` | Reuse keys from descriptor |
| UCA / AW / CodeCraft | Coordination & L2 invoke only | **No** replay ownership |

## 4. Non-goals

- Production code in ADR2 (R6 implements).
- Second continuation lifecycle, second HITL owner, or Worker/AW redispatch on HITL approve.
- `RootExecutionLaunch` / new Execution on resume.
- Extending `ExecutionContinuationStateStore` into an operation payload database (default **reject**).
- Public Nexus types or consumer-facing replay ports for UCA/CodeCraft.
- Whole-handler / whole-QCE replay (`dispatch_once` from scratch).
- TIGAE removal (R6 scope per ADR1).

## 5. Primary question (decision)

**Durable artifact:** a versioned, immutable **`SuspendedExecutionOperationDescriptor`** (Execution-owned contract), persisted by a pluginable **`SuspendedExecutionOperationStore`**, correlated to but not merged with `continuation_id`.

**Re-entry mechanism:** after authorized lifecycle resume, an Execution-internal **`ExecutionSuspendedWorkReentryPort`** (coordinator) loads the descriptor, validates correlation to the active continuation episode, reconstructs L2 `ExecutionBoundCatalogToolInvokeRequest` and L3 `ToolExecutionRequest` (including re-resolved `ToolInvocationWiringResolver`), attaches canonical grant/`invocation_scope_id` from HITL state **internally**, and invokes `RuntimeToolInvoker` once through the existing bridge path.

`ExecutionContinuationPort.resume()` remains **lifecycle-only**; work re-entry is a **separate EE step** triggered only when lifecycle is `RESUMED` (or immediately after successful `resume()` CAS) and governance authorizes backend effect.

## 6. Descriptor scope options (D1–D4)

| Option | Represents | Exact tool replay | Restart-safe | Domain-neutral | Side-effect safe | Verdict |
|--------|------------|-------------------|--------------|----------------|-----------------|---------|
| **D1** | Serialized `ToolExecutionRequest` | Partial | No | No | Risky | **Reject** — embeds `ToolInvocationContext` / `wiring_resolver`, runtime wiring |
| **D2** | **Execution-bound catalog operation** → EE materializes `ToolExecutionRequest` | Yes | Yes | Yes (catalog plane) | Yes — single blocked invoke | **Accept** |
| **D3** | Whole QCE work item | Over-broad | Partial | No | No — replays handler reads | **Reject** |
| **D4** | Whole handler replay descriptor | No | Partial | No | **Reject** — may repeat reads / transient state |

**Chosen level:** **D2** — smallest durable surface that pins **exact** catalog invocation under admitted execution scope, without serializing Nexus `RuntimeState`, resolvers, or callables.

### 6.1 Descriptor shape (conceptual contracts — R6)

Location target: `intergrax/contracts/execution/suspended_operation.py` (names may adjust; semantics frozen here).

- **`SuspendedExecutionOperationDescriptor`** (immutable, `extra=forbid`, frozen)
  - `schema_version` (e.g. `suspended_execution_operation.v1`)
  - `suspended_operation_id` — **local durable entity id** (not execution identity authority)
  - `operation_kind` — typed enum; initial value: `execution_bound_catalog_tool`
  - `identity` — `ExecutionContinuationIdentity` (four IDs)
  - `continuation_id` — correlates to lifecycle episode (**not** overloaded as operation id)
  - `materialization_state` — `PREPARED` \| `BLOCKED` \| `CONSUMED` \| `ABANDONED` (materialization only; **not** continuation lifecycle)
  - `revision` + store CAS (concurrency / claim)
  - `payload_digest` — optional content digest for HITL correlation (reuse platform digest utilities)
  - **`payload`**: nested typed **`ExecutionBoundCatalogToolOperationPayload`**
    - `tool_id`
    - `tool_input_schema_id` + **`tool_input`** — typed JSON round-trip of catalog input (for UCA-6C: `CodeExecInput` fields)
    - Catalog scope: `tenant_id`, `task_id`, `run_id`, `agent_id`, `step_id`
    - `correlation_request_id`, `idempotency_key` (optional)
    - **No** `wiring_resolver`, **no** grant, **no** `RuntimeState`
  - Optional **`execution_context_pins`** (future-proof, minimal for UCA-6C):
    - `qualified_execution_target_reference` — audit-only correlation to `QualifiedCapabilityExecutionTarget.execution_target_reference` (routing ref, **not** sufficient alone for code)
    - `code_artifact_reference` — when CodeCraft exposes durable artifact; until then **`tool_input.code` is authoritative** for exact replay

**Logical vs runtime:** descriptor = **intent**; `ToolExecutionRequest` = **runtime materialization** produced only inside EE L3 host at re-entry.

### 6.2 Code & input durability audit

| Artifact | Serializable | Durable today | ADR2 rule |
|----------|--------------|---------------|-----------|
| `CodeExecInput` | Yes (Pydantic) | Only if copied into descriptor | Persist typed input in payload; note **sensitive code** → encryption/redaction via provider policy (no new crypto subsystem) |
| `CodeCraftSession` / `CodeCraftSessionManager` | Yes in memory | **No** (process-local) | **Not** persistence owner; descriptor pins exact code at pause |
| `QualifiedCapabilityExecutionTarget` | Yes | Routing reference | Supplementary; **cannot** replace pinned `tool_input` |

### 6.3 Tool / provider drift

Re-entry **fail-closed** if `tool_id` or provider wiring no longer resolves. Descriptor may later add optional provider/tool revision pins; R6 must at minimum validate tool registry presence and wiring resolution at reconstruct time (reuse existing provenance patterns where present).

## 7. Persistence owner options (P1–P5)

| Option | Owner | Verdict | Reason |
|--------|-------|---------|--------|
| **P1** Task checkpoint | `TaskCheckpointPersistence` | **Reject** | Task/runner replay semantics; wrong aggregate; mixes coordination |
| **P2** Execution checkpoint (ACP) | `AgentRunCheckpoint` | **Reject** | Agent-step `state_root` bag; not typed catalog op; tier mismatch |
| **P3** Generic durable work item | None suitable in EE | **Reject** | No existing EE contract matches catalog op + continuation correlation |
| **P4** **New EE store** | `SuspendedExecutionOperationStore` | **Accept** | Contract-first, pluginable, `is_durable`, no lifecycle authority |
| **P5** Extend continuation store | `ExecutionContinuationStateStore` | **Reject** | Violates separation: lifecycle truth ≠ work materialization; no aggregate-root evidence |

**Decision:** **P4** — new **`SuspendedExecutionOperationStore`** (ABC in contracts), implementations in `intergrax/runtime/execution/...` (in-memory for tests/dev; SQL/KV via providers). **No** ORM/DB/vendor types in contract.

## 8. Re-entry owner

**`ExecutionSuspendedWorkReentryPort`** (Protocol, EE-owned):

```text
authorized continuation (RESUMED) + grant evidence internal
  → load descriptor by (continuation_id, suspended_operation_id)
  → validate identity, tenant, operation_kind, materialization_state == BLOCKED
  → claim / CAS → CONSUMED (at-most-once logical re-entry)
  → reconstruct ExecutionBoundCatalogToolInvokeRequest (+ resolver from host registry)
  → L3 host → bridge → RuntimeToolInvoker
  → fresh governance + MSE
  → backend (idempotency via existing pre-effect coordinator)
```

- Registry of **`SuspendedOperationReentryStrategy`** keyed by `operation_kind` (typed, no reflection routing).
- Initial strategy: `execution_bound_catalog_tool` (L3-internal; may live under `intergrax/runtime/nexus/...` for materialization only).
- **No** UCA/CodeCraft types on generic port surface.

**Reuse check:** no existing EE “work rehydrator” port; `ExecutionLifecyclePort` is recovery handoff only — **new minimal port allowed** per ADR1 additive contracts.

## 9. Separation of concerns (canonical triple)

```text
ExecutionContinuationPort           = lifecycle (frozen)
SuspendedExecutionOperationStore    = durable work description
ExecutionSuspendedWorkReentryPort   = reconstruction + ToolRuntime invoke
```

Consumers (AW, UCA, CodeCraft) continue to call **L2 only**; host composition wires the three EE capabilities.

## 10. Cardinality

Evidence: `ExecutionContinuationStateStore` documents **many historical continuation episodes** per four-ID, **one current episode** at a time (`begin_current_episode_if_predecessor_allows`, `AMBIGUOUS_IDENTITY` fail-closed). Multi-continuation projection tests (GR-5-R3-R2) confirm multiple episodes per execution identity over time.

**Suspended operations:**

```text
Execution (four IDs)
  → 0..N continuation episodes (over time; 1 current)
  → per episode at REQUIRE_HITL: exactly 1 active BLOCKED catalog suspended operation
```

`continuation_id` ↔ `suspended_operation_id` is **1:1** for consumable blocked catalog work. Multiple **historical** descriptors (CONSUMED/ABANDONED) may exist per execution for audit.

`GovernedContinuationCorrelation.operation_id` remains the **HITL governance scope** id; it must **match** descriptor correlation established at pause but is **not** the store primary key.

## 11. Consistency model (no 2PC)

**Write ordering (W2 — persist on REQUIRE_HITL, preferred):**

1. **Insert** descriptor `PREPARED` (full payload + `continuation_id` + identity) — durable intent before stack unwind completes.
2. **Pause** via canonical bridge → `ExecutionContinuationPort` until `WAITING_FOR_HUMAN` (or equivalent paused gate).
3. **CAS** descriptor `PREPARED` → `BLOCKED` iff continuation snapshot matches `continuation_id` + identity + non-terminal lifecycle.

**Fail-closed recovery:**

| Partial state | Behavior |
|---------------|----------|
| `PREPARED`, no continuation | Reconcile → `ABANDONED`; no resume execution |
| `BLOCKED`, continuation missing / mismatch | Resume/reentry **denied** |
| Continuation waiting, no `BLOCKED` descriptor | **Deny** re-entry |
| Descriptor `CONSUMED` | No second re-entry (stale operation test) |
| Tampered payload / wrong execution | **Deny** |

Idempotent retries: store CAS + continuation CAS; reentry claim prevents multi-host duplicate workers (**claim ≠ authority**).

## 12. Restart & multi-host model

**Process restart (required):**

```text
Process A: code.exec → REQUIRE_HITL → PREPARED → pause → BLOCKED → terminate
Process B: load continuation → human approve → resume() → RESUMED
         → load BLOCKED descriptor → reentry port → reconstruct → ToolRuntime → MSE → effect → CONSUMED
```

Enterprise: durable provider (`store.is_durable is True`) required for restart qualification (mirror continuation store pattern).

**Multi-host:** supported with durable store + claim/CAS on re-entry.

## 13. Identity & HITL semantics

- **Same** TaskId, RunId, AttemptId, ExecutionId — **no** new root execution.
- **No** new identity types (`ReplayExecutionId`, etc.).
- Canonical **`invocation_scope_id`** (`dhr_*`) remains bridge-owned; descriptor does not mint scope.
- **Grant** remains canonical HITL artifact; not transported on L2; EE binds internally after resume.
- **Fresh** governance policy evaluation and **fresh MSE** after resume; approval ≠ ALLOW.
- Descriptor stores **no** stale ALLOW decision as authority.

## 14. ToolRuntime & idempotency

- Re-entry **must** pass through `RuntimeToolInvoker` (mandatory).
- Reuse **`idempotency_key`** from descriptor in reconstructed `ToolExecutionRequest`.
- Reuse **`IdempotencyPreEffectCoordinator`** / effect evidence for after-effect crash (effect committed, descriptor not yet CONSUMED).
- Logical single re-entry ≠ distributed exactly-once (UCA-8 future).

## 15. When to persist (W1 vs W2)

| Strategy | Verdict |
|----------|---------|
| **W1** Before every protected invoke | Reject for efficiency unless policy later requires |
| **W2** Only on `REQUIRE_HITL` | **Accept** — L3 host persists in bridge exception path **before** unwind; closes crash window per §11 |

Non-HITL fast path: **no** suspended-operation persistence (ADR1 non-HITL unchanged).

## 16. Observability & security

Emit (reuse EventBus contracts): operation suspended, descriptor persisted, continuation paused, human decision, reentry claim, reconstruction, governance re-eval, MSE, effect, descriptor consumed.

Trace correlation: ExecutionId, `continuation_id`, `invocation_scope_id`, `suspended_operation_id`.

Retention: until terminal continuation + execution completion; GC via provider strategy (tombstone vs delete configurable).

## 17. Migration (R6 phases)

1. Add contracts: descriptor, store, reentry port, `operation_kind` enum, catalog payload type.
2. L3 continuation-aware catalog host (ADR1): on `REQUIRE_HITL`, persist descriptor + wire continuation (remove TIGAE path per ADR1).
3. On `resume()`, EE coordinator invokes reentry port (not Worker, not new `RootExecutionLaunch`).
4. Default in-memory store for tests; durable provider for enterprise qualification.
5. Architecture regression gates + restart qualification test (§100 ADR2 task list).
6. TIGAE / `uca6c-scope` removal (ADR1 §13) — separate commits within R6.

## 18. Options matrix (summary)

| Option | Persistence owner | Reentry owner | New lifecycle? | Restart-safe | Generic? | Verdict |
|--------|-------------------|---------------|---------------:|-------------:|---------:|---------|
| Reuse Task checkpoint | Task | Task runner | No | Partial | No | **Reject** |
| Reuse Execution/ACP checkpoint | Agent | ACP resume | No | Partial | No | **Reject** |
| Extend continuation state | Continuation store | EE ad hoc | No | Yes | No | **Reject** (concern mix) |
| **New EE suspended-operation store** | **`SuspendedExecutionOperationStore`** | **`ExecutionSuspendedWorkReentryPort`** | **No** | **Yes** | **Yes** | **Accept** |
| Worker redispatch | AW ledger | Worker | No | No | No | **Reject** |
| CodeCraft replay store | CodeCraft | CodeCraft | No | No | No | **Reject** |
| In-memory callback | EE dev only | Callback | No | No | No | **Reject** for production |

## 19. Permanent architecture gates

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
```

## 20. R6 readiness

| Gate | Status |
|------|--------|
| Descriptor type & owner | **Defined** |
| Persistence contract | **Defined** (P4) |
| Reentry contract | **Defined** |
| Frozen core unchanged | **Yes** |
| Implementation | **UCA-6C-R6** (blocked until this ADR accepted) |

**R6 forbidden scope:** identity authority, `ExecutionContinuationPort` semantics, Governance/ToolRuntime core, public Nexus, AW/CodeCraft replay ownership, new universal L2 invoker.

## 21. Architecture gates (ADR2)

| Check | Result |
|-------|--------|
| Single canonical fulfillment flow preserved | **PASS** |
| Second continuation lifecycle | **NO** |
| Second HITL | **NO** |
| `ExecutionContinuationPort` unchanged | **YES** |
| EE owns durable work + reentry | **YES** |

---

**UCA-6C-ADR2 — Durable Suspended Operation & Execution Work Re-entry**
