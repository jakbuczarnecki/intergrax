# HARNESS-02-ADR1 — Canonical Execution Time Authority & Durable Deadline Semantics

| Field | Value |
| ----- | ----- |
| **Status** | **ACCEPTED** (architecture + contract design; implementation in HARNESS-02-R1+) |
| **Date** | 2026-09-19 |
| **Parent audit** | HARNESS-02 (`a1e22784b390285826c97906b6edd10e36b835a7` — ancestor of ADR baseline) |
| **Qualification catalog** | `tests/qualification/harness_02/catalog.py` |
| **R1 manifest** | [`HARNESS_02_R1_QUALIFICATION_MANIFEST.md`](../qualification/HARNESS_02_R1_QUALIFICATION_MANIFEST.md) |

---

## A. Problem

Independent HARNESS-02 qualification remains **BLOCKED** on three confirmed blockers:

| ID | Finding | Consequence |
| --- | --- | --- |
| **B1** | `global_deadline_monotonic` is not enforced before new child, tool, or LLM work | Expired monotonic authority can still start protected work on paths that skip `enforce_wall_time_budget` |
| **B2** | First physical tool attempt may start after `cancellation_requested` | Cooperative cancel is not checked in `_prepare_invocation` before side effect |
| **B3** | Durable resume/redelivery preserves ledger counters but remints full process-local monotonic deadline from `RunBudget` | Worker restart can grant a fresh wall allowance inconsistent with run SLA |

Additional debt **D1**: parallel authorities — `ActiveExecutionBudgetState.global_deadline_monotonic` vs `RuntimeState.started_at_utc` / `BudgetEnforcer.check_wall_time`.

Remediation must not be local `if monotonic > deadline` patches; one contract-driven model is required.

---

## B. Existing architecture

| Component | Location (current) | Role today |
| --- | --- | --- |
| `RunBudget.max_wall_time_seconds` | `intergrax/runtime/nexus/budget/budget_models.py` | Root policy input; re-read on every `bind_root_execution_budget` |
| `bind_root_execution_budget` | `intergrax/runtime/execution/active_execution_budget.py` | Mints `global_deadline_monotonic = monotonic() + max_wall_time` per bind |
| `ActiveExecutionBudgetState` | same | ContextVar carrier for ledger + process-local monotonic deadline |
| `peek_active_execution_global_deadline_monotonic` | same | Retry eligibility consumer (allowlisted) |
| `ChildExecutionRunner` | `intergrax/runtime/execution/child.py` | Inherits parent monotonic deadline; no pre-admission guard |
| `BudgetEnforcer.check_wall_time` | `intergrax/runtime/nexus/budget/budget_enforcer.py` | Nexus-internal wall check vs `RunBudget` limit |
| `enforce_wall_time_budget` | `intergrax/runtime/nexus/budget/budget_ticks.py` | Uses `RuntimeState.started_at_utc` elapsed wall clock |
| `ExecutionRetryEligibilityRequest` | `intergrax/contracts/execution_retry.py` | Carries `global_deadline_monotonic` + `now_monotonic` for backoff vs deadline |
| `RunBudgetPersistence` / `RunBudgetLedgerSnapshot` | `intergrax/runtime/execution/budget/persistence.py` | Durable **consumption** ledger (UE-9AR1); no execution deadline SSOT |
| `CancellationCoordinator` | `intergrax/runtime/cancellation/coordinator.py` | Task-metadata coordination; graph retry path wired |
| `ExternalOperationCancellationPort` | `intergrax/contracts/external_operation_cancellation.py` | Integration boundary; remains authoritative for in-flight external ops |
| Provider timeouts | LLM/integration contracts | Independent operation caps |

**Nexus** (`RuntimeState`, `BudgetEnforcer`, tool loop ticks) is an internal execution-engine implementation. Public consumers must not treat it as deadline owner.

---

## C. Requirements

### Functional

1. One **run-scoped** global execution wall-time authority, stable across attempts, redelivery, and worker restart.
2. Propagation to child/grandchild with **narrowing only** (`min` semantics).
3. Consumption by retry backoff (`min(policy_delay, remaining)` or deny).
4. Pre-effect gate before child admission, tool physical attempt, LLM call, integration/external op admission, and retry attempt scheduling.
5. Provider operation timeout `≤ remaining execution time` when a global deadline exists.
6. Typed outcomes: deadline expired vs cancelled vs available; fail-closed on corrupt/missing durable state on resume.

### Architectural

1. Neutral contracts under `intergrax/contracts/` (not Nexus-owned).
2. No global service locator (`GlobalDeadlineManager`, registry, `CurrentExecutionTimeoutService`).
3. Explicit scope ownership: durable SSOT vs process projection vs enforcement primitive.
4. `RunBudget.max_wall_time_seconds` remains **policy input**, not resume remint source.
5. No persist/reuse of raw `time.monotonic()` across processes.
6. Layer direction: `contracts → runtime execution → Nexus`; `contracts → LLM/integration adapters`.

---

## D. Options (durable representation)

### Model A — durable absolute UTC deadline (`deadline_at_utc`)

Persist timezone-aware UTC instant. On worker: `remaining = max(0, deadline_at_utc - now_utc)`; project `local_monotonic_deadline = monotonic() + remaining`.

### Model B — durable remaining wall budget (`remaining_wall_time_seconds`)

Persist a float budget; subtract active compute before checkpoint. Downtime/pause semantics are ambiguous and race-prone.

### Model C — hybrid (durable UTC authority + monotonic projection)

Same durable field as A; monotonic is **derived per process** and never persisted.

---

## E. Decision

**SELECTED MODEL: C (hybrid)** — durable **`deadline_at_utc`** (optional `None` = no global deadline) as SSOT; **`global_deadline_monotonic` on `ActiveExecutionBudgetState` becomes a runtime projection only**, recomputed on root bind, resume, and redelivery from SSOT + injected clocks.

**Rejected:**

| Alternative | Why rejected |
| --- | --- |
| Persist raw monotonic deadline | Semantically invalid across processes (HARNESS-02 B3 root cause) |
| Model B only (remaining float) | Unclear whether queue/crash time counts; harder to audit; CAS resume races |
| Model A without monotonic projection | Still need fast in-process checks; projection layer is required anyway |
| Remint `monotonic() + max_wall_time_seconds` on every attempt/bind | Violates run SLA (B3) |
| Independent deadline logic per subsystem | D1 debt; blocks unified pre-effect guard |
| `started_at_utc` + limit as sole SSOT | Observability field mixed with enforcement; skew vs monotonic bind |
| Store deadline only in budget ledger snapshot | Conflates consumption accounting with lifecycle authority (§51) |

---

## F. Contract ownership

| Concern | Owner | Public/neutral contract | Runtime implementation |
| --- | --- | --- | --- |
| Durable deadline SSOT | Platform execution lifecycle (run scope) | `ExecutionDeadlineAuthoritySnapshot` + `ExecutionDeadlineAuthorityPersistencePort` (`intergrax/contracts/execution_deadline/`) | `intergrax/runtime/execution/deadline_authority/` (new, R1) |
| Runtime projection | Active execution scope | `ExecutionDeadlineProjection` (immutable view: `deadline_at_utc`, `remaining_seconds`, `is_expired`) | Derived in `active_execution_budget` bind/resume helpers |
| Pre-effect guard | Platform execution (shared primitive) | `ExecutionProtectedWorkAdmissionPort` + `ExecutionProtectedWorkAdmissionResult` | `intergrax/runtime/execution/protected_work_admission.py` (new, R1) |
| Cancellation view | Cancellation domain | `ExecutionCancellationView` (read-only `is_cancelled`, optional reason) | Adapter from `CancellationCoordinator` + durable terminal store |
| Provider timeout derivation | Integration/LLM adapters | Existing provider timeout fields + `min(configured, remaining)` rule in adapter contract doc | Adapter wiring (R1 M5) |
| Persistence encoding | Platform storage | Versioned snapshot schema v1 | KV/document store parallel to `RunBudgetPersistence` keys |
| Retry deadline inputs | Retry policy | Extend `ExecutionRetryEligibilityRequest` with `deadline_at_utc` + `now_utc` (R1); deprecate monotonic fields after migration | `intergrax/runtime/execution/retry/policy.py` |
| Nexus wall ticks | Nexus (internal) | None public | `enforce_wall_time_budget` → compatibility shim to projection (M7) |

**Canonical owners (one line each):**

```text
Durable execution deadline SSOT = ExecutionDeadlineAuthoritySnapshot persisted per (tenant_id, run_id)
Process-local deadline projection = ActiveExecutionBudgetState.global_deadline_monotonic (derived only)
Pre-effect enforcement owner = ExecutionProtectedWorkAdmissionPort (contracts) implemented at runtime execution layer
```

---

## G. Lifecycle

| Event | Deadline behavior |
| --- | --- |
| **new run** | Resolve `max_wall_time_seconds` from policy → compute `deadline_at_utc = now_utc + limit` (or `None`) → **CAS-create** durable snapshot once → derive monotonic projection |
| **new attempt** | **Same** `deadline_at_utc`; reload SSOT; re-derive projection; never remint from `RunBudget` alone |
| **child** | Inherit parent `deadline_at_utc`; if child `requested_budget.max_wall_time_seconds = X`, effective `deadline_at_utc = min(parent, now_utc + X)` (UTC plane), then project monotonic |
| **grandchild** | `min` chain; never extend |
| **retry** | Eligibility uses durable/projected remaining; backoff capped by remaining; shared admission semantics |
| **worker restart** | Load SSOT → if expired, terminalize without new work; else project monotonic |
| **queue redelivery** | Same as resume; identical `deadline_at_utc` for `run_id` |
| **checkpoint resume** | Load SSOT before bind; fail-closed if missing/corrupt on existing run |
| **human pause / HITL** | **Global wall SLA keeps running** (enterprise default). Pausing *active compute* only is a future **separate** `ActiveExecutionTimeBudget` (type G), not mixed into global deadline |
| **host shutdown deadline** | Type D — caps worker drain only; does not extend run SLA |
| **detached work** | New run → new authority; same-run background → same run deadline |

### Clock semantics (explicit)

> **Does global execution wall-time run during worker crash, queue wait, pause, checkpoint, and human wait?**

**Yes** for crash, queue wait, checkpoint I/O, and human wait — the durable `deadline_at_utc` is wall-clock from root authority creation. **Pause does not freeze** the global SLA in this ADR. If product later requires freeze, introduce type **G** (`active_execution_time`) without overloading global deadline.

Precision: **microseconds** in UTC (`datetime` timezone-aware UTC); APIs may round to milliseconds for provider timeouts.

`None` deadline: **no global deadline configured** (unlimited wall SLA at execution layer), not “unknown”.

---

## H. Pre-effect enforcement

### Ordering (tool path — **selected**)

```text
registry / invoker bind
  → ExecutionProtectedWorkAdmission (deadline + cancel)   # hard invariant
  → governance / policy / MSE authorization
  → idempotency pre-effect claim (when semantic cost)
  → physical side effect (thread pool submit / provider call)
```

**Rationale:** B1/B2 require blocking before idempotency claims that consume keys or emit partial evidence; cancel must precede first tool attempt.

### Child admission

Guard **before** `grant_child_budget`, `mint_child_execution_id`, and delegate — owner: `ChildExecutionRunner` calls admission port first (R1 M3).

### Matrix

| Boundary | Guard before side effect? | Deadline | Cancel | Remaining-time propagation |
| --- | --- | --- | --- | --- |
| child | Yes (before budget grant) | Yes | Yes | Narrowed deadline to child context |
| tool | Yes (before submit/execute) | Yes | Yes | `min(provider, remaining)` |
| LLM | Yes (before provider call) | Yes | Yes | Request timeout from projection |
| integration | Yes (admission) | Yes | Yes | Same |
| external op | Yes (attempt admission) | Yes | Yes | Reuse `ExternalOperationCancellationPort` + time view |
| retry | Yes (before scheduling attempt) | Yes | Yes | Backoff cap |

Hard invariant: **expired global deadline → no new protected work** — not configurable via `DeadlinePolicy.allow_after_expiry()`.

---

## I. Cancellation integration

Separate authority, shared checkpoint:

```text
CancellationCoordinator / durable terminal metadata
        → ExecutionCancellationView (adapter)
        → ExecutionProtectedWorkAdmissionPort
```

Deadline and cancel storage remain distinct lifecycles; guard returns distinguishable results (`EXPIRED` vs `CANCELLED`). Tool code must not depend on `task.metadata["cancellation_requested"]` at steady state (adapter allowed in M4).

---

## J. Persistence (SSOT)

| Item | Value |
| --- | --- |
| **Storage contract** | `ExecutionDeadlineAuthorityPersistencePort` |
| **Document/KV partition** | `intergrax.run_execution_deadline.v1:{tenant_id}` |
| **Key** | `run_id` (validated `RunId`) |
| **Schema** | `schema_version: 1`, `deadline_at_utc: datetime \| null`, `authority_created_at_utc`, `policy_max_wall_time_seconds` (audit only) |
| **Initialization** | CAS on first root admission for new run; idempotent read on resume |
| **Concurrency** | Two workers → same loaded `deadline_at_utc`; projection may differ per process but remaining wall time is identical in UTC plane |
| **Legacy snapshots** | Ledger v1 without deadline record: on resume, **fail-closed** if `max_wall_time_seconds` was set and no authority record (no silent full SLA reset). New runs after R1 deploy create authority. Optional one-time migration job may derive `deadline_at_utc` from trusted `started_at_utc` + limit only when provably consistent — otherwise fail-closed (§110–111) |
| **TaskCheckpoint** | Identity only; not SSOT for deadline |

---

## K. Migration (legacy components)

| Step | Change | Risk | Tests |
| --- | --- | --- | --- |
| **M1** | Contracts: snapshot, projection view, admission port | Low | Contract immutability / import gates |
| **M2** | Durable persistence + root create/load on `ExecutionRuntime` | Medium | Root creates once; CAS |
| **M3** | Child pre-admission guard | Medium | Child/grandchild narrowing proofs |
| **M4** | Tool first-attempt guard + thread-pool before submit | High | H02-tool-pre-effect |
| **M5** | LLM/integration `min(timeout, remaining)` | Medium | Provider boundary gates |
| **M6** | Resume/redelivery reload SSOT → projection | High | UE-9 / H02-redelivery |
| **M7** | `enforce_wall_time_budget` delegates to projection; classify `started_at_utc` | Medium | Budget tick parity |
| **M8** | HARNESS-02-R1 qualification manifest | — | Full gate suite |

### `started_at_utc`

**Classification:** **A** observability/trace primary; **C** temporary migration hint only when deriving legacy authority under controlled migration; **not** canonical enforcement after M7.

### `BudgetEnforcer.enforce_wall_time_budget` path

**Role after migration:** **(2) compatibility adapter** delegating to canonical remaining seconds from projection until Nexus callers are routed through admission port; eventual **(4)** if wall dimension splits into type G budget.

### `RunBudget.max_wall_time_seconds`

Unchanged public semantics: requested limit at run creation. **Not** re-applied on resume to mint deadline.

### Backward compatibility

- Existing APIs keep `RunBudget` fields.
- `ExecutionRetryEligibilityRequest.global_deadline_monotonic` deprecated in favor of UTC fields in R1; monotonic accepted temporarily via adapter from projection.
- No silent behavior change: deploy notes + qualification gates document stricter resume fail-closed.

---

## L. Failure semantics

| Condition | Typed result |
| --- | --- |
| deadline expired | `ExecutionProtectedWorkAdmissionResult.EXPIRED` → `ExecutionFailureKind.DEADLINE_EXCEEDED` / terminalization |
| cancelled | `CANCELLED` → `ExecutionFailureKind.CANCELLED` |
| corrupt durable deadline | Fail-closed load error; no remint SLA |
| no deadline configured | `AVAILABLE` (no time cap at execution layer) |
| provider timeout | Provider error / retry classification; distinct from global deadline |
| unknown side effect after expiry in-flight | Existing external_operations reconciliation authoritative |

Evidence on exhaustion: `run_id`, `execution_id`, `deadline_at_utc`, `observed_at_utc`, `remaining_seconds=0`, reason code — via existing execution evidence paths.

---

## M. Qualification (R1)

See [`HARNESS_02_R1_QUALIFICATION_MANIFEST.md`](../qualification/HARNESS_02_R1_QUALIFICATION_MANIFEST.md).

---

## N. Risks

| Risk | Mitigation |
| --- | --- |
| Wall-clock skew / NTP step | UTC authority documented; monotonic for intra-process ordering only; skew affects all UTC consumers equally |
| Legacy resume without authority row | Fail-closed or explicit migration; no implicit remint |
| In-flight thread/provider after expiry | Intentional; block **new** work; late results cannot resume orchestration |
| Dual authority during migration | M7 retires `started_at_utc` enforcement as primary |
| Breaking persisted schema | `schema_version` + explicit migration; ADR required for breaking changes |

---

## O. Rejected alternatives (summary)

See §E. Additionally: coupling deadline into five snapshots; Nexus-owned public ABI; global registries; metadata dict as public cancel contract.

---

## P. Architecture boundaries

- Nexus remains internal.
- No provider → Nexus import for deadline.
- No global service locator.
- No cross-layer private field coupling.
- Contracts remain neutral and typed (no `dict[str, Any]` for known structures).

---

## Layer direction

```text
intergrax/contracts/execution_deadline/
        ↓
intergrax/runtime/execution/  (authority load, projection, admission)
        ↓
intergrax/runtime/nexus/    (consumes admission / projection adapters only)
```

---

## Planned neutral contracts (R1 skeleton)

Minimal surface (names frozen by this ADR):

- `ExecutionDeadlineAuthoritySnapshot` — durable, versioned
- `ExecutionDeadlineProjection` — `remaining_seconds`, `is_expired`, `deadline_at_utc`
- `ExecutionProtectedWorkAdmissionPort.assert_can_start_work(...) -> ExecutionProtectedWorkAdmissionResult`
- `ExecutionCancellationView` — `is_cancelled()`
- `ExecutionProtectedWorkAdmissionResult` — `AVAILABLE | EXPIRED | CANCELLED`

Clock injection: reuse platform UTC/monotonic providers where present; centralize in projection helper (no scattered `datetime.now()` / `time.monotonic()`).

---

## Remediation follow-up

Next implementation task: **HARNESS-02-R1 — Canonical Pre-Effect Time & Cancellation Enforcement** (per migration table).

---

## Audit requirement

Wprowadzona decyzja architektoniczna, ewentualne kontrakty oraz wynik HARNESS-02-ADR1 muszą zostać niezależnie zaudytowane na podstawie aktualnego kodu z GitHub przed rozpoczęciem implementacji HARNESS-02-R1.
