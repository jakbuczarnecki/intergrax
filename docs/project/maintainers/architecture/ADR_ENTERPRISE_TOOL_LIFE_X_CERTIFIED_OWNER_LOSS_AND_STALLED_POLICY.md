# ADR-ENTERPRISE-TOOL-LIFE-X-CERTIFIED-OWNER-LOSS-AND-STALLED-POLICY

| Field | Value |
| --- | --- |
| **Status** | **B1 ARCHITECTURE RATIFIED** (`OPERATOR APPROVED — L3-R5-B OPTION B1`, 2026-10-25 @ `141aaa3f…`) — **Gate A NOT PROVEN**; **L3 qualification incomplete**; **production delta = 0** @ this ratification task |
| **Task** | `TOOL-LIFE-X L3-R5` — Enterprise Architecture ADR (TL7/TL8) |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **BLOCKED** / **not CLOSED** |
| **Date** | 2026-10-24 |
| **Branch** | `development` |
| **START_HEAD** | `dd63629d75f28ce25e45c07c90e52a59bce814fe` |
| **L3-R5 remediation** | Independent audit R5-F01–F04 (2026-10-24) — this revision |
| **Operator authority** | TL7/TL8 enterprise-safe direction **pre-approved** (ten principles §3); this ADR **does not** reopen them |
| **Canonical roadmap** | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.5 |
| **Prior evidence (reuse)** | P0 [`TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md`](../qualification/TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md) · Batch C [`TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md`](../qualification/TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md) · Batch E [`TOOL_LIFE_X_BATCH_E_ADVERSARIAL_REAL_INFRASTRUCTURE_EVIDENCE.md`](../qualification/TOOL_LIFE_X_BATCH_E_ADVERSARIAL_REAL_INFRASTRUCTURE_EVIDENCE.md) · L3-R1 [`TOOL_LIFE_X_L3_R1_CONSOLIDATED_QUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R1_CONSOLIDATED_QUALIFICATION_EVIDENCE.md) · L3-R2 [`TOOL_LIFE_X_L3_R2_MCP_HOST_LIFECYCLE_QUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R2_MCP_HOST_LIFECYCLE_QUALIFICATION_EVIDENCE.md) |
| **Related ADR** | W4-C external operations · [`ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md`](ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md) · STATE-X recovery plane (separate concern) |
| **Does not** | Close `TOOL-LIFE-X` · start `DIST-SCALE-X` · implement recovery code · promote global `FRZ-*` PASS · authorize new distributed backend or blind takeover |

**Publication evidence:** [`TOOL_LIFE_X_L3_R5_ENTERPRISE_CERTIFIED_OWNER_LOSS_ADR_QUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R5_ENTERPRISE_CERTIFIED_OWNER_LOSS_ADR_QUALIFICATION_EVIDENCE.md) · **L3-R5-B ratification + Gate A prequalification:** [`TOOL_LIFE_X_L3_R5_B_ARCHITECTURE_RATIFICATION_AND_GATE_A_PREQUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R5_B_ARCHITECTURE_RATIFICATION_AND_GATE_A_PREQUALIFICATION_EVIDENCE.md) (publication SHA = containing commit on `development`).

---

## 1. Problem statement and business risk

Tool invocations that reach **physical `RUNNING`** on an external dependency (HTTP, MCP, DB, LLM, browser) can outlive the worker process that submitted them. Without a **single, typed, durable certified-owner-loss authority**, operators face:

- **Duplicate side effects** from blind retry after timeout or assumed worker death.
- **Indefinite `RUNNING`** or unauthorized takeover from token inequality, lease expiry, or shadow watchdogs.
- **Split truth** between idempotency leases, external-operation durable state, and Tool invocation terminals.
- **Tenant bleed** if recovery identity is not bound to tenant-scoped operation keys.

Batch C audit **C-F01–C-F04** @ remediation lineage proved that inferring owner death from `ProcessLocalExternalOperationOwner` mismatch or idempotency lease expiry is **unsafe**. Path B correctly **fail-closes** `recover_running_operation_after_certified_owner_loss` until this ADR’s contracts are implemented and qualified.

**Business risk:** silent duplication of invoices, double tool effects, corrupted execution terminals, and false “recovery succeeded” narratives during independent enterprise freeze.

---

## 2. Existing architecture inventory (@ START_HEAD)

| Concern | Canonical owner | Contract / module | Production-reachable @ HEAD | Notes |
| --- | --- | --- | --- | --- |
| Tool invocation lifecycle | `RuntimeToolInvoker` + Execution lifecycle | `ToolContract`, `ToolExecutionRequest` | Yes | No parallel Tool-only SM |
| Execution deadlines | Execution runtime + invoker | `execution_deadline` hierarchy §3.0.5 | Yes | Batch A/E evidence |
| External op durable state | `ExternalOperationStateStore` | `intergrax/runtime/external_operations/external_operation_state_store.py` | Yes (in-memory default; shared backends via composition) | Revision CAS |
| Physical ownership fencing | `external_operation_ownership` | `ProcessLocalExternalOperationOwner`, owner-matched terminal CAS | Yes | Token ≠ death |
| Cancellation intent | `request_operation_cancellation` | `ExternalOperationCancellationPort` | Yes | W4-C + Batch C |
| Recovery admission (UNKNOWN) | `prepare_external_operations_for_recovery` | `recovery_external_operation_gate.py` | Yes | Blocks RUNNING/UNKNOWN |
| RUNNING→UNKNOWN recovery API | `recover_running_operation_after_certified_owner_loss` | Same ownership module | Yes — **denies** on RUNNING (Path B) | C-F04 |
| Orphan sweep | `reconcile_orphaned_running_operations` | ownership + `tool_operation_recovery` | Yes — **no-op** | C-F01 |
| Pre-effect idempotency recovery | `IdempotencyStore` + `PreEffectSuspendedWorkRecoveryAuthority` | `idempotency_store.py` | Yes | **Pre-effect only** — not RUNNING external op |
| STALLED observation (deadline) | `is_stalled_running_external_operation` | `tool_operation_recovery.py` | Yes | **DEADLINE_ONLY**; observational |
| STALLED terminal projection | `ToolInvocationTerminalDisposition.STALLED` | same | Yes | Does not authorize takeover |
| Abandonment / UNKNOWN replay guard | `must_abandon_*` + Batch B | `ToolEffectCertainty` | Yes | TL5/TL6 scoped |
| STATE-X recovery interfaces | Checkpoint / continuation stores | ADR checkpoint admission | Yes | **Not** external-op owner-loss |
| Architecture gates (no bypass) | U5 zero-bypass tests | platform execution unification | Yes | TL14 |

**Conclusion (@ L3-R5 remediation):** W4-C **`ExternalOperationStateStore` + `external_operation_ownership`** remain the **durable physical-state owner**. **No** competing Tool lifecycle SM. However, **no sanctioned implementation today can mint or verify `CertifiedOwnerLossEvidenceV1`**, and the prior ADR’s `evidence_digest` + out-of-band `verify()` → `compare_and_set()` sequence **does not** establish issuer authenticity, atomic consumption, or tenant-safe shared-store access. **Classification: B — ARCHITECTURE DECISION REQUIRED** (§20) before **L3-R6** implementation.

---

## 3. Approved operator principles (frozen — not reopened)

1. Exactly one canonical **certified-owner-loss** authority.
2. Strongly typed contracts (`extra="forbid"`, no `Any`/`dict` semantic boundaries).
3. Durable, verifiable ownership evidence.
4. Atomic recovery transitions and monotonic fencing.
5. Tenant-scoped recovery identity.
6. Canonical reconciliation of uncertain effects.
7. No blind retry of potentially successful external operations.
8. No shadow execution authority, separate Tool lifecycle state machine, or competing watchdog.
9. Fail-closed when evidence is unavailable or insufficient.
10. **STALLED** remains observational and never automatically authorizes takeover.

---

## 4. Proposed canonical solution (narrowest enterprise-valid)

### 4.1 Authority graph

```text
Composition root (process / serving unit)
  └─ CertifiedOwnerLossEvidenceIssuerPort  (exactly one binding per deployment topology)
        └─ mint CertifiedOwnerLossEvidenceV1  (durable, tenant-bound, epoch-bound)
              └─ verify_certified_owner_loss_evidence(...)
                    └─ recover_running_operation_after_certified_owner_loss(..., evidence=...)
                          └─ ExternalOperationStateStore CAS: RUNNING → UNKNOWN (+ intent TERMINATING)
                                └─ reconcile_external_operation_outcome(...)  [provider capability gated]
```

**Semantic owner of RUNNING/UNKNOWN/terminal physical state:** unchanged — **`ExternalOperationStateStore`** via `external_operation_ownership` helpers.

**Semantic owner of “may transition RUNNING→UNKNOWN for recovery”:** **`ExternalOperationStateStore` linearization** — the **only** authority that may **commit** the transition. **`CertifiedOwnerLossEvidenceIssuerPort`** may **stage** loss certification into the **same durable record** (never caller-supplied bypass). **No** freestanding verifier may authorize CAS without store-enforced one-time consumption (§20.2).

**Explicit non-authorities (forbidden as proof of death):**

| Candidate | Verdict |
| --- | --- |
| Expired idempotency lease | **Not** proof external work stopped |
| `owner_token` mismatch across processes | **Not** proof of death (C-F01) |
| `reconcile_orphaned_*` sweep | **No-op** — must stay no-op |
| `PreEffectSuspendedWorkRecoveryAuthority` | Pre-effect ledger only |
| Deadline exceeded / STALLED signal | Observation only — **no** takeover (TL7) |
| Separate Tool watchdog SM | **Forbidden** |

### 4.2 Typed contracts (planned modules)

**Module (implementation batch L3-R6-A):** `intergrax/contracts/certified_owner_loss_evidence.py`

```python
# Illustrative — names frozen by this ADR; implementation follows in L3-R6

class ExternalOperationOwnershipEpochV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation_id: str
    tenant_id: str
    owner_token: str
    store_revision: int  # revision at evidence bind time

class CertifiedOwnerLossEvidenceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str  # UUID
    epoch: ExternalOperationOwnershipEpochV1
    issuer_id: str  # composition-scoped issuer identity
    issued_at: datetime  # UTC
    monotonic_fence: int  # strictly increasing per (tenant_id, operation_id)
    loss_reason: CertifiedOwnerLossReason  # enum: WORKER_LOST | LEASE_EXPIRED_WITH_LIVENESS_PROOF | OPERATOR_DECLARED | ...
    liveness_proof_ref: str | None  # opaque handle to verifier-owned proof bundle
    evidence_digest: str  # integrity fingerprint only — NOT issuer authenticity (§20.1)

class CertifiedOwnerLossEvidenceIssuerPort(Protocol):
    def issue_owner_loss_evidence(
        self,
        *,
        epoch: ExternalOperationOwnershipEpochV1,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None,
    ) -> CertifiedOwnerLossEvidenceV1: ...

class CertifiedOwnerLossEvidenceVerifierPort(Protocol):
    """Deprecated shape — MUST NOT be used as a separate pre-CAS gate (§20.2)."""

    def verify(
        self,
        evidence: CertifiedOwnerLossEvidenceV1,
        *,
        expected_tenant_id: str,
        expected_operation_id: str,
    ) -> None: ...  # raises CertifiedOwnerLossEvidenceRejectedError
```

**Store-native API (mandatory for L3-R6 — single linearization owner):**

Extend `ExternalOperationStateStore` (same durability backend as W4-C):

1. `stage_certified_owner_loss_evidence(tenant_id, operation_id, expected_revision, staged: StagedCertifiedOwnerLossEvidenceV1) -> ExternalOperationState` — **issuer-only** composition path; CAS writes staged slot on RUNNING record.
2. `apply_certified_owner_loss(tenant_id, operation_id, expected_revision, evidence_id: str) -> ExternalOperationState` — **one** CAS: validate tenant + revision + RUNNING + staged `evidence_id` unconsumed + epoch fields match staged payload + monotonic fence; consume evidence; transition physical `UNKNOWN`, intent `TERMINATING`; clear or fence `owner_token` per W4-C.

`recover_running_operation_after_certified_owner_loss(...)` becomes a thin wrapper over **`apply_certified_owner_loss`** only (no separate verify).

**Preconditions (all enforced inside store CAS — not application-only):**

- `tenant_id` matches durable `ExternalOperationState.tenant_id` (new mandatory field — §20.3).
- `physical_state == RUNNING`.
- Staged evidence present, `evidence_id` not in `consumed_certified_loss_evidence_ids`.
- `staged.epoch` matches live `owner_token`, `store_revision`, `operation_id`, `tenant_id`.
- `issuer_id` ∈ `CompositionCertifiedOwnerLossIssuerBinding.allowed_issuer_ids` (composition config — not secrets/crypto).
- **Post:** UNKNOWN physical; intent TERMINATING; evidence marked consumed.

**Idempotency:** Re-`apply` with same consumed `evidence_id` → return current record if already UNKNOWN from that certification; conflicting terminal → `ExternalOperationFencedCommitRejectedError`.

### 4.3 Persistence and atomicity

| Element | Rule |
| --- | --- |
| Durable schema | **Mandatory** on `ExternalOperationState`: `tenant_id: str`; `ownership_epoch_fence: int | None`; `staged_certified_owner_loss: StagedCertifiedOwnerLossEvidenceV1 | None`; `consumed_certified_loss_evidence_ids: frozenset[str]` (or normalized store equivalent) |
| Store key | **All** `load` / `create_if_absent` / `compare_and_set` / `list_running` / certification methods take **`tenant_id` + `operation_id`** — no cross-tenant load by bare `operation_id` |
| Transaction boundary | **Forbidden:** `verifier.verify()` then `compare_and_set()` as two observable steps. **Required:** `stage_*` and `apply_*` each use **one** store CAS (or backend-native single transaction implementing equivalent checks) |
| Linearization | **`apply_certified_owner_loss` CAS** (revision + evidence consumption + transition) |
| Epoch semantics | `monotonic_fence` increases only on successful `apply_certified_owner_loss` |
| Concurrent recovery | Loser CAS → `StaleExternalOperationStateError`; re-stage if still RUNNING |
| Crash consistency | After successful `apply`, UNKNOWN survives restart; after failed CAS, RUNNING unchanged — **no** “verified but not applied” state |
| Store partition / quorum loss | Fail-closed — no transition; no optimistic UNKNOWN |
| Compatibility | Protocol extension + migration: backfill `tenant_id` at create from admission context; shared-store deployments **BLOCKED** until tenant-scoped protocol ships |

### 4.4 Recovery and reconciliation

| State | Meaning | Permitted after certified loss |
| --- | --- | --- |
| `RUNNING` | Physical work may be in flight | → `UNKNOWN` **only** with verified evidence |
| `UNKNOWN` | Outcome uncertain | Reconciliation required before retry or terminal commit |
| `ABANDONED` (tool terminal) | Policy/abandonment path | No replay of side effect |
| Reconciliation | Provider outcome query / idempotency oracle / operator runbook | May promote `UNKNOWN` → `SUCCEEDED`/`FAILED`/`CANCELLED` with **owner-matched or reconciliation authority** — separate from loss certification |

**UNKNOWN vs ABANDONED:** `UNKNOWN` is **physical** durable state on external-op store; `ABANDONED` is **tool invocation terminal** when policy forbids retry (Batch B/C). Certified loss leads to **UNKNOWN physical**, not automatic ABANDONED tool terminal.

**No unauthorized replay:** Invoker MUST NOT start new physical attempt while physical `UNKNOWN` unless reconciliation explicitly permits (existing Batch B guards).

**Providers without outcome-query:** Fail-closed to `UNKNOWN` + operator/runbook terminalization; **no** fabricated success.

### 4.5 STALLED policy (TL7)

| Mode | Status |
| --- | --- |
| **DEADLINE_ONLY (default)** | **Approved** — `is_stalled_running_external_operation(..., parent_execution_deadline_exceeded=True)` |
| Heartbeat / progress STALLED | **OPERATOR DECISION REQUIRED** for any future implementation — separate certification; **not** part of L3-R6 default |
| Takeover on STALLED alone | **Forbidden** |
| Interaction with TL8 | STALLED may trigger **observability** and human/operator workflow; **must not** call recovery without `CertifiedOwnerLossEvidenceV1` |
| Terminal / late result | Existing fenced terminal rules (Batch C) — late callback cannot override mismatched owner |

### 4.6 Replaceability and enterprise boundaries

- Public contracts in `intergrax/contracts/` only.
- Issuer implementations are replaceable (process supervisor, orchestrator, cloud lease — **composition-bound**).
- Governance ≠ Execution: certification does not grant new tool authorization.
- Fail-closed unsupported configs: missing issuer binding → recovery API continues to raise `ExternalOperationCertifiedOwnerLossRequiredError`.
- Tenant isolation: wrong-tenant evidence rejected at verifier.

---

## 5. Rejected alternatives

| Alternative | Rejection reason |
| --- | --- |
| A — Auto RUNNING→UNKNOWN on lease expiry | Violates principles 7, 9; Batch C C-F04 |
| B — Second Tool lifecycle state machine + watchdog | Violates principle 8; FRZ-BND-03 |
| C — Idempotency store as death detector | Violates principle 9; wrong plane |
| D — Implicit takeover on STALLED/deadline | Violates principles 7, 10 |
| E — New distributed consensus backend (mandated) | Not required for semantics; **BLOCKED** as default implementation — physical qual separate |
| F — Separate “RecoveryManager” orchestrating tools | Shadow authority — forbidden |

---

## 6. Contract definitions and ownership map

See §4.1–4.2. **Composition owner** for issuer binding: same owner as `ExternalOperationStateStore` wiring in production composition (`ProductionProcessComposition` / harness equivalents) — **one** issuer port per serving unit.

---

## 7. Atomicity / transaction / fencing model

**Linearization model (closed world @ HEAD):**

```text
[Issuer] stage_certified_owner_loss_evidence  →  CAS#1 (RUNNING, staged slot)
[Recovery] apply_certified_owner_loss         →  CAS#2 (consume evidence + RUNNING→UNKNOWN)
[Worker]   mark_physical_terminal_if_owner_matches → CAS (owner_token match)
```

Certification validity, owner epoch (`owner_token` + `store_revision`), `monotonic_fence`, and one-time `evidence_id` consumption **MUST** be checked inside **CAS#2** — not in a prior call.

**Equivalence note:** A two-phase `stage` + `apply` is admissible only because **neither phase alone** authorizes UNKNOWN; only **CAS#2** does. A caller-constructed `CertifiedOwnerLossEvidenceV1` without prior **CAS#1** from an allowed issuer binding **MUST** fail.

Fencing: existing `owner_token` terminal rules (Batch C) plus post-loss UNKNOWN reconciliation gates.

---

## 8. Tenant isolation model

**@ HEAD — gap (R5-F03):** `ExternalOperationState` and `ExternalOperationStateStore` use **`operation_id` only**; `ExternalOperationIdentity` / `mint_stable_operation_id` bind `execution_id` but **do not** embed `tenant_id`. Admission layers (`ExternalOperationAdmissionContext.tenant_id`) enforce tenant on **admit** paths; **durable external-op records are not tenant-keyed**.

**Minimum shared-store contract (mandatory before multi-tenant shared durability):**

- Persist `tenant_id` on every external-op record at `create_if_absent`.
- Every store API requires explicit `tenant_id`; reject load/CAS when record tenant ≠ caller tenant.
- `list_running(tenant_id)` — no global unscoped scans in production composition.
- Certificates **cannot** be the sole tenant guard if bare `operation_id` remains globally addressable.

**Defense in depth:** `epoch.tenant_id` must match durable `tenant_id` inside `apply_certified_owner_loss`.

Evidence digests MUST NOT embed secrets; `liveness_proof_ref` is opaque and **does not** substitute store staging.

---

## 9. Recovery and reconciliation semantics

See §4.4. Reconciliation owner: existing provider ports + tool invoker policy — **downstream** of UNKNOWN admission.

### 9.1 Distinct facts (no silent equivalence)

| # | Fact | Meaning | May imply |
| --- | --- | --- | --- |
| 1 | **Ownership authority lost** | Sanctioned certified owner-loss committed on store | **Not** provider terminated |
| 2 | **Platform record fenced** | `owner_token` / revision CAS rejected stale worker | **Not** loss certification |
| 3 | **Provider operation physically terminated** | Provider port / observation says stopped | **Not** business outcome |
| 4 | **Business outcome observed** | Typed terminal or reconciliation oracle | **Not** automatic retry right |
| 5 | **Retry authorized** | Canonical policy + idempotency class + invoker | Requires explicit gates |

Ambiguous provider effect → remain **UNKNOWN**; no blind replay; no Governance bypass; no new retry owner without sanctioned policy.

---

## 10. Failure / partition / restart behavior

| Event | Behavior |
| --- | --- |
| Worker alive but slow | Remain RUNNING; STALLED observational only |
| Store unavailable | Fail-closed — no transition |
| Split brain recovery | CAS + fence rejects stale evidence |
| Restart mid-CAS | After commit: UNKNOWN; before commit: RUNNING unchanged |
| Issuer unavailable | No evidence minted — recovery denied |

---

## 11. Complete adversarial test matrix

Columns: **Case** | **Expected durable state** | **Permitted transitions** | **Forbidden actions** | **Canonical owner** | **Mandatory evidence**

| Case | Expected durable state | Permitted | Forbidden | Owner | Evidence |
| --- | --- | --- | --- | --- | --- |
| Worker alive but slow | RUNNING | heartbeat obs (future), deadline STALLED signal | RUNNING→UNKNOWN, replay | Invoker worker | None |
| Worker crashed before physical submit | ACTIVE intent, not RUNNING | cancel / fail attempt | certified loss | Invoker | None (not RUNNING) |
| Worker crashed after physical submit | RUNNING until certified | RUNNING→UNKNOWN with evidence | blind retry | Store + issuer | `CertifiedOwnerLossEvidenceV1` |
| Provider committed, ack lost | RUNNING or UNKNOWN after reconcile | reconcile→terminal | auto SUCCESS retry | Reconciliation | Provider oracle + idempotency |
| Lease expired, work continues | RUNNING | wait / reconcile | lease expiry alone → UNKNOWN | Store | Liveness proof **required** in evidence |
| Network partition | RUNNING or UNKNOWN | evidence only after partition resolution policy | dual writer terminals | CAS owner | Issuer liveness proof |
| Simultaneous recovery attempts | one wins CAS | one UNKNOWN transition | two successful CAS | Store revision | Distinct evidence; one fence wins |
| Stale owner terminal commit | RUNNING held by live owner | live owner terminal | stale owner commit | Owner token match | N/A |
| Stale certification evidence | RUNNING | re-issue evidence | apply stale fence | Verifier | Fresh evidence |
| Wrong tenant | unchanged | reject evidence | apply evidence | Verifier | Valid tenant binding |
| Wrong operation | unchanged | reject | CAS | Verifier | op id match |
| Old ownership epoch | unchanged | reject | transition | Verifier | revision+token match |
| Duplicate certification | UNKNOWN (idempotent) | return current | second CAS changing terminal | Store | Same fence idempotent |
| Cancellation vs recovery | CANCELLATION_REQUESTED / TERMINATING | cancel path precedence per W4-C | loss+cancel race without CAS | Cancellation coordinator | Evidence still required for UNKNOWN from RUNNING |
| Timeout vs recovery | tool TIMED_OUT projection | uncertain effect path | timeout→auto UNKNOWN | Invoker policy | Not substitute for loss evidence |
| Crash during CAS | RUNNING or UNKNOWN | recovery retry idempotent | assume success | Store | Re-verify revision |
| Restart during reconciliation | UNKNOWN persists | continue reconcile | auto replay | Reconciliation | Prior UNKNOWN record |
| Unavailable shared store | last committed | fail-closed | guess | Store | N/A |
| Unavailable provider | UNKNOWN or RUNNING | abandon policy / wait | fake reconcile OK | Policy | Operator runbook |
| Provider no outcome-query | UNKNOWN | manual terminalization | automated SUCCESS | Fail-closed | Operator |
| Effect committed once, response lost | UNKNOWN→reconcile | reconcile once | duplicate POST | Idempotency + reconcile | External oracle |
| Late terminal callback | terminal set | fenced ignore/reject | resurrect RUNNING | Owner fence | Terminal record |
| Idempotency conflict | ledger conflict | fail closed | replay | Idempotency store | N/A |
| Reconciliation UNKNOWN | UNKNOWN | stay UNKNOWN / ABANDON tool | blind retry | Invoker | Reconcile result |
| Takeover without replay permission | RUNNING or ABANDON | ABANDON tool terminal | new physical attempt | Governance + invoker | No loss evidence ≠ replay right |

### 11.2 R5 extended negative-case matrix (L3-R6 design gate)

Columns: **Precondition** | **Semantic owner** | **Durable state** | **Permitted** | **Forbidden** | **Linearization** | **Expected error** | **Qual evidence**

| Case | Precondition | Semantic owner | Durable state | Permitted | Forbidden | Linearization | Expected error | Qual evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Forged digest | Caller builds payload | Store | RUNNING | reject apply | UNKNOWN transition | `apply_certified_owner_loss` | `CertifiedOwnerLossEvidenceRejectedError` / not staged | L3-R7 |
| Tampered issuer identity | `issuer_id` ∉ allowlist | Store | RUNNING staged | reject apply | commit | CAS#2 | rejected | L3-R7 |
| Cross-tenant evidence | `epoch.tenant_id` ≠ record | Store | RUNNING | reject | apply | CAS#2 | tenant mismatch | L3-R7 + TEN scoped |
| Wrong operation | `evidence_id` for other op | Store | RUNNING | reject | CAS | CAS#2 | not found / mismatch | L3-R7 |
| Replayed certificate | `evidence_id` consumed | Store | UNKNOWN or RUNNING | idempotent return if same UNKNOWN | second consume | CAS#2 | rejected / no-op | L3-R7 |
| Expired certificate | `issued_at` > TTL policy | Store | RUNNING staged | re-stage | apply stale | CAS#2 | expired | L3-R7 + operator TTL |
| Duplicate consumption | two appliers same id | Store | RUNNING→UNKNOWN once | one winner | double UNKNOWN flip | CAS#2 | stale / rejected | L3-R7 |
| Recovery CAS vs terminal | terminal committed | Invoker + store | SUCCEEDED/FAILED | reconcile only | RUNNING→UNKNOWN | terminal CAS | fenced / rejected | Batch C |
| Worker alive after lease expiry | idempotency lease gone | Invoker | RUNNING | observe STALLED | certify loss w/o proof | — | deny stage | Batch C + L3-R7 |
| Worker revival during cert | old worker resumes | Store + owner | RUNNING | live owner terminal | stale owner apply | owner-matched CAS | `ExternalOperationFencedCommitRejectedError` | Batch C |
| Cert accepted before store fails | N/A — no split accept | Store | — | — | verify-without-CAS | single CAS only | — | ADR §20.2 |
| Two concurrent verifiers | dual apply | Store | RUNNING | one CAS win | two UNKNOWN | CAS#2 | `StaleExternalOperationStateError` | L3-R7 |
| Store partition | quorum loss | Store | last committed | fail-closed | guess UNKNOWN | none | propagate unavailable | L3-R8 |
| Stale epoch after restart | revision advanced | Store | RUNNING | re-stage | old staged apply | CAS#2 | stale revision | L3-R7 |
| Partial persistence migration | missing `tenant_id` | Store | legacy row | fail create in shared mode | cross-tenant load | load | fail-closed | migration qual |
| Provider committed, ack lost | timeout | Reconciliation | RUNNING/UNKNOWN | reconcile | auto SUCCESS retry | reconcile CAS | UNKNOWN until oracle | L3-R1 / Batch B |
| Reconciliation UNKNOWN | oracle silent | Reconciliation | UNKNOWN | stay UNKNOWN / ABANDON policy | blind retry | invoker policy | blocked replay | Batch B |
| Unauthorized new physical attempt | UNKNOWN w/o policy | Invoker | UNKNOWN | reconcile first | new attempt | invoker gate | blocked | Batch B |

---

## 12. Implementation impact graph

```text
L3-R5-B  operator ratification of §20 option B1 (issuer + tenant-scoped store)
    → L3-R6-A  contracts + store protocol extension (tenant_id, stage/apply)
        → L3-R6-B  apply_certified_owner_loss + recover_* wrapper
            → L3-R6-C  InMemory + migration guards
                → L3-R6-D  composition issuer allowlist binding (fail-closed)
                    → L3-R7  §11 + §11.2 adversarial pytest
                        → L3-R8  issuer liveness proofs (operator Docker — TL10–13 blockers preserved)
```

**L3-R6 entry:** **was prohibited** until **B — ARCHITECTURE DECISION REQUIRED** closed; **L3-R5-B ratified §21** (2026-10-25). **R6–R10 implementation landed before ratification record** — historical ADR prohibition preserved in §20 audit text; **independent implementation acceptance and Gate A remain separate** from architecture ratification.

**Touches (planned):** `external_operation_ownership.py`, `external_operation_state_store.py`, `contracts/external_operation_cancellation.py` (if record extended), production composition factories, qualification docs.

**Does not touch:** `DIST-SCALE-X`, new Tool SM, idempotency lease semantics.

---

## 13. Migration and compatibility rules

- Path B behavior remains until issuer wired — default **deny**.
- Callers of `recover_running_operation_after_certified_owner_loss` gain required `evidence` parameter in implementation batch — compile-time break acceptable in enterprise cycle.
- Historical Batch C deny tests remain valid.

---

## 14. Implementation batch plan

| Batch | Deliverable | Classification |
| --- | --- | --- |
| **L3-R5-B** | Operator ratification of §20.4 recommended option | **RATIFIED** (§21) — **not** Gate A / **not** L3 parent closure |
| **L3-R6** | Store-native stage/apply + tenant-scoped protocol + composition binding | **IMPLEMENTED** (evidence R6) — **VALIDATION REQUIRED** / independent audit |
| **L3-R7** | §11 adversarial unit/integration tests | VALIDATION REQUIRED |
| **L3-R8** | Issuer liveness proof with approved Docker/supervisor qual | VALIDATION REQUIRED (operator infra) |
| **L3-R9** | TL8 positive path closure evidence + FRZ-REL-05 scoped update proposal | VALIDATION REQUIRED |

---

## 15. Validation L1 / L2 / L3

| Level | Scope | Command policy |
| --- | --- | --- |
| **L1** | New module unit tests (verifier, fence, deny paths) | Single `uv run pytest` path per test-selection record |
| **L2** | Consolidated TOOL-LIFE Batch C + B + W4-C + new L3-R7 matrix | One pytest invocation |
| **L3** | Real issuer liveness + PG/MCP physical qual (existing blockers) | Operator-authorized Docker only |

**Test-selection (mandatory before any pytest in implementation batches):** (1) production delta non-zero → yes for L3-R6+; (2) behavior change → recovery API; (3) scope → listed modules; (4) reuse Batch C denies; (5) no broad suite; (6) logs under `.tmp/session/TOOL-LIFE-X-L3-R6/`.

---

## 16. Applicable FRZ / ECAP

| ID | ADR disposition |
| --- | --- |
| **FRZ-REL-05** | OPEN — this ADR is **architecture input**; scoped PASS only after L3-R7 evidence |
| **FRZ-REL-01** | TL7 DEADLINE_ONLY observational — scoped partial @ Batch C |
| **FRZ-REC-07** | Supporting — UNKNOWN modeling aligns; no global PASS |
| **FRZ-BND-03** | No second lifecycle owner — **preserved** |
| **ECAP-P1..P4** | Documentation task — no self-PASS |
| **ECAP-P6** | Corrective scope proven by C-F04 |

---

## 17. TL1–TL15 closure matrix (@ L3-R5 remediation)

| Criterion | Semantic owner | Composition owner | Contract | Valid SHA | Classification | Missing action | Test | FRZ | Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TL1 | RuntimeToolInvoker | `runtime_tool_invoker_composition` | ToolContract timeout | Batch A | ALREADY PROVEN | — | Batch A L2 | FRZ-REL-01 scoped | — |
| TL2 | Execution runtime | Execution composition | deadline hierarchy | Batch A/E | ALREADY PROVEN | — | consolidated | FRZ-REL-02 scoped | — |
| TL3 | Cancellation + ownership helpers | Invoker wiring | W4-C cancel ports | Batch C + W4 | ALREADY PROVEN (scoped) | PG/MCP physical cancel | W4-C | FRZ-REL-04 scoped | TL9 |
| TL4 | RuntimeToolInvoker | Invoker composition | ToolRetryPolicy | Batch B | ALREADY PROVEN | — | Batch B | FRZ-REL-07 scoped | TL5 |
| TL5 | Invoker + idempotency | Production composition | ToolEffectCertainty | B + L3-R1 | ALREADY PROVEN (scoped) | runbooks | L3-R1 | FRZ-REL-03 scoped | TL6 |
| TL6 | Invoker | Production composition | SideEffectRetrySafety | Batch B | ALREADY PROVEN | retry re-auth matrix | Batch B | FRZ-GOV scoped | TL5 |
| TL7 | tool_operation_recovery | Invoker | STALLED DEADLINE_ONLY | Batch C @ `dd63629d…` | VALIDATION REQUIRED | heartbeat policy | `is_stalled_*` | FRZ-REL-01 scoped | ADR §4.5 |
| TL8 | external_operation_ownership + store CAS | **Unbound** issuer | stage/apply certified loss | Batch C deny @ HEAD | **OPERATOR DECISION REQUIRED** | §20 ratification + L3-R6 | deny today; L3-R7 positive | FRZ-REL-05, FRZ-REC scoped | L3-R5-B |
| TL9 | ExternalOperationStateStore | Invoker optional store | fenced terminal CAS | Batch C | ALREADY PROVEN (scoped) | tenant-scoped store protocol | Batch C | FRZ-REL-05 scoped | TL8 |
| TL10 | SQL tool adapters | Tool composition | row/byte limits | D + E partial | VALIDATION REQUIRED | Docker PG | D/E | FRZ-REL-10 scoped | **BLOCKED** C10 |
| TL11 | MCP host bridge | Host composition | ToolContract path | L3-R2 | VALIDATION REQUIRED | hung-server qual | L3-R2 | FRZ-REL-06 scoped | **BLOCKED** |
| TL12 | HTTP tools | Invoker | AllowlistHttpClient | L3-R1 | VALIDATION REQUIRED | lost-ack | L3-R1 | FRZ-REL-06 scoped | — |
| TL13 | Cross-provider qual | Platform qual | real infra matrix | E partial | VALIDATION REQUIRED | Docker PG+MCP | Batch E | FRZ-REG scoped | TL10–11 |
| TL14 | Arch gates | Platform | single invoker | A–D + L3-R1 | ALREADY PROVEN | — | U5 gates | FRZ-BND-03 scoped | — |
| TL15 | TOOL-LIFE-X parent | Roadmap §3.0.5 | EC1–15 rollup | P0 + waves | BLOCKED | TL8 + TL10–13 | parent L3 | — | all |

---

## 18. Remaining parent blockers

| Blocker | Class | Notes |
| --- | --- | --- |
| Certified owner-loss issuer + store protocol | **RATIFIED (B1)** | §21 — implementation evidence R6–R10; **Gate A OPEN** |
| TL8 positive recovery | **BLOCKED** | Positive RUNNING→UNKNOWN **not authorized**; production fail-closed |
| PostgreSQL / Docker physical qual | **BLOCKED** (operator C10) | TL10/TL13 preserved |
| MCP hung-server physical qual | **BLOCKED** | TL11 preserved |
| TL7 heartbeat policy | OPERATOR DECISION REQUIRED | Optional; DEADLINE_ONLY default |
| `TOOL-LIFE-X` parent CLOSED | BLOCKED | TL8 + TL10–13 + TL15 |

---

## 19. Post-step enterprise discovery and roadmap completeness

- **DIST-SCALE-X:** not started — excluded.
- **Semantic authority:** RUNNING/UNKNOWN/terminal owner remains W4-C store; **new** sanctioned surfaces: tenant-scoped store protocol + composition-bound issuer allowlist + store-native certification CAS — **require ratification** (not self-closed by this doc).
- **Ratification:** **B1 ratified** §21 (2026-10-25); **does not** close Gate A, TL8, or TOOL-LIFE-X parent.
- **Tenant Isolation Audit (TOOL-LIFE-X L3-R5 scope):** **FAIL** for shared `ExternalOperationStateStore` until `tenant_id` durable keying ships — admission-only tenant checks insufficient for cross-tenant evidence (R5-F03). Global **TENANT-X CLOSED** does not auto-pass this gap.
- **FRZ / ECAP:** **new global FRZ PASS = 0**; FRZ-REL-05 / FRZ-REC remain architecture inputs; FRZ-BND-03 preserved; ECAP P1/P2/P4 docs-only — no self-PASS.

---

## 20. L3-R5 independent audit remediation (R5-F01–F04)

### 20.1 R5-F01 — Certificate authenticity (`evidence_digest`)

| Threat | Enforceable boundary (no new crypto/CA) |
| --- | --- |
| Caller-constructed evidence | **Reject** unless `evidence_id` present in **staged** slot from **CAS#1** by allowlisted issuer |
| Tampered payload | `evidence_digest` = integrity check **of staged bytes only** inside CAS#2 |
| Forged issuer | `issuer_id` must match `CompositionCertifiedOwnerLossIssuerBinding` |
| Replay | `consumed_certified_loss_evidence_ids` on record |
| Stale | `store_revision` + `owner_token` + `monotonic_fence` + optional `issued_at` TTL |
| Wrong deployment | `issuer_id` includes `serving_unit_id` from composition binding |
| Wrong tenant | durable `tenant_id` + epoch tenant match in CAS#2 |

**Disposition:** Integrity field **alone** is insufficient — **OPERATOR DECISION REQUIRED** to ratify **B1** staging model; until then **prohibit L3-R6**.

**Reuse:** W4-C revision CAS + Batch C owner fencing; **not** idempotency lease as death proof.

### 20.2 R5-F02 — Atomic certification and CAS

Prior ADR `verify(evidence) → compare_and_set(...)` is **rejected** — crash between calls yields false confidence; two callers are not linearizable.

**Required:** `apply_certified_owner_loss` single CAS checks validity + consumption + transition. Optional `stage_*` is **not** authority to enter UNKNOWN.

**Scenarios:** concurrent recovery → one CAS winner; worker revival → fenced terminals; cancellation vs recovery → W4-C intent precedence inside same store; terminal vs recovery → terminal blocks loss apply; store failover → fail-closed.

**Disposition:** Architecture correction **in this ADR**; implementation **BLOCKED** until L3-R5-B.

### 20.3 R5-F03 — Tenant-safe persistence

| Artifact @ HEAD | Tenant binding |
| --- | --- |
| `ExternalOperationIdentity` | No `tenant_id` field |
| `mint_stable_operation_id` | Hashes `execution_id` only |
| `ExternalOperationState` | `operation_id` only |
| `ExternalOperationStateStore` | Single-key `operation_id` |
| Admission / audit | `tenant_id` enforced on admit — **not** on durable load |

**Disposition:** **OPERATOR DECISION REQUIRED** to adopt tenant-scoped store protocol (§4.3, §8). Migration: set `tenant_id` at create from invoker/admission context; fail shared-store reads without tenant.

### 20.4 R5-F04 — Actual authority and backend inventory

| Mechanism | Canonical owner | Production reachable | Shared durability | Worker binding | Links to ext-op record | Epoch/fence | Loss determination | Restart/partition | Admissible proof today |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ProcessLocalExternalOperationOwner` | owning worker process | Yes | No | process-local token | `owner_token` on record | revision CAS | **No** — mismatch ≠ death | new token on restart | fencing only |
| `ExternalOperationStateStore` | W4-C composition | Yes (default in-memory) | Only if operator wires shared backend | via `owner_token` | `operation_id` key | `revision` | **No** — stores state only | last committed wins | CAS outcomes |
| `IdempotencyStore` / pre-effect recovery | idempotency plane | Yes | varies | pre-effect lease | **not** RUNNING ext-op | lease epoch | **No** for RUNNING | lease expiry ≠ stop work | pre-effect only |
| `PreEffectSuspendedWorkRecoveryAuthority` | W2 recovery | Yes | ledger | worker claim | pre-effect keys | claim CAS | **No** | per STATE-X patterns | not equivalent |
| `reconcile_orphaned_running_operations` | none | Yes (no-op) | — | — | — | — | **No** | — | forbidden |
| `recover_running_operation_after_certified_owner_loss` | none @ Path B | Yes (deny) | — | — | load by `operation_id` | — | **denied** | — | raises `ExternalOperationCertifiedOwnerLossRequiredError` |
| Checkpoint / scheduler atomic claim | STATE-X / tasks | Yes | task store | worker id | **task** records | claim lease | worker loss for **tasks** | partition fail-closed | **not wired** to ext-op store |
| `CertifiedOwnerLossEvidenceIssuerPort` | **unimplemented** | **No** | — | — | — | planned fence | — | — | **none** |

**Answer:** **No sanctioned implementation can issue or verify `CertifiedOwnerLossEvidenceV1` @ HEAD.** Idempotency lease **≠** owner-loss evidence (Batch C Path B).

### 20.5 Classification and implementation eligibility

| Disposition | Value |
| --- | --- |
| **A — IMPLEMENTATION READY** | **Not met** |
| **B — ARCHITECTURE DECISION REQUIRED** | **Selected** — ratify **Option B1 (recommended):** tenant-scoped `ExternalOperationStateStore` + composition allowlisted issuer + `stage_certified_owner_loss_evidence` / `apply_certified_owner_loss` atomic CAS |
| **C — FAIL-CLOSED ONLY** | Remains default for providers without reconciliation — parallel to B for positive recovery |

**L3-R6 eligibility @ R5 audit:** **none** until operator ratification — **superseded for architecture direction only** by §21 (implementation may pre-exist; acceptance gates separate).

**Provider reconciliation limitations:** No outcome-query → remain UNKNOWN; no auto SUCCESS; physical termination ≠ business outcome (§9.1).

---

## 21. L3-R5-B operator architecture ratification (Option B1)

| Field | Value |
| --- | --- |
| **Authority** | `OPERATOR APPROVED — L3-R5-B OPTION B1 — ARCHITECTURE RATIFICATION ONLY` |
| **Date** | 2026-10-25 |
| **START_HEAD** | `141aaa3f8260929cd861e761f11739a753d9ba5d` |
| **Ratified architecture** | Tenant-scoped durable external-operation state; exactly one canonical semantic owner; store-native atomic stage/apply; composition-bound trusted issuer; durable revocation and fencing; controlled certified owner-loss; fail-closed defaults |
| **Architecture** | **RATIFIED** (B1 = ADR §20.4 recommended option) |
| **R6–R10 implementation** | **Exists @ HEAD** — **not** retrospectively accepted by this record; reconcile via [`TOOL_LIFE_X_L3_R5_B_…`](../qualification/TOOL_LIFE_X_L3_R5_B_ARCHITECTURE_RATIFICATION_AND_GATE_A_PREQUALIFICATION_EVIDENCE.md) |
| **Gate A** | **NOT PROVEN** — central question OPEN (enforceable loss of commit/effect authority) |
| **L3 / TOOL-LIFE-X parent** | **INCOMPLETE / BLOCKED** |
| **Positive production recovery** | **NOT AUTHORIZED** (RUNNING→UNKNOWN activation, worker replacement, re-dispatch) |
| **TL7** | **DEADLINE_ONLY** — no heartbeat as owner-death evidence |
| **TL8** | **BLOCKED** for positive recovery closure |
| **DIST-SCALE-X** | **Not entered** |
| **Global FRZ PASS** | **0** — ratification is not checklist PASS |

**Explicitly not authorized by this ratification:** production-positive recovery activation; automatic worker replacement; new worker-death/scheduler/lease/heartbeat/shadow authority; new lifecycle state machine; infrastructure/cloud; production recovery policy changes; formal TL7/TL8/L3/TOOL-LIFE-X parent closure.

**Historical continuity:** §20.1–§20.4 audit findings, rejected alternatives, and “prohibit L3-R6 until ratification” language remain the **pre-ratification audit baseline**. §21 records operator direction **without** erasing that R6–R10 were implemented on `development` before this publication.

**Evidence package:** [`TOOL_LIFE_X_L3_R5_B_ARCHITECTURE_RATIFICATION_AND_GATE_A_PREQUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R5_B_ARCHITECTURE_RATIFICATION_AND_GATE_A_PREQUALIFICATION_EVIDENCE.md).

---

**Disposition:** `TOOL-LIFE-X L3-R5 — READY FOR INDEPENDENT AUDIT` · **L3-R5-B — B1 RATIFIED (architecture only)** · **Gate A — FAIL/BLOCKED (prequalification)** · **STOP** on positive production recovery or Gate A self-closure
