# ADR-ENTERPRISE-TOOL-LIFE-X-CERTIFIED-OWNER-LOSS-AND-STALLED-POLICY

| Field | Value |
| --- | --- |
| **Status** | **READY FOR INDEPENDENT AUDIT** — architecture/documentation only; **not RATIFIED**; **production delta = 0** |
| **Task** | `TOOL-LIFE-X L3-R5` — Enterprise Architecture ADR (TL7/TL8) |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **BLOCKED** / **not CLOSED** |
| **Date** | 2026-10-24 |
| **Branch** | `development` |
| **START_HEAD** | `db7cb510c0231be88d38c89123f2b60c3d26564e` |
| **Operator authority** | TL7/TL8 enterprise-safe direction **pre-approved** (ten principles §3); this ADR **does not** reopen them |
| **Canonical roadmap** | [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.5 |
| **Prior evidence (reuse)** | P0 [`TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md`](../qualification/TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md) · Batch C [`TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md`](../qualification/TOOL_LIFE_X_BATCH_C_STALLED_ABANDONED_RECOVERY_EVIDENCE.md) · Batch E [`TOOL_LIFE_X_BATCH_E_ADVERSARIAL_REAL_INFRASTRUCTURE_EVIDENCE.md`](../qualification/TOOL_LIFE_X_BATCH_E_ADVERSARIAL_REAL_INFRASTRUCTURE_EVIDENCE.md) · L3-R1 [`TOOL_LIFE_X_L3_R1_CONSOLIDATED_QUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R1_CONSOLIDATED_QUALIFICATION_EVIDENCE.md) · L3-R2 [`TOOL_LIFE_X_L3_R2_MCP_HOST_LIFECYCLE_QUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R2_MCP_HOST_LIFECYCLE_QUALIFICATION_EVIDENCE.md) |
| **Related ADR** | W4-C external operations · [`ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md`](ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md) · STATE-X recovery plane (separate concern) |
| **Does not** | Close `TOOL-LIFE-X` · start `DIST-SCALE-X` · implement recovery code · promote global `FRZ-*` PASS · authorize new distributed backend or blind takeover |

**Publication evidence:** [`TOOL_LIFE_X_L3_R5_ENTERPRISE_CERTIFIED_OWNER_LOSS_ADR_QUALIFICATION_EVIDENCE.md`](../qualification/TOOL_LIFE_X_L3_R5_ENTERPRISE_CERTIFIED_OWNER_LOSS_ADR_QUALIFICATION_EVIDENCE.md) (publication SHA = containing commit on `development`).

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

**Conclusion:** Sanctioned infrastructure **can** host certified owner-loss **as an extension of the W4-C external-operation ownership domain** without a second lifecycle authority. **No** new semantic owner is required beyond the typed **certified-owner-loss evidence plane** bound to existing `ExternalOperationStateStore` transitions.

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

**Semantic owner of “may transition RUNNING→UNKNOWN for recovery”:** **`CertifiedOwnerLossEvidenceIssuerPort`** — the **only** authority that may mint evidence; verifiers are pure functions + store CAS preconditions.

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
    evidence_digest: str  # canonical serialization digest for cross-process verify

class CertifiedOwnerLossEvidenceIssuerPort(Protocol):
    def issue_owner_loss_evidence(
        self,
        *,
        epoch: ExternalOperationOwnershipEpochV1,
        loss_reason: CertifiedOwnerLossReason,
        liveness_proof_ref: str | None,
    ) -> CertifiedOwnerLossEvidenceV1: ...

class CertifiedOwnerLossEvidenceVerifierPort(Protocol):
    def verify(
        self,
        evidence: CertifiedOwnerLossEvidenceV1,
        *,
        expected_tenant_id: str,
        expected_operation_id: str,
    ) -> None: ...  # raises CertifiedOwnerLossEvidenceRejectedError
```

**API extension (same symbol, new parameter — breaking at type level only in implementation batch):**

`recover_running_operation_after_certified_owner_loss(store, *, operation_id, expected_revision, evidence: CertifiedOwnerLossEvidenceV1, verifier: CertifiedOwnerLossEvidenceVerifierPort) -> ExternalOperationState`

**Preconditions (all mandatory):**

- Loaded record `physical_state == RUNNING`.
- `evidence.epoch.operation_id == operation_id`.
- `evidence.epoch.store_revision == expected_revision` (or CAS retry loop with fresh evidence).
- `evidence.epoch.owner_token == record.owner_token`.
- `evidence.epoch.tenant_id` matches tool invocation tenant binding.
- Verifier accepts digest, issuer, fence monotonicity, and liveness proof handle.
- **Post:** single CAS to `UNKNOWN` physical state; intent moves to `TERMINATING` (existing W4-C pattern); owner token cleared per existing finalize rules.

**Idempotency:** Second compatible evidence with same fence → return current record if already `UNKNOWN` with matching terminal class; conflicting terminal → `ExternalOperationFencedCommitRejectedError`.

### 4.3 Persistence and atomicity

| Element | Rule |
| --- | --- |
| Durable schema | Extend **optional** columns/fields on external-op store record: `ownership_epoch_fence: int | None`, `last_certified_loss_evidence_id: str | None` — **or** side table keyed by `(tenant_id, operation_id)` owned by same store implementation |
| Tenant identity | `tenant_id` on evidence epoch; store keys remain `operation_id` but operation_id MUST be derived from tenant-scoped tool external-op identity (existing `ExternalOperationIdentity`) |
| Transaction boundary | `verify` → `compare_and_set` in one composition call; no intermediate observable RUNNING mutation |
| Linearization | Store `revision` CAS is the linearization point |
| Epoch semantics | `monotonic_fence` increases on each **accepted** owner-loss certification for that operation |
| Concurrent recovery | Loser CAS → `StaleExternalOperationStateError`; re-load and re-issue evidence if still RUNNING |
| Crash consistency | If CAS succeeded before crash, restart sees UNKNOWN; reconciliation proceeds — no auto-replay |
| Compatibility | `InMemoryExternalOperationStateStore` + future shared store implementations implement same protocol |

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

See §4.3. Fencing: `owner_token` + `monotonic_fence` + store `revision` CAS. Superseded workers cannot publish terminals (`ExternalOperationFencedCommitRejectedError` — already proven Batch C).

---

## 8. Tenant isolation model

- `CertifiedOwnerLossEvidenceV1.epoch.tenant_id` MUST match invocation tenant.
- Verifier MUST reject cross-tenant evidence even if `operation_id` collides across tenants (defense in depth with tenant-scoped operation_id generation).
- Evidence digests MUST NOT embed secrets; `liveness_proof_ref` is opaque.

---

## 9. Recovery and reconciliation semantics

See §4.4. Reconciliation owner: existing provider ports + tool invoker policy — **downstream** of UNKNOWN admission.

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

---

## 12. Implementation impact graph

```text
L3-R6-A  contracts/certified_owner_loss_evidence.py + errors
    → L3-R6-B  verify + recover_running_operation_after_certified_owner_loss (evidence param)
        → L3-R6-C  store schema optional fields + InMemory impl
            → L3-R6-D  composition issuer binding (fail-closed if missing)
                → L3-R7  adversarial pytest matrix §11
                    → L3-R8  real-infra issuer liveness proofs (optional operator Docker)
```

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
| **L3-R6** | Contracts + verifier + CAS transition + composition fail-closed/default deny | IMPLEMENTATION REQUIRED |
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

## 17. TL1–TL15 dependency classification (@ L3-R5 ADR)

| Criterion | Owner | Contract | Valid evidence SHA | Classification | Missing requirement | Tests | FRZ | Dependency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TL1 | RuntimeToolInvoker | ToolContract timeout | Batch A `8f6b9a23…` | ALREADY PROVEN | — | Batch A L2 | FRZ-REL-01 scoped | — |
| TL2 | ExecutionDeadlineAuthority | deadline hierarchy | Batch A/E | ALREADY PROVEN | — | consolidated | FRZ-REL-02 scoped | — |
| TL3 | CancellationCoordinator | cancel ports | W4-A + Batch C | ALREADY PROVEN (scoped) | PG/MCP physical cancel | W4-C | FRZ-REL-04 scoped | TL9 |
| TL4 | RuntimeToolInvoker | ToolRetryPolicy | Batch B | ALREADY PROVEN | — | Batch B | FRZ-REL-07 scoped | TL5 |
| TL5 | Invoker + idempotency | ToolEffectCertainty | B + L3-R1 HTTP | ALREADY PROVEN (scoped) | runbooks | L3-R1 | FRZ-REL-03 scoped | TL6 |
| TL6 | Invoker | SideEffectRetrySafety | Batch B | ALREADY PROVEN | retry re-auth matrix | Batch B | FRZ-GOV scoped | TL5 |
| TL7 | tool_operation_recovery | STALLED observational | Batch C @ Path B | VALIDATION REQUIRED | heartbeat policy optional | `is_stalled_*` | FRZ-REL-01 scoped | ADR §4.5 |
| TL8 | external_operation_ownership | CertifiedOwnerLossEvidenceV1 | Batch C deny @ HEAD | IMPLEMENTATION REQUIRED | L3-R6..R7 | deny + future positive | FRZ-REL-05, FRZ-REC scoped | **this ADR** |
| TL9 | ExternalOperationStateStore | fenced terminal | Batch C | ALREADY PROVEN (scoped) | late result all providers | Batch C | FRZ-REL-05 scoped | TL8 impl |
| TL10 | SQL tools | row/byte limits | D + E partial | VALIDATION REQUIRED | Docker PG cancel | D/E | FRZ-REL-10 scoped | **PG Docker BLOCKED** |
| TL11 | MCP host | ToolContract path | D + L3-R2 scoped | VALIDATION REQUIRED | full 15-scenario host bridge | L3-R2 | FRZ-REL-06 scoped | **MCP physical BLOCKED** |
| TL12 | HTTP provider | AllowlistHttpClient | D + E + L3-R1 | VALIDATION REQUIRED | lost-ack (improved) | L3-R1 | FRZ-REL-06 scoped | — |
| TL13 | Qual cross-provider | real infra | E partial | VALIDATION REQUIRED | Docker PG + MCP | Batch E | FRZ-REG scoped | TL10, TL11 |
| TL14 | Arch gates | single invoker | A–D + L3-R1 | ALREADY PROVEN | — | U5 gates | FRZ-BND-03 scoped | — |
| TL15 | TOOL-LIFE-X parent | EC1–15 rollup | P0 + batches | BLOCKED | TL8 impl, TL10–13 infra | parent L3 | — | all |

---

## 18. Remaining parent blockers

| Blocker | Class | Notes |
| --- | --- | --- |
| TL8 positive recovery | IMPLEMENTATION REQUIRED | L3-R6+ per §14 |
| PostgreSQL / Docker physical qual | **BLOCKED** (operator C10) | TL10/TL13 — separate from ADR |
| MCP hung-server physical qual | **BLOCKED** | TL11 — separate from ADR |
| TL7 heartbeat policy | OPERATOR DECISION REQUIRED | Optional; default DEADLINE_ONLY sufficient for ADR |
| `TOOL-LIFE-X` parent CLOSED | BLOCKED | Requires TL1–14 qual rollup |

---

## 19. Post-step enterprise discovery and roadmap completeness

- **DIST-SCALE-X:** not started — **explicitly excluded** from this ADR.
- **New semantic authority conflict:** **none** — certified loss is an evidence plane on existing W4-C store, not a competing lifecycle.
- **Ratification:** Operator pre-approved direction; ADR status remains **READY FOR INDEPENDENT AUDIT** until independent exact-SHA review. Implementation batches do not imply RATIFIED until governance record says so.

---

**Disposition:** `TOOL-LIFE-X L3-R5 — ADR READY FOR INDEPENDENT AUDIT`
