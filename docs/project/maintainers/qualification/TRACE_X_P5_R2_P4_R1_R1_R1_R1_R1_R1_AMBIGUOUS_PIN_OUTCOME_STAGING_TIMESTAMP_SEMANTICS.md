# TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1 — Ambiguous Pin Outcome & Staging Timestamp Semantics

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** |
| **Production delta** | **0** |
| **FINAL_COMMIT** | `fcfd59e3727163a9e67d61498a293e6002d41159` |
| **Parent** | [`TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_P2_PIN_RECOVERY_STAGING_CONTRACT_LOCK.md`](TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_P2_PIN_RECOVERY_STAGING_CONTRACT_LOCK.md) |
| **Rejected design baseline** | `2e337c4b04c112d118aa0a4817be2ea13c08ef39` — §8 treated `same provenance + different staging` as unconditional **`CONFLICT`**, breaking safe retry after lost pin acknowledgement; §4 named `requirement_boundary_recorded_at` as pin-success instant while staging is built **before** first durable accept |
| **Blocker** | `R2-P4-P2-PIN-AMBIGUOUS-COMMIT-OUTCOME-31` — **RESOLVED IN DESIGN** (this artifact) |
| **P4 / R1 chain** | **BLOCKED** on P4 implementation after audit |
| **FRZ-TRC-11** | **OPEN** |
| **P5 / CERT** | **NOT ENTERED** |

## 1. Reconciliation boundary

Parent [`TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1`](TRACE_X_P5_R2_P4_R1_R1_R1_R1_R1_P2_PIN_RECOVERY_STAGING_CONTRACT_LOCK.md) locks the canonical pin row, extended `pin()`, and `read_pin_records()`. It does **not** distinguish **semantic pin conflict** from **ambiguous first-write outcome** (durable row may exist while the caller never received success).

This artifact closes only **ambiguous-outcome retry**, **pre-write staging identity**, **race reconciliation**, **revised idempotency**, and **timestamp truth** (Option A). It does **not** reopen: one canonical `PinRecord`; staging inside the same P2 row; `read_pin_records()`; no second store; runtime builds staging; adapters only serialize; exact staging conflicts on **overwrite attempts**; Option B as-of; runtime spine ownership; deterministic `RuntimeEvent` identity.

## 2. Blocker disposition

| Blocker | Disposition |
|---|---|
| `R2-P4-P2-PIN-AMBIGUOUS-COMMIT-OUTCOME-31` | **RESOLVED IN DESIGN** — mandatory reconcile-before-pin algorithm; two conflict classes; canonical stored staging after first durable accept; `requirement_boundary_prepared_at` (Option A) |

## 3. Ambiguous outcome state (normal)

```text
pin request issued
→ CAS / put_if_absent may have committed the row
→ caller may not know (crash, timeout, transport loss before return)
```

Distributed write ambiguity is **expected**. Retry must be safe without fabricating a false staging **`CONFLICT`** solely because a fresh local candidate differs in timestamp or trace snapshot from the durable winner.

## 4. Pre-write staging identity (owner)

| Concern | Owner |
|---|---|
| Build immutable staging candidate **before** first `pin()` | Runtime (or application orchestration at pin boundary) — unchanged |
| **When** a new candidate may be created | Only when **no** durable row exists for `(tenant_id, execution_id, subject)` after sanctioned read (§6) |
| **When** candidate must **not** be recreated | After durable row exists for that obligation — reuse `requirement_recovery_staging` from the pin record |
| Durable canonical staging | First accepted row — **never** mutated on retry |

**Not** a second store. **Not** adapter-derived staging. **Not** re-derived from `datetime.now()` on every retry.

If no row exists and the in-memory candidate was lost (Case §9), a **new** candidate `S2` is valid — no earlier durable fact exists.

## 5. Timestamp semantics — Option A (selected)

Reject Option C (post-commit-only in same row without pre-commit field) for Case C: staging is prepared before first pin attempt.

Reject Option B for this lock: no separate pre-existing canonical boundary timestamp is wired for all configured-adopted obligations at pin boundary without duplicating the staging object.

**Canonical field name** on `ExecutionIntegrationConfigurationRequirementRecoveryStaging`:

```text
requirement_boundary_prepared_at   # timezone-aware UTC
```

| Phase | Meaning |
|---|---|
| Before first durable accept | Instant captured when the **immutable** staging candidate is constructed for the **first** pin attempt of this obligation |
| After first durable accept | Value in the stored row is **canonical** for Case C timestamp canon (parent R1-R1-R1-R1 §5 Case C — field renamed, semantics corrected) |

Forbidden label: “pin-success instant” for this field. Pin success is the **atomic row commit**; the timestamp documents **preparation** of the requirement boundary snapshot bundled in that row.

Legacy docs referencing `requirement_boundary_recorded_at` in staging payloads map to **`requirement_boundary_prepared_at`** in implementation (codec may accept old key on decode only if a migration lock explicitly requires — default: new writes use prepared name only).

## 6. Sanctioned read before pin (no global scan)

**Minimum read surface** (unchanged store count):

```text
read_pin_records(tenant_id, execution_id) -> tuple[PinRecord, ...]
```

**Exact obligation lookup** (orchestration helper — not a second store):

```text
find_pin_record_for_subject(records, subject) -> PinRecord | None
```

Implementations **may** add a convenience on the same `ExecutionIntegrationConfigurationPinningStore`:

```text
read_pin_record(tenant_id, execution_id, subject) -> PinRecord | None
```

only as an indexed projection of the same row key — **no** new persistence owner.

Forbidden: full-tenant scan, cross-execution scan, shadow index for recovery.

## 7. Safe retry algorithm (exact)

For each configured-adopted obligation `(tenant_id, execution_id, subject)` before issuing `pin()` after any uncertain outcome:

```text
records = read_pin_records(tenant_id, execution_id)
existing = find_pin_record_for_subject(records, subject)

if existing is not None:
    assert semantic_pin_obligation_equal(
        subject, provenance_candidate, existing.subject, existing.provenance
    ) or fail CONFLICT   # semantic class (§8)
    staging_for_pin = existing.requirement_recovery_staging
    if staging_for_pin is None and obligation_requires_staging:
        fail closed RECOVERY_STAGING_UNAVAILABLE
    pin(subject, provenance_candidate, requirement_recovery_staging=staging_for_pin)
    → idempotent success (no overwrite)
    continue spine / I/O

# no durable row
staging_for_pin = build_new_staging_once_for_this_attempt()   # new candidate allowed
pin(subject, provenance_candidate, requirement_recovery_staging=staging_for_pin)
→ first writer wins on CAS
```

**Lost acknowledgement (§8 proof):** after successful durable write, retry **must** hit `existing is not None`, reuse stored `S1`, and **must not** build `S2`.

## 8. Two conflict classes

### 8.1 Semantic pin conflict → `CONFLICT`

Different:

- `IntegrationConfigurationSubject` identity,
- `ExecutionIntegrationConfigurationProvenance` (configured/effective content, mode, fingerprints),
- `tenant_id` / `execution_id` binding,
- provenance slice mismatch on retry.

Caller **must not** adopt or overwrite.

### 8.2 Recovery reconciliation (not overwrite)

Same semantic obligation; caller held local candidate `S_local` but durable row already contains `S_stored`.

| Caller action | Result |
|---|---|
| `pin(..., staging=S_local)` while row has `S_stored` and `S_local ≠ S_stored` | **`CONFLICT`** — overwrite forbidden |
| Reconcile path: read row → `S_stored` → `pin(..., staging=S_stored)` | **Idempotent success** |
| Reconcile path: read row → use `S_stored` for spine only, skip redundant `pin` if already committed | **Allowed** orchestration optimization; store must still reject divergent staging on explicit `pin` |

Timestamp-only difference between `S_local` and `S_stored` with same semantic obligation is **recovery reconciliation**, not semantic configuration conflict.

## 9. Concurrent first writers

```text
A: read → absent → candidate S1 → pin
B: read → absent → candidate S2 → pin
```

| Outcome | Rule |
|---|---|
| CAS | Exactly **one** row wins |
| Loser `pin` with own staging | **`CONFLICT`** (staging mismatch vs stored) |
| Loser retry | `read_pin_records` → winner row → validate semantic obligation → adopt `S_winner` → `pin` with `S_winner` → idempotent success |

No last-write-wins on staging bytes. Stored first-accepted staging is canonical.

## 10. Revised idempotency (supersedes parent §8 for staging)

Let `P` = normalized provenance + subject semantic equality. Let `S` = canonical staging equality.

| Situation | Result |
|---|---|
| Row absent | First `pin` with staging → **success** |
| Row present; `P` and `S` match request | **Idempotent success** |
| Row present; `P` match; request staging `S' ≠ S` | **`CONFLICT`** (overwrite attempt) |
| Row present; `P` differs | **`CONFLICT`** |
| Uncertain outcome; row present; `P` match; orchestration re-reads `S` and calls `pin` with `S` | **Idempotent success** — **not** classified as “different staging retry conflict” |

**Superseded rule:** “same provenance + different staging = always CONFLICT” without distinguishing **discovery/reconcile** vs **overwrite**.

## 11. Mandatory proof cases

**§8 — write accepted, acknowledgement lost:**

```text
S1 → pin succeeds durably → crash before return
→ read_pin_records → recover S1
→ pin(S1) idempotent → spine commit
→ no S2, no CONFLICT
```

**§9 — write not accepted, candidate lost:**

```text
S1 lost in memory → no row
→ restart → S2 built → first pin succeeds → S2 canonical
```

**§10 — concurrent same obligation:**

```text
A S1, B S2 → one winner → loser reads winner → same P → adopt winner staging
```

**§10 — concurrent different provenance:**

```text
→ CONFLICT (semantic class)
```

## 12. Store API implication

`pin(...)` + `read_pin_records(...)` is **sufficient** when orchestration uses `find_pin_record_for_subject`. Optional `read_pin_record(..., subject)` is ergonomics only.

## 13. Adapter parity

Identical semantics on KV, DocumentStore, and in-memory reference store:

- lost-ack retry adopts stored staging;
- concurrent loser reconciles to winner staging;
- provenance mismatch stays `CONFLICT`;
- no staging field mutation on idempotent replay.

## 14. Corrected P4 tests (post-implementation)

Qualification gates: [`test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_ambiguous_pin_outcome_architecture_gates.py`](../../../../tests/qualification/trace_x/test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_ambiguous_pin_outcome_architecture_gates.py).

Contract tests (skip until extended pin API lands): [`test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_pin_ambiguous_outcome_contract.py`](../../../../tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r1_r1_r1_r1_r1_r1_pin_ambiguous_outcome_contract.py).

1. First pin accepted, simulated response loss, retry succeeds using stored staging.
2. First pin not accepted, retry may create new staging.
3. Two concurrent candidates same semantic pin → one winner, loser reconciles.
4. Two concurrent candidates different provenance → `CONFLICT`.
5. Stored staging never overwritten.
6. Restart recovers stored staging before spine append.
7. No second durable staging store (static gate).
8. Docker-backed ambiguous-write recovery (contract test; Docker marker when store extended).

## 15. Tracker status

| ID | Status after this docs-only reconciliation |
|---|---|
| `TRACE-X-P5-R2-P4` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1-R1` | **BLOCKED** |
| `TRACE-X-P5-R2-P4-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1** (implementation) |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1-R1** |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1` | **BLOCKED ON R1-R1-R1-R1-R1-R1** (ambiguous outcome supersession) |
| `TRACE-X-P5-R2-P4-R1-R1-R1-R1-R1-R1` | **READY FOR AUDIT** |
| `FRZ-TRC-11` | **OPEN** |
| `P5` | **NOT ENTERED** |
| `CERT` | **NOT ENTERED** |

## 16. Exit criteria

> If the first pin write succeeded but the caller never learned that it succeeded, how does retry continue without generating a false staging conflict?

**Answer:** Mandatory `read_pin_records` (or `read_pin_record`) **before** constructing a new staging candidate. If the row exists and semantic obligation matches, reuse stored staging and call `pin` with that staging for idempotent success — never submit a fresh local candidate with a different timestamp as an overwrite.

> What does the staging timestamp truthfully mean relative to the actual pin commit?

**Answer:** `requirement_boundary_prepared_at` is the UTC instant when the immutable staging candidate was prepared **before** the first pin attempt; it becomes the canonical Case C boundary time in the durable row once that row is accepted. It is **not** the pin-commit instant.

## 17. Independent audit notice

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
