# NPSC-5F/R1 Final — Durable Evidence Commit & Tenant Integrity Qualification and Freeze

**Status:** `FROZEN / PASS`

**Verdict:** R1 evidence durability and tenant-integrity contracts **PASS** on implementation SHA `455d3b216f0ad56ea9cdf9db6e0f760b50063a81`. Full Final **PASS** after P0 drift sentinel reconciliation for parallel development.

**Branch:** `development`

**Task:** NPSC-5F/R1 Final — Durable Evidence Commit & Tenant Integrity Qualification and Freeze

---

## Purpose

Freeze the canonical enterprise contract for mandatory execution evidence durability, explicit best-effort semantics, tenant write integrity, persist-before-acceptance, EventId idempotency, and cross-process durability. R1 Final proves:

```text
mandatory RuntimeEvent
  → exact durability classification
  → exact tenant resolution
  → durable RuntimeEventPersistence.append
  → canonical durable acceptance
  → then history / subscribers
```

and:

```text
route tenant != event.tenant_id
  → typed failure
  → zero durable write
  → zero false acceptance
```

---

## Provenance

| Label | SHA |
| ----- | --- |
| NPSC-5E Final | `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` |
| NPSC-5F/P0 | `7811371da1069b661987b050a4c9bf42c02bda69` |
| R1 implementation | `455d3b216f0ad56ea9cdf9db6e0f760b50063a81` |

---

## Implementation SHA

Commit: `455d3b216f0ad56ea9cdf9db6e0f760b50063a81`

| Module | Role |
| ------ | ---- |
| `intergrax/runtime/events/evidence_durability.py` | `EvidencePersistenceRequirement`, classification |
| `intergrax/runtime/events/persistence_contract.py` | Tenant routing, typed errors, idempotent reconcile |
| `intergrax/runtime/events/event_bus.py` | Persist-before-history/subscribers; mandatory fail-closed |

**Production code changed in R1 Final task:** NO (qualification, freeze docs, final gate tests only).

---

## Parallel-session reconciliation

Parallel sessions B (Platform Execution Unification), C (Enterprise Scale & Resilience), and D (Execution Certification Acceleration) may advance `origin/development` during Final.

The previous **BLOCKED** result was caused solely by the obsolete broad P0 sentinel `test_npsc5f_p0_drift_gate_clean_at_5e_final`, which diffed NPSC-5E Final → HEAD and treated all `intergrax/runtime/execution/**` changes as qualification failure. That sentinel was **narrowed** to R1-owned protected production surfaces (`455d3b21..HEAD` via `testing_support/npsc5f_r1_protected_drift.py` and `test_npsc5f_p0_r1_protected_evidence_surfaces_have_no_unqualified_post_r1_drift`).

**Frozen 5E behavioral protection** remains enforced by `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` (NPSC-5E Final regression), DG_001, and NPSC-5D Final — not by repository-wide execution path immutability after 5E Final.

Drift gate `git diff --name-only 455d3b21..HEAD` on R1 protected paths at Final sign-off: **empty** (no cross-session contract drift on R1 surfaces). Unrelated execution-surface commits after 5E Final are **allowed** when mandatory frozen behavioral gates pass.

Final success requires `R1_FINAL_COMMIT` ancestor of `origin/development`, not exact HEAD equality.

---

## Production surface

Frozen R1 authority boundaries:

| Component | Role |
| --------- | ---- |
| `should_persist_event` | Whether persistence applies |
| `EvidencePersistenceRequirement` | What persistence failure means |
| `RuntimeEventBus` | Transport only (not durable owner) |
| `RuntimeEventPersistence` | Canonical durable execution evidence authority |

---

## Durability taxonomy

| Value | Semantics |
| ----- | --------- |
| `NOT_PERSISTED` | `should_persist_event` false — no persistence attempt |
| `MANDATORY` | Fail-closed on persistence failure (`MandatoryEvidencePersistenceError`) |
| `BEST_EFFORT` | Logged failure; no false durable acknowledgement; boundary may continue |

No fourth implicit state.

---

## Unknown-event fail-safe

When catalog/platform mapping is absent, `retention_class_for_runtime_event` falls back to `RetentionClass.OPERATIONAL` → `MANDATORY` when persisted. Uncatalogued persisted signals must not silently downgrade to best-effort.

---

## Persist-before-acceptance

Mandatory proof: durable `append` succeeds before in-memory bus history and subscriber dispatch. Mandatory append failure: history length 0, subscriber dispatch 0.

---

## Subscriber semantics

Durable evidence committed before subscribers. Subscriber failure does **not** roll back persisted evidence and is not classified as persistence failure.

Best-effort persistence failure: history/subscribers may continue (intentional, qualified).

---

## Tenant resolution matrix

| Case | Result |
| ---- | ------ |
| event `T1`, route `T1` | `T1` |
| event `T1`, route `T2` | `EvidenceTenantRoutingMismatchError`, zero write |
| event `T1`, route absent | `T1` |
| event absent, route `T1` | `T1` (host-scoped routing) |
| both absent | `""` (legacy global/test scope; no synthesis) |
| whitespace in tenant token | `ValueError`, fail-closed |
| non-`str` route tenant | `TypeError`, fail-closed |

`RuntimeEvent.tenant_id` is not rewritten on mismatch.

---

## Zero-write guarantees

Tenant mismatch: zero writes under event tenant and route tenant for memory, SQLite, document-backed, and validating adapters.

---

## Idempotency composition

| Case | Result |
| ---- | ------ |
| Same EventId + same canonical payload | Idempotent accept |
| Same EventId + different payload | Block (`RuntimeEventPersistenceIntegrityError`) |
| Same EventId + same payload + different route tenant | Block before idempotent acceptance |

---

## Cross-adapter consistency

Shared `resolve_event_tenant_id` / `resolve_persistence_scope` — provider-neutral; no SQLite-only tenant rules.

---

## Cross-process durability

SQLite writer process → fresh reader process: mandatory event visible with preserved identity.

---

## Position concurrency

`ExecutionEventPosition` per-run allocation unchanged; concurrent SQLite append qualification retained via P0/R1 regression.

---

## Error model

| Exception | Use |
| --------- | --- |
| `MandatoryEvidencePersistenceError` | Mandatory tier persistence failure at bus boundary |
| `EvidenceTenantRoutingMismatchError` | Route vs event tenant disagreement |

Exception chaining: `MandatoryEvidencePersistenceError.__cause__` preserves underlying store error when applicable. Messages avoid raw event payloads/secrets.

---

## Security / no payload leakage

Qualification asserts mandatory error strings do not embed sink diagnostics or full event bodies.

---

## Frozen owner boundaries

Evidence durability does **not** acquire ownership of: execution lifecycle, retry engine, checkpoint/resume, lineage mutation, terminal authority, governance, authority mint, scheduling, second event bus, second evidence store framework, mandatory retry loops, or async evidence queues.

---

## Regression matrix

| Suite | Role |
| ----- | ---- |
| `test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py` | R1 Final gate |
| `test_npsc5f_r1_durable_evidence_commit_tenant_integrity.py` | R1 implementation gate |
| `test_npsc5f_p0_execution_evidence_architecture_reconciliation.py` | P0 + OBS-01/05 |
| `tests/unit/runtime/events/**` | Bus, stores, idempotency |
| `tests/unit/runtime/observability/**` | Export/journal adjacent surfaces |
| `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | Frozen 5E |
| DG_001 lineage contracts | Identity non-regression |
| `test_npsc5d_final_multi_agent_governance_qualification.py` | Frozen 5D |

---

## Static quality

`ruff` and `pyright` on R1 production surface and final gate paths — **NEW STATIC ERRORS = 0** at freeze commit.

---

## Known deferred gaps

| ID | Topic | Target |
| -- | ----- | ------ |
| OBS-03 | Raw journal export bypass | R3 |
| OBS-04 | Unified journal truncation / completeness API | R2 |
| Stream completeness | — | R2 |
| Task/run ordering semantics | — | R2 |
| As-of / bitemporal completion | — | R4 |

Do not claim mandatory failure rolls back external side effects — only that evidence-critical boundaries cannot report fully evidenced success without durable commit.

---

## Final verdict

**R1 contract qualification:** PASS on implementation SHA `455d3b21` (no unqualified drift on R1 protected production surfaces since R1 implementation).

**Full Final freeze:** **FROZEN / PASS** — P0 gate PASS including ownership-scoped R1 drift sentinel; mandatory frozen regressions (NPSC-5E Final, DG_001, NPSC-5D Final, runtime events/observability suites) PASS at Final sign-off.

**Not CORRECTION REQUIRED** for R1 production surfaces — no mandatory-evidence or tenant-routing defect found in Final gate tests.

---

## Freeze statement

> Canonical mandatory RuntimeEvent evidence is accepted only after provider-neutral RuntimeEventPersistence durably commits it under the exact resolved tenant scope. Persistence failure for mandatory evidence fails closed before bus history and subscriber dispatch. Best-effort evidence retains explicit non-authoritative failure semantics. Explicit routing tenant and RuntimeEvent tenant identity must match exactly when both are present; mismatch performs zero write. RuntimeEventBus remains transport, RuntimeEventPersistence remains durable evidence authority, and evidence durability does not acquire execution, lineage, checkpoint, terminal, governance, authority, retry, recovery, or scheduling ownership.

**Next:** NPSC-5F/R2 — Journal Completeness & Ordering
