# Diagnostic Hardening — Program Closeout

**Status:** COMPLETE (HARDEN-1 through HARDEN-5)  
**Branch baseline:** `development`  
**Canonical architecture:** [`docs/project/architecture/DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)  
**Qualification matrix:** [`DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md`](DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md)  
**Operational gap ledger:** [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) (separate from qualification closure)

---

## Final status

| Program slice | Status |
| ------------- | ------ |
| HARDEN-1 — durable Problem persistence + failure isolation | ✅ |
| HARDEN-2 — concurrency / OCC | ✅ |
| HARDEN-3 — observability export failure + exporter health | ✅ |
| HARDEN-4 — product-host E2E matrix (M1–M24) | ✅ |
| HARDEN-5 — documentation truth audit | ✅ |

```text
DIAGNOSTIC HARDENING COMPLETE
```

---

## Qualification summary

| Metric | Value |
| ------ | ----: |
| Total (M1–M24) | 24 |
| **PROVEN** | 22 |
| **PARTIALLY_PROVEN** | 0 |
| **MISSING** | 0 |
| **NOT_APPLICABLE** | 2 |
| **DEFERRED** | 0 |
| **P0 / P1 / P2 qualification gaps** | 0 |

### NOT_APPLICABLE rationale

| ID | Rationale |
| -- | --------- |
| **M21** | Central engine is deterministic. AI / `InvestigationConclusion` is a separate non-canonical layer. |
| **M22** | Central `DiagnosticOrchestrator` has no typed valid-but-unsupported scope outcome. Clean supported input with `has_findings=False` is M1 semantics, not unsupported rejection. |

---

## Architecture invariants (frozen)

1. Persisted platform facts are truth.
2. AI is not canonical truth.
3. `RuntimeEvent` is canonical execution evidence.
4. `Problem` is durable derived diagnostic state.
5. Diagnostics is vendor-neutral.
6. Problem Store uses `DocumentStore` abstraction.
7. Mongo is a provider, not diagnostics architecture.
8. Diagnostics failure does not destroy correct business result.
9. Observability is derived from platform truth.
10. Vendor telemetry is not authority.
11. Export failure does not change canonical truth.
12. Problem lifecycle: OPEN → RESOLVED → OPEN on recurrence (same `problem_id`).
13. Problem identity is tenant-scoped.
14. Diagnostic read reconstructs from canonical evidence.
15. Missing evidence yields UNAVAILABLE, not fabrication.
16. Failed Problem write is not automatically replayed.
17. Real Mongo outage supports same-process recovery (qualified).
18. Real observability provider outage supports recovery (qualified).
19. Specialized direct OTel spans (RAG/context) are derived observability only.
20. Central diagnostics has no typed valid-but-unsupported scope outcome.

---

## External proof references (selected)

| Proof | Invariant |
| ----- | --------- |
| `test_harden_4f_mongo_problem_store_failure_e2e.py` | Real Mongo FI-A outage + same-process recovery (M8) |
| `test_harden_1c_durable_problem_restart_proof.py` | Cross-process Mongo durability (M9) |
| `test_harden_2a_*` / `test_harden_2c_*` | Mongo concurrency + OCC (M10/M11) |
| `test_diag_final_external_otel_e2e.py` | OTLP collector outage/recovery; canonical truth survives (M14/M15) |

Full inventory: matrix § Existing proof inventory.

---

## Known non-guarantees

| Topic | Boundary |
| ----- | -------- |
| Failed Problem writes | No automatic replay queue |
| Unsupported diagnostic scope | No typed unsupported outcome (M22 N/A) |
| Exporter health registry | Process-local operator state — not canonical durable platform state |
| Vendor telemetry | Visualization only — not execution truth |
| AI investigation | Interpretation only — not canonical Problem authority |
| Global deployment | Cross-process proofs are qualified slices — not unlimited multi-region guarantees |
| Automatic telemetry replay | Failed export during outage is not automatically replayed (M15) |

---

## Qualification vs operational gaps

**Closed:** HARDEN qualification matrix (M1–M24) — no open P0/P1/P2.

**Open (separate backlog):** DG-001..DG-005 in [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md) — operational/platform improvement candidates discovered during real application qualification. These are **not** HARDEN blockers and do not contradict diagnostic architecture freeze.

---

## Documentation map

```text
DIAGNOSTICS.md (canonical diagnostics entry)
  → OBSERVABILITY.md (HOS, export, journal)
  → APPLICATION_HOSTING.md (host wiring)
  → DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md (engine proof index)
  → DIAGNOSTIC_HARDENING_CLOSEOUT.md (engine closeout)
  → DIAGNOSTIC_PLATFORM_ADOPTION_MATRIX.md (adoption inventory)
  → DIAGNOSTIC_MULTI_SCENARIO_E2E_MATRIX.md (multi-scenario E2E)
  → DIAGNOSTIC_PLATFORM_QUALIFICATION_CLOSEOUT.md (platform closeout)
```

**Qualification layers:**

```text
Engine qualification = HARDEN complete
Platform adoption qualification = DIAG-PLATFORM complete
```

