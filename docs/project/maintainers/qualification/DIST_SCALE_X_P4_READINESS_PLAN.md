# DIST-SCALE-X P4 — Readiness plan (PLAN ONLY)

| Field | Value |
| --- | --- |
| **Mode** | Read-only planning — **no implementation**, **no physical execution** |
| **SAC** | `SAC-DIST-SCALE-X-v1` |
| **Whole-stage plan** | `dist-scale-x_whole_stage_e7172e11.plan.md` (Gate A/B accepted; cited in P0 inventory) |
| **Git baseline** | Enter P4 implementation from **`8191334e97130ec83ef75b603a30732237990b24`** onward |
| **P3 closure SHA** | `8191334e97130ec83ef75b603a30732237990b24` ([DIST_SCALE_X_P3_FORMAL_CLOSURE_RECONCILIATION.md](DIST_SCALE_X_P3_FORMAL_CLOSURE_RECONCILIATION.md)) |

---

## 1. P4 exit-criteria matrix

| Obligation | SAC dim | WL | Evidence doc (to publish) | Pass semantics |
| --- | --- | --- | --- | --- |
| Typed partial dependency failure | **F** | WL-4 (reuse negatives), WL-5 | `DIST_SCALE_X_F_EVIDENCE.md` | Fail-open/closed matches architecture; no silent unsafe fallback |
| Crash consistency around durable steps | **G** | WL-5 | `DIST_SCALE_X_G_EVIDENCE.md` | No duplicate semantic action; no lost required evidence |
| Durable restart after hard termination | **H** | WL-5 | `DIST_SCALE_X_H_EVIDENCE.md` | Fresh process recovers semantic state from **PostgreSQL M** |
| Persistence-heavy load profile | — | **WL-5** | Provenance + manifests | Write-dominated checkpoint/event path @ real PG; invariants empty |

**Frozen boundaries:** P5 owns **I**, **J**, **L** full qualification; P6 owns **M**, **N**, WL-7; P7/parent closure later. **No new numeric SLOs** unless locked from product (threshold record remains `NOT_LOCKED` unless operator amends).

---

## 2. Reusable historical evidence

| Source | SHA | Valid for P4 | Gap |
| --- | --- | --- | --- |
| **STATE-X** | `bd54941d933069b8bfb2819bb819c1cbdbe71576` | Ownership, CAS, recovery flow matrix, family inventory | **RE-VALIDATE** under WL-5 write load + kill/restart — not sufficient alone |
| **TOOL-LIFE-X** | `1548640708c850b3fbfca3f2db0e5ecfee4274dd` | Unknown-outcome, bounded retry, abandonment | Does not prove PG crash windows |
| **P3 WL-4** | `e6020efc3` | HTTP 503/slow/lost-ack **partial failure** slice for provider path | Does not cover store/event outage classes for **F** |
| **P3 WL-3** | `8656fb7d8` | Idempotency under multi-process | SQLite/PG qual schema only — not checkpoint families |
| **HARNESS-W4** | `f38e455ccc391dad9a6a64d7867a33cfc5c9b21d` | CAS / recovery ambiguity patterns | Harness scope — **NOT SUFFICIENT** for DIST WL-5 |
| **PG CAS proofs** | `intergrax/collaborative_work/postgresql_*_cas_proof.py` | Cross-process CAS on CW repos | Representative **R** for relational CAS — WL-5 must target **runtime persistence families** (SX-F01, F03, F07, F09 per WL spec) |

**Invalidation rule:** Any change to touched persistence protocol, commit ordering, or qual harness after EXECUTION_SHA forces scoped replay (SAC §3.1).

---

## 3. Production-effective call graphs (semantic owners)

| Concern | Owner (do not duplicate) | P4 touch surfaces |
| --- | --- | --- |
| Decision / execution checkpoints | `DecisionCheckpointPersistence` + `decision_checkpoint_persistence.py`; STATE-X families | Qual harness writes via **sanctioned store adapters**, not parallel checkpoint tables |
| Runtime events / evidence | `RuntimeEventPersistence` / `ValidatingRuntimeEventPersistence` | Append under load; crash mid-batch |
| Attempt / execution lifecycle | `AttemptLifecycleService` / `AttemptLifecycleStore` | Terminal state after restart |
| Task queue durability | `DocumentStoreTaskQueue` + worker claim (P3 path) | Optional **R**; WL-5 primary = PG state families |
| Relational **M** | `open_postgresql_relational_store`, `open_postgresql_collaborative_work_repositories` | Container PG @ qual DSN (Windows: Docker publish IP workaround documented P3) |
| Default dev bundle | `SQLiteRuntimePersistenceBundle` (`sqlite_composition.py`) | **R** smoke only per vendor matrix |
| Recovery activation | G-A07 default-off | Qual topology only — **not** `OD-R4-G-A07-PROD-ACTIVATE` |
| Retry / duplicate FX | TOOL-LIFE-X + P3 WL-6 | P4 must not add retry controller; replay safe after crash |

**Composition:** Qual uses existing `integration_profile_wiring` / host-injected bundles — no new composition root in `applications/`.

---

## 4. Execution DAG (operator + agent)

```text
OD-P4-ENVELOPE (lock WL-5 duration/concurrency TBD-LOCK)
  → OD-INFRA-P4 (PostgreSQL M healthy; optional Rabbit not blocking WL-5)
  → AGENT-P4-IMPL (wl_05 harness + F/G/H evidence stubs + selection batch)
  → L1/L2 green @ IMPLEMENTATION_SHA
  → OD-AGENT-P4 physical (WL-5 + F/G/H scenarios)
  → P4 provenance doc + dimension F/G/H evidence
  → Independent plan/execution audit
  → P5 handoff (I/J/L) — NOT STARTED from P4
```

---

## 5. Physical scenarios (bounded)

| Scenario | Intent | Injection |
| --- | --- | --- |
| WL-5 baseline | Sustained checkpoint/event writes ≥2 tenants | PG **M**, 90/10 write/read per WL spec |
| F-store-down | Storage unavailable during write phase | PG pause / reject (container control — no volume destroy) |
| F-event-persist | Event persistence unavailable | Fail-closed vs architecture doc |
| G-mid-commit | Kill worker after decision, before ack | SIGKILL / `taskkill` on qual worker |
| G-post-effect | Crash after side-effect class (simulated) | Harness-only external effect ledger |
| H-restart | `load → write → kill → fresh process → recover` | New PID; same DSN |
| Negative | Duplicate replay after recovery | Assert ≤1 semantic effect (reuse P3 pattern) |
| Negative | Cross-tenant recovery | TENANT-X fail-closed |

---

## 6. Negative acceptance assertions

- Duplicate semantic side effect after crash/replay → **FAIL**
- Lost required durable evidence → **FAIL**
- Recovery reads another tenant's checkpoint → **FAIL**
- New retry loop or shadow dispatcher in qual → **FORBIDDEN** (L1 authority guard)
- Claiming PG multi-node partition tolerance without mechanism → **FORBIDDEN**
- Promoting **FRZ-REL-*** to PASS from P4 evidence → **FORBIDDEN**

---

## 7. L1–L4 qualification strategy

| Layer | P4 content |
| --- | --- |
| **L1** | Extend catalog: WL-5 module, F/G/H evidence paths, no forbidden imports (`authority_surface_guard`) |
| **L2** | F/G/H evidence markdown contracts; manifest fields for crash point + recovery PID |
| **L3** | `wl_05_*` pytest @ EXECUTION_SHA; manifests under `.tmp/proof/dist-scale-x/` |
| **L4** | SIGKILL / multi-process writers / PG-backed persistence (not in-memory alone) |

**L7** class proofs align with STATE-X restart/recovery flows — reference matrix, do not re-litigate ownership.

---

## 8. Infrastructure and operator register

| Decision | Status |
| --- | --- |
| PostgreSQL **M** `postgres:16.6` @ 5434 | Reuse `intergrax-postgresql`; DSN via env |
| Windows Docker loopback | Document `INTERGRAX_POSTGRESQL_DSN` @ Docker publish IP if `127.0.0.1` wslrelay breaks handshake |
| RabbitMQ | **Not** prerequisite for WL-5 PG-only path |
| Redis / Kafka | **Not authorized** |
| Chaos/soak long duration | **P6** — out of P4 unless envelope explicitly extends |

---

## 9. READ / EDIT / TEST / FORBIDDEN

| | |
| --- | --- |
| **READ** | SAC, P0 inventory, STATE-X inventory, WL-5 spec, P3 harness (`p3_multiprocess_worker`, `_p2_support`), `sqlite_composition`, PG opens, existing WL-4 fault tests |
| **EDIT** | `tests/qualification/dist_scale_x/wl_05_*`, `_p4_support` (if needed), `selection.py` P4 batch, F/G/H evidence docs, P4 provenance, envelope record **amend** (operator) |
| **TEST** | Single bounded `uv run pytest` per verification step; P4 selection batch only after OD-AGENT-P4 |
| **FORBIDDEN** | `intergrax/` production semantic changes without HEP; new recovery owner; volume wipe; P3 physical rerun without invalidation; P5/P6 work; parent CLOSED |

---

## 10. FRZ / ECAP impact

- **ECAP:** P4 delivers L3/L4 candidate evidence only; L9 remains for parent closure (ECAP-P4).
- **FRZ-REL-01..11:** May receive **local scoped contribution** notes in evidence — **global PASS = 0**.
- **FRZ-REG / FRZ-PRD / FRZ-SEC:** Unchanged by P4 plan alone.

---

## 11. Evidence and provenance requirements

Each physical run: `run_id`, `EXECUTION_SHA`, `catalog_integrity_sha256`, F02 manifest, environment_definition (PG DSN redacted pattern), crash_point enum, worker topology, correctness invariant set (empty), metrics (write rate, recovery latency — **document**, not guard unless locked).

Publication: `DIST_SCALE_X_P4_PHYSICAL_PROVENANCE.md` + update F/G/H evidence + optional P4 formal reconciliation after independent audit.

---

## 12. Proposed AGENT-P4 batch-first package

1. **Preflight:** HEAD = P3 closure SHA; clean tree; L1/L2.
2. **Implement:** `wl_05_persistence_heavy.py` (+ worker module), `DIST_SCALE_X_P4_SELECTION_BATCH`, evidence stubs F/G/H, `test_dist_scale_x_p4_qualification_batch.py` (subprocess wrapper pattern from P3).
3. **Envelope:** OD-P4-ENVELOPE — lock WL-5 warm-up/measurement/duration (fill `TBD-LOCK` from operator).
4. **Physical:** WL-5 baseline → G/H kill scenarios → F dependency faults (ordered to minimize PG thrash).
5. **Docs:** Provenance + dimension records; roadmap row **P4 scoped accepted**.
6. **STOP** if PG handshake fails — report diagnostics (no restart loop).

---

## 13. STOP conditions and P4→P5 handoff

**STOP:** undefined WL-5 parameters without envelope; PG unavailable; L1 violation; duplicate semantic owner introduced; attempt to close parent DIST-SCALE-X.

**Handoff to P5:** P4 reconciliation published; F/G/H evidence complete; WL-5 authoritative EXECUTION_SHA recorded; P5 scope **I/J/L** explicitly **NOT ENTERED** until operator opens envelope.

---

## 14. Predictive failure modes (pre-agent)

| Risk | Mitigation |
| --- | --- |
| Qual uses SQLite only | FAIL vendor matrix — mandate PG **M** for G/H |
| Kill harness races Windows port exhaustion | Throttle workers; reuse P3 PG connection discipline |
| Confuse CW PG CAS proof with runtime bundle | Wire WL-5 to STATE-X family list, not CW-only |
| Re-run P3 modules | Invalidates lineage — forbidden unless code invalidation |
| Recovery default-off blocks H | Use qual-sanctioned persistence wiring per STATE-X / integration profile — not prod G-A07 flag |
| Manifest says `localhost:5434` but host uses Docker IP | Record effective DSN in manifest `environment_definition` |
