# SAC-DIST-SCALE-X-v0 — Stage Acceptance Contract (Proposal)

| Field | Value |
| --- | --- |
| **Contract ID** | `SAC-DIST-SCALE-X-v0` |
| **Status** | **PROPOSED / NOT FROZEN** |
| **Baseline SHA (P0 docs)** | `35344557d3ce67a631f82df96c86b79b92b4db93` |
| **Freeze gate** | `OPERATOR APPROVED — DIST-SCALE-X SAC FREEZE` + independent review |
| **Parent stage** | `DIST-SCALE-X` — **not CLOSED** |
| **Prerequisite** | `TOOL-LIFE-X` = **CLOSED — INDEPENDENTLY ACCEPTED** @ `1548640708c850b3fbfca3f2db0e5ecfee4274dd` |

**Explicit:** Unresolved **mandatory** or **release-blocking** defect prevents **unconditional** parent `CLOSED`. Conditional closure only with documented OPEN debt accepted by operator.

---

## 1. Purpose

Bind whole-platform scale/resilience certification per roadmap §3.0.6 without new semantic authority, without substituting HARNESS-W4 or TOOL-LIFE-X, and without PROD-Q live vendor as a DIST exit requirement.

---

## 2. Canonical exit criteria (roadmap §3.0.6 items 1–16)

| # | Criterion | Acceptance (v0) | Evidence type | Operator / lock |
| --- | --- | --- | --- | --- |
| 1 | Representative workloads defined | WL-1..7 specs @ P0 SHA + any P1+ amendments tracked | L2 | — |
| 2 | Production adapters exercised | All **M** backends in vendor matrix executed @ AUDIT_BASELINE_SHA | L3/L4 | Broker **M** choice |
| 3 | Multi-worker scale tested | ≥2 worker processes, ≥2 workers on **M** queue | L4 | Worker count **TO BE LOCKED** |
| 4 | Overload/backpressure tested | Dim **C** scenarios; bounded behavior | L3 | Queue depth **TO BE LOCKED** |
| 5 | Retry storms tested | WL-6 + dim **E** | L3 | Amplification **TO BE LOCKED** |
| 6 | Hot partition/key tested | WL-2, WL-3 | L3 | Hot share **TO BE LOCKED** |
| 7 | Crash/restart tested | Dim **G** matrix | L4 | — |
| 8 | Partial failures tested | Dim **F** matrix | L3/L4 | — |
| 9 | Tenant stress isolation | WL-2 + dim **K** | L3 | Degradation **TO BE LOCKED** |
| 10 | Resource exhaustion | Dim **L** | L3 | Pool limits **TO BE LOCKED** |
| 11 | Soak completed | WL-7 | L4+ dedicated | Duration **TO BE LOCKED** |
| 12 | Chaos matrix completed | Dim **N** seeded | L4+ | Seed catalog |
| 13 | No correctness invariant violated | Zero automatic FAIL (§5) | L3+ | — |
| 14 | Capacity envelope documented | tested/expected/unsupported | L2 P7 | — |
| 15 | Failure/recovery envelope documented | same | L2 P7 | — |
| 16 | Unresolved scale blockers | 0 **or** explicit release-blocking OPEN list | L2 | Operator accepts debt |

---

## 3. Automatic correctness FAIL (unconditional)

Regardless of throughput, **FAIL** if any §3.0.6 invariant: duplicate side effect; lost state; lost evidence; tenant leakage; wrong recovery; stale authority; corruption; governance bypass under load (see [DIST_SCALE_X_THRESHOLD_PROPOSAL.md](DIST_SCALE_X_THRESHOLD_PROPOSAL.md) §3.1).

---

## 4. Certification dimensions A–N

| Dim | Mandatory evidence doc (pattern) | WL / batch | Clear now | TO BE LOCKED |
| --- | --- | --- | --- | --- |
| A Throughput | `DIST_SCALE_X_A_EVIDENCE.md` | WL-1,2,4,5; P2+ | Invariants | Throughput, latency percentiles |
| B Horizontal scale | `DIST_SCALE_X_B_EVIDENCE.md` | WL-1; P2 | No dup FX on scale-out | Scaling efficiency |
| C Backpressure | `DIST_SCALE_X_C_EVIDENCE.md` | WL-1,6; P2–3 | Bounded queues, fail-closed | Max depth, rejection rate |
| D Hot partition | `DIST_SCALE_X_D_EVIDENCE.md` | WL-2,3; P3 | Hotspot behavior documented | Tail latency caps |
| E Retry storm | `DIST_SCALE_X_E_EVIDENCE.md` | WL-4,6; P3 | No dup FX; drain | Amplification factor |
| F Partial failure | `DIST_SCALE_X_F_EVIDENCE.md` | WL-4,5; P4 | Typed fail-open/closed | Outage duration tested |
| G Crash consistency | `DIST_SCALE_X_G_EVIDENCE.md` | WL-5; P4 | No lost evidence / dup action | — |
| H Durable restart | `DIST_SCALE_X_H_EVIDENCE.md` | WL-5; P4 | Semantic state match | Recovery time |
| I Partition | `DIST_SCALE_X_I_EVIDENCE.md` | P5 | **N/A WITH EVIDENCE** if not claimed | — |
| J Ordering | `DIST_SCALE_X_J_EVIDENCE.md` | WL-3; P5 | Invariants > ordering | — |
| K Tenant stress | `DIST_SCALE_X_K_EVIDENCE.md` | WL-2; P3 | Zero leakage | Neighbor degradation |
| L Resource exhaustion | `DIST_SCALE_X_L_EVIDENCE.md` | WL-4,6; P5 | REJECT/degrade | Pool limits |
| M Soak | `DIST_SCALE_X_M_EVIDENCE.md` | WL-7; P6 | No monotonic leak | Duration, drift |
| N Chaos | `DIST_SCALE_X_N_EVIDENCE.md` | WL-7; P6 | Reproducible seeds | — |

---

## 5. Workloads WL-1–WL-7

Authoritative parameters: [DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md).  
**WL-4:** HTTP loopback physical **M** in **P3** — **excluded:** PROD-Q live cloud (**E**).

---

## 6. Physical backend requirements

| Class | Requirement for parent CLOSED |
| --- | --- |
| **M** | Must appear in executed L3/L4 evidence @ CLOSURE_SHA |
| **R** | May satisfy secondary paths only with behavioral proof @ CLOSURE_SHA; else **EVIDENCE GAP** |
| **F** | Not sufficient alone for items 2–3 |
| **E** | Out of DIST parent scope (PROD-Q) |
| **U** | Must not be claimed |

Matrix: [DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md](DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md).

---

## 7. Test and evidence levels (ECAP-aligned)

| Level | DIST-SCALE-X use |
| --- | --- |
| L1 | Static gates: no shadow dispatcher; P0 matrix completeness (P1+) |
| L2 | This SAC, specs, matrix, envelopes, lock record |
| L3 | Executed pytest qual @ SHA; logs `.tmp/proof/dist-scale-x/<run-id>/` |
| L4+ | Multi-process, compose-backed, kill -9, soak, chaos |
| L7 | Restart/recovery proofs (WL-5, H) |
| L9 | **Independent exact-SHA audit** — mandatory before parent **CLOSED** |

**Cursor evidence:** L1–L4 candidate only until L9.

---

## 8. Historical evidence reuse

| Artifact | SHA | Disposition |
| --- | --- | --- |
| HARNESS-W4 | `f38e455ccc391dad9a6a64d7867a33cfc5c9b21d` | **REUSE** harness scope — **NOT SUFFICIENT** DIST multi-platform |
| TOOL-LIFE-X | `1548640708c850b3fbfca3f2db0e5ecfee4274dd` | **REUSE** lifecycle — **RE-VALIDATE** WL-4/6 load |
| STATE-X | `bd54941d933069b8bfb2819bb819c1cbdbe71576` | **REUSE** ownership — **RE-VALIDATE** WL-5 load/restart |
| TENANT-X | `5ccb72fccd4e3820ebdc9c18008afe6f7d59f558` | **REUSE** — **RE-VALIDATE** WL-2/K |
| BG-01 | @ START_HEAD catalog | **REUSE** — **RE-VALIDATE** on broker **M** |
| W2-C / distributed RL | harness tests | **REUSE** partial — **EVIDENCE GAP** WL-6 full stack |

---

## 9. Authority boundaries

- No new semantic owners; no shadow scheduler/dispatcher/retry/state machine.  
- Parallel qual only with P0-documented isolation.  
- **G-A07:** qual topology ≠ `OD-R4-G-A07-PROD-ACTIVATE`.  
- **FRZ global:** DIST closure does **not** promote **FRZ-REL-01..11** to PASS.

---

## 10. Formal parent closure sequence

```text
P7 → DIST_SCALE_X_PARENT_QUALIFICATION_EVIDENCE.md
   → disposition: DIST-SCALE-X — READY FOR INDEPENDENT AUDIT @ AUDIT_BASELINE_SHA
L9 → independent exact-SHA audit (Gate B post-qual)
OPERATOR APPROVED — DIST-SCALE-X FORMAL PARENT CLOSURE
Roadmap §5 → CLOSED — INDEPENDENTLY ACCEPTED @ CLOSURE_SHA
```

**Forbidden:** Cursor marks parent **CLOSED** in same step as P7 evidence publication.

---

## 11. Operator gates (checklist)

- [ ] SAC-DIST-SCALE-X-v0 **FREEZE**  
- [ ] `DIST_SCALE_X_THRESHOLD_LOCK_RECORD` before P2 perf  
- [ ] `OPERATOR APPROVED — DIST-SCALE-X QUAL INFRA RUN` (per batch)  
- [ ] Broker **M** (Rabbit vs Kafka)  
- [ ] Representative host (LKW default)  
- [ ] WL-7 soak host + duration lock  
- [ ] WL-4 scope = loopback only  
- [ ] PROD-Q live explicitly out of scope  
- [ ] P7 → L9 → operator closure (no skip)

---

## 12. P0 SAC status

| Item | Status |
| --- | --- |
| SAC complete v0 | **YES** |
| SAC frozen | **NO** |
| Thresholds locked | **NO** |
| P0 independently CLOSED | **NO** — disposition **READY FOR INDEPENDENT AUDIT** |

---

## 13. Related P0 artifacts

- [DIST_SCALE_X_P0_CLOSED_WORLD_INVENTORY.md](DIST_SCALE_X_P0_CLOSED_WORLD_INVENTORY.md)  
- [DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md](DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md)  
- [DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md)  
- [DIST_SCALE_X_THRESHOLD_PROPOSAL.md](DIST_SCALE_X_THRESHOLD_PROPOSAL.md)
