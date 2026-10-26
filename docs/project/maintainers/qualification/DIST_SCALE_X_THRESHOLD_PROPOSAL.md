# DIST-SCALE-X — Threshold Proposal (P0)

| Field | Value |
| --- | --- |
| **Task** | `DIST-SCALE-X-P0` |
| **Baseline SHA** | `35344557d3ce67a631f82df96c86b79b92b4db93` |
| **Status** | **PROPOSAL — NOT LOCKED** |
| **Frozen artifact (future)** | `DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md` — required **before** P2 performance/saturation/soak tests |
| **Product SLOs** | **None canonical in repo @ P0** → metrics below marked **TO BE LOCKED** |

---

## 1. Locking process (mandatory)

```text
DIST_SCALE_X_THRESHOLD_PROPOSAL.md (this file, P0)
  → independent review + SAC-DIST-SCALE-X-v0 operator freeze
  → engineering fills proposed values OR envelope-only mode
  → OPERATOR APPROVED — DIST-SCALE-X THRESHOLD LOCK
  → DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md @ SHA
  → P2+ performance-class tests allowed
```

**Forbidden:** inventing final certification numbers in P0; marking thresholds **APPROVED** without `THRESHOLD_LOCK_RECORD`.

---

## 2. Threshold source hierarchy

| Priority | Source | P0 finding |
| --- | --- | --- |
| 1 | Canonical product/production SLO | **Not found** in governing docs @ START_HEAD |
| 2 | Operator lock record | **Not yet created** |
| 3 | Envelope-only (relative) | Allowed only with explicit operator choice per metric in LOCK_RECORD |

---

## 3. Measurable proposals — by workload

### 3.1 Qualitative (pass/fail without numeric SLO) — **active @ P2 even before lock**

These are **roadmap §3.0.6 automatic FAIL** conditions — no lock required:

- Duplicate external side effect  
- Lost durable state or mandatory evidence  
- Tenant data/authority/cache contamination  
- Wrong recovery or stale authority becoming effective  
- Governance/ToolRuntime bypass under load  
- Unbounded memory/thread/task growth (no backpressure)  
- Tool hang beyond TOOL-LIFE-X deadline semantics  
- Unbounded retry (`while True` / unbounded amplification without drain)

### 3.2 Quantitative — **TO BE LOCKED** (proposal structure only)

| Metric ID | Description | WL | Evidence source for proposed value | Proposed methodology | Lock status |
| --- | --- | --- | --- | --- | --- |
| M-WL1-THR | Min useful throughput @ N workers | WL-1 | **None** — envelope-only candidate | Compare 1 vs N worker median tasks/s; require non-decreasing useful throughput | **TO BE LOCKED** |
| M-WL1-ERR | Max error rate (non-transient) | WL-1 | **None** | Count terminal FAILED not classified retryable / total | **TO BE LOCKED** |
| M-WL1-LAT-P95 | p95 execution latency | WL-1 | **None** | Histogram from qual harness metrics | **TO BE LOCKED** |
| M-WL1-QD | Max queue depth under overload | WL-1,C | W4 harness classification (not numeric limit) | Observe bounded queue; reject/defer semantics | **TO BE LOCKED** |
| M-WL2-HOT | Hot tenant traffic share | WL-2 | Operator policy | % requests from hot tenant | **TO BE LOCKED** |
| M-WL2-DEG | Max p95 degradation ratio cold/hot | WL-2 | **None** | Ratio cold p95 / baseline p95 | **TO BE LOCKED** |
| M-WL3-P99 | p99 under contention | WL-3 | **None** | Single-key family stress | **TO BE LOCKED** |
| M-WL4-RPS | Provider call rate sweep max | WL-4 | TOOL-LIFE-X qual topology (not throughput SLO) | Stepwise RPS until saturation or lock ceiling | **TO BE LOCKED** |
| M-WL4-AMP | Max end-to-end retry amplification factor | WL-4,6 | W2-C historical (bounded, not numeric) | (total attempts)/(initial requests) under injected failures | **TO BE LOCKED** |
| M-WL4-CONC | Max concurrent provider calls | WL-4 | W4-R1 admission chain | Observe REJECT/WAIT vs hang | **TO BE LOCKED** |
| M-WL5-WR | Sustained state write rate | WL-5 | **None** | PG append/checkpoint load | **TO BE LOCKED** |
| M-WL5-REC | Max recovery time after SIGKILL | WL-5,H | STATE-X L7 class (qualitative) | Time to consistent restart | **TO BE LOCKED** |
| M-WL6-FAIL% | Injected transient failure rate | WL-6 | **None** | Layered fault injection | **TO BE LOCKED** |
| M-WL6-DRAIN | Max time to drain backlog post-restore | WL-6 | **None** | Queue depth → baseline | **TO BE LOCKED** |
| M-WL7-DUR | Soak duration | WL-7 | Operator infra | Wall clock dedicated env | **TO BE LOCKED** |
| M-WL7-MEM | Max memory slope / leak threshold | WL-7 | **None** | Process metrics sampling | **TO BE LOCKED** |
| M-WL7-DRIFT | Max p95 latency drift % | WL-7 | **None** | Compare first vs last hour | **TO BE LOCKED** |
| M-SCALE-EFF | Worker scaling efficiency 1→N | B | **None** | Useful throughput(N)/N vs throughput(1) | **TO BE LOCKED** or envelope-only |

---

## 4. Envelope-only alternative (operator decision)

If no product SLO:

- Document **tested point** (N workers, RPS achieved, duration) in `DIST_SCALE_X_CAPACITY_ENVELOPE.md` (P7).  
- PASS requires **shape** properties (e.g. no invariant violation; amplification bounded by observed curve) — **not** marketing limits.  
- Must be explicitly selected in `THRESHOLD_LOCK_RECORD` per metric ID.

---

## 5. Operator decisions required (threshold-related)

1. Approve **envelope-only** vs numeric caps per metric ID.  
2. Set **M-WL7-DUR** and dedicated host policy.  
3. Set **M-WL2-HOT** share.  
4. Select broker **M** (feeds concurrency/saturation tests).  
5. Approve **SAC freeze** before lock record publication.

---

## 6. P0 checks

| Check | Result |
| --- | --- |
| All WL have parameter specs | **PASS** — [DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md) |
| No final numeric thresholds approved | **PASS** |
| Lock process documented | **PASS** |
| Automatic FAIL invariants listed | **PASS** §3.1 |
