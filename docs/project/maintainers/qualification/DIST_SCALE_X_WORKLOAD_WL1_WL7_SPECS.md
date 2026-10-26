# DIST-SCALE-X — Workload Specifications WL-1–WL-7 (P0)

| Field | Value |
| --- | --- |
| **Task** | `DIST-SCALE-X-P0` |
| **Baseline SHA** | `35344557d3ce67a631f82df96c86b79b92b4db93` |
| **Roadmap** | §3.0.6 workload models — mandatory fields per workload |
| **Numeric thresholds** | **TO BE LOCKED** — see [DIST_SCALE_X_THRESHOLD_PROPOSAL.md](DIST_SCALE_X_THRESHOLD_PROPOSAL.md) |
| **Execution** | **Not run in P0** — specs only |

**Notation:** `TBD-LOCK` = requires `DIST_SCALE_X_THRESHOLD_LOCK_RECORD` + operator approval before performance-class tests (P2+).

---

## WL-1 — Many independent executions

| Field | Specification |
| --- | --- |
| **Intent** | High parallel independent tasks across tenants |
| **Request rate** | Sweep: baseline → `TBD-LOCK` max RPS (stepwise) |
| **Concurrency** | `concurrent_executions`: 1 → `TBD-LOCK` N (workers × in-flight per worker) |
| **Read/write ratio** | Representative: 30% read-heavy status checks / 70% write-admit execution (adjustable in lock) |
| **Payload size** | `payload_bytes`: 1 KiB default; spot check `TBD-LOCK` large class |
| **Tenant count** | `tenants_min`: **2**; scale to `TBD-LOCK` |
| **Key distribution** | Uniform across tenants |
| **Fan-out** | Within platform caps (256 items / 64 concurrency per `bounded_multi_agent_fanout.py`) |
| **External dependency latency** | Minimal (local execution) unless mixed into WL-7 |
| **Failure %** | **0%** baseline; fault cases in WL-6 |
| **Duration** | Smoke: `TBD-LOCK` minutes; sustained slice `TBD-LOCK` |
| **Backends** | DocumentStore queue **M**, PG **M**, workers **M** |
| **Dimensions** | A, B, C |
| **Batch** | P2 |
| **Qual owner (future)** | `tests/qualification/dist_scale_x/wl_01_*` |

---

## WL-2 — Hot tenant

| Field | Specification |
| --- | --- |
| **Intent** | Noisy neighbor — one tenant dominates traffic |
| **Request rate** | Hot tenant: `TBD-LOCK` share of WL-1 peak RPS |
| **Concurrency** | Same pool as WL-1; hot tenant `TBD-LOCK` fraction of in-flight |
| **Read/write ratio** | Same as WL-1 unless lock specifies read-heavy hot path |
| **Payload size** | WL-1 default |
| **Tenant count** | **≥3** (1 hot + ≥2 cold) |
| **Key distribution** | Hot tenant concentrated task names / execution families |
| **Fan-out** | Moderate within caps |
| **Dependency latency** | WL-1 baseline |
| **Failure %** | 0% baseline |
| **Duration** | `TBD-LOCK` |
| **Backends** | MB-1..MB-3 |
| **Dimensions** | A, D, K |
| **Batch** | P2–P3 |
| **Qual owner** | `tests/qualification/dist_scale_x/wl_02_*` |

---

## WL-3 — Hot key / execution family

| Field | Specification |
| --- | --- |
| **Intent** | Contention on idempotency / execution family / CAS hotspots |
| **Request rate** | `TBD-LOCK` focused on single key family |
| **Concurrency** | `TBD-LOCK` workers hammering same idempotency key / execution id family |
| **Read/write ratio** | Write-heavy (state updates) |
| **Payload size** | Minimal |
| **Tenant count** | **1** tenant (isolate key contention) + optional cross-tenant negative probe |
| **Key distribution** | **Skewed** — single family |
| **Fan-out** | Low |
| **Dependency latency** | N/A |
| **Failure %** | 0% except ordering tests (dim J) |
| **Duration** | `TBD-LOCK` |
| **Backends** | PG **M**, DocumentStore queue **M** |
| **Dimensions** | D, J |
| **Batch** | P3 |
| **Qual owner** | `tests/qualification/dist_scale_x/wl_03_*` |

---

## WL-4 — Provider-heavy (HTTP loopback physical)

| Field | Specification |
| --- | --- |
| **Intent** | External/provider I/O dominates — **DIST scope = loopback HTTP Class B**, not live vendor |
| **Request rate** | `TBD-LOCK` provider calls/sec sweep |
| **Concurrency** | `TBD-LOCK` parallel tool invocations |
| **Read/write ratio** | N/A (tool calls) |
| **Payload size** | Default + `TBD-LOCK` large response pressure case |
| **Tenant count** | **≥2** |
| **Key distribution** | Multiple tool identities; one hot tool optional |
| **Fan-out** | Tool fan-out within invoker bounds |
| **Dependency latency** | Injected: 0 ms baseline; storm: **> tool deadline** (`TBD-LOCK` % requests) |
| **Failure %** | `TBD-LOCK` 5xx/429; `TBD-LOCK` unknown-outcome (reset/lost ack) |
| **Duration** | `TBD-LOCK` |
| **Backends** | HTTP loopback server in qual fixtures **M**; reuse TOOL-LIFE-X strict-host patterns @ `154864070…` |
| **Dimensions** | A, E, F, L |
| **Batch** | **P3 (mandatory)** |
| **PROD-Q** | Cloud HTTP = **E** — **excluded** from DIST parent CLOSED |
| **Qual owner** | `tests/qualification/dist_scale_x/wl_04_http_physical_*` |
| **Scenarios (physical)** | Timeout storm; retry amplification (Retry-After); UNCERTAIN outcomes; connection pool saturation |

---

## WL-5 — Persistence-heavy

| Field | Specification |
| --- | --- |
| **Intent** | State/evidence/checkpoint writes dominate |
| **Request rate** | `TBD-LOCK` state writes/sec |
| **Concurrency** | `TBD-LOCK` parallel writers |
| **Read/write ratio** | **90% write** / 10% read (checkpoint/event append) |
| **Payload size** | Representative checkpoint/event blob `TBD-LOCK` |
| **Tenant count** | **≥2** |
| **Key distribution** | Round-robin tenants |
| **Fan-out** | Low |
| **Dependency latency** | PG commit latency natural |
| **Failure %** | 0% baseline; crash tests in P4 |
| **Duration** | `TBD-LOCK` |
| **Backends** | PostgreSQL **M**; SQLite **R** smoke only |
| **Families** | SX-F01, F03, F07, F09 priority (inventory) |
| **Dimensions** | A, G, H |
| **Batch** | P4 |
| **Qual owner** | `tests/qualification/dist_scale_x/wl_05_*` |

---

## WL-6 — Retry storm

| Field | Specification |
| --- | --- |
| **Intent** | High transient failure rate — measure amplification |
| **Request rate** | `TBD-LOCK` (may be lower than WL-1) |
| **Concurrency** | `TBD-LOCK` |
| **Read/write ratio** | Mixed |
| **Payload size** | WL-1 default |
| **Tenant count** | **≥2** |
| **Key distribution** | Uniform |
| **Fan-out** | Moderate |
| **Dependency latency** | Injected latency on HTTP and/or broker |
| **Failure %** | `TBD-LOCK` transient (layered: transport + provider) |
| **Duration** | `TBD-LOCK` until queue drained post-restore |
| **Backends** | MB-1..MB-4; optional Redis **R** if operator enables |
| **Dimensions** | E, C |
| **Batch** | P3 |
| **Qual owner** | `tests/qualification/dist_scale_x/wl_06_*` |

---

## WL-7 — Long-running soak

| Field | Specification |
| --- | --- |
| **Intent** | Mixed representative production load |
| **Request rate** | Mix WL-1..6 at **TBD-LOCK** fractions |
| **Concurrency** | Steady `TBD-LOCK` |
| **Read/write ratio** | Mixed per sub-workload weights in lock |
| **Payload size** | Mixed |
| **Tenant count** | `TBD-LOCK` |
| **Key distribution** | Mixed uniform + mild hot spot |
| **Fan-out** | Occasional within caps |
| **Dependency latency** | Mixed |
| **Failure %** | Low background `TBD-LOCK` |
| **Duration** | **`TBD-LOCK` soak** — dedicated environment; not default PR CI |
| **Backends** | Full mandatory matrix MB-1..MB-5 |
| **Dimensions** | M, N (+ governance negatives) |
| **Batch** | P6 |
| **Qual owner** | `tests/qualification/dist_scale_x/wl_07_*` |

---

## Cross-WL traceability

| Document | Link |
| --- | --- |
| A–N mapping | [DIST_SCALE_X_SAC_PROPOSAL.md](DIST_SCALE_X_SAC_PROPOSAL.md) §4 |
| Backend classes | [DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md](DIST_SCALE_X_VENDOR_BACKEND_SCALE_MATRIX.md) |
| Thresholds | [DIST_SCALE_X_THRESHOLD_PROPOSAL.md](DIST_SCALE_X_THRESHOLD_PROPOSAL.md) |
