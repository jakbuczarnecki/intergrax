# DIST-SCALE-X — Parent qualification evidence (P7 / L9 dossier)

| Field | Value |
| --- | --- |
| **SAC** | `SAC-DIST-SCALE-X-v1` |
| **START_HEAD** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **EXECUTION_SHA (P5/P6 physical)** | `9ac74ca3849013e729dbc8c893f64427ef588f94` |
| **AUDIT_BASELINE_SHA** | `6f4193fa9a06204b32fab76a9774511ef0a6ad72` |
| **Disposition** | **`CLOSED — OPERATOR SCOPED ACCEPTANCE`** @ [DIST_SCALE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md](DIST_SCALE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md) (2026-10-28) |
| **G-06** | Operator scoped acceptance — **no external Protocol 03 audit in repo** |
| **G-07** | `OPERATOR APPROVED — DIST-SCALE-X FORMAL PARENT CLOSURE (SCOPED — WL-7 4H DEFERRED)` |
| **WL-7 EXECUTION_SHA (1 h)** | `9a062322826de8444d430eed1556d7463a2cd2bf` |
| **G-06 handoff** | [DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md](DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md) |

## L9 exit matrix (16 rows) — T-L9-01

| ID | Requirement | Evidence | Phys | EXECUTION_SHA | Disp |
| --- | --- | --- | --- | --- | --- |
| 1 | WL-1..7 specs | [WL specs](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md); WL-7 **1 h** soak | PG+Rabbit | `9a062322826de8444d430eed1556d7463a2cd2bf` | **PASS (SCOPED)** † 4 h deferred |
| 2 | Production adapters MB-1..5 | P2–P5 reuse + P6 | LKW+PG+Rabbit | Multiple cited | **ALREADY PROVEN** + P5/P6 slices |
| 3 | Multi-worker | P2 WL-1 | N≥2 | `ee5a8371…` | **ALREADY PROVEN** |
| 4 | Overload | P2/P3 | — | P2/P3 SHAs | **ALREADY PROVEN** |
| 5 | Retry storm | WL-6 | Rabbit M | `e6020efc3` | **ALREADY PROVEN** |
| 6 | Hot key | WL-3 | 4 proc | `8656fb7d8` | **ALREADY PROVEN** |
| 7 | Crash/restart | P4 G/H | PG | `f0c49fd3…` | **ALREADY PROVEN** |
| 8 | Partial failure | P4 F + P5 I | PG+HTTP | `f0c49fd3…` + `9ac74ca…` | **PASS** |
| 9 | Tenant stress | WL-2 K | multi-tenant | `8656fb7d8` | **ALREADY PROVEN** |
| 10 | Resource exhaustion L | P5 L + P3 reuse | admission | `9ac74ca…` | **PASS** |
| 11 | Soak M | WL-7 | mixed load **3600 s** | `9a062322826de8444d430eed1556d7463a2cd2bf` `wl07-soak-c8e86ebef60e` | **PASS (SCOPED)** — **1 h PROVEN**; **14400 s deferred non-blocking** |
| 12 | Chaos N | P6 matrix | seeded | `9ac74ca…` | **PASS** |
| 13 | Zero invariant violations | manifests | — | per run | **PASS** on accepted runs |
| 14 | Capacity envelope | [P7 capacity](DIST_SCALE_X_CAPACITY_ENVELOPE.md) | — | `9ac74ca…` | **PASS** (ENVELOPE_ONLY) |
| 15 | Failure envelope | [P7 failure](DIST_SCALE_X_FAILURE_ENVELOPE.md) | — | same | **PASS** (ENVELOPE_ONLY) |
| 16 | Blockers = 0 | this matrix | — | — | **PASS (SCOPED)** — no open implementation blockers |

## Proof bundle index (T-L9-02)

| Class | Location |
| --- | --- |
| Session logs | `.tmp/session/dist-scale-x-p5-l9/` |
| F02 manifests | `.tmp/proof/dist-scale-x/<run_id>/` |
| P5 provenance | [DIST_SCALE_X_P5_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P5_PHYSICAL_PROVENANCE.md) |
| P6 provenance | [DIST_SCALE_X_P6_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P6_PHYSICAL_PROVENANCE.md) |
| P7 reconciliation | [DIST_SCALE_X_P7_READINESS_RECONCILIATION.md](DIST_SCALE_X_P7_READINESS_RECONCILIATION.md) |
| L9 G-06 handoff | [DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md](DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md) |
| WL-7 deferral | [DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md](DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md) |

## T-L9-03

† **Operator envelope amendment + 4 h deferral:** full **14400 s** WL-7 soak **NOT PROVEN** — [DIST_SCALE_X_WL7_4H_DEFERRED_NON_BLOCKING_RECORD.md](DIST_SCALE_X_WL7_4H_DEFERRED_NON_BLOCKING_RECORD.md); substitute **3600 s** @ [DIST_SCALE_X_WL7_ENVELOPE_AMENDMENT_1H_RECORD.md](DIST_SCALE_X_WL7_ENVELOPE_AMENDMENT_1H_RECORD.md). Manifest: `.tmp/proof/dist-scale-x/wl07-soak-c8e86ebef60e/`.

**Parent @ publication:** **`CLOSED — OPERATOR SCOPED ACCEPTANCE`** — no global FRZ PASS; no EBH-5 entered.
