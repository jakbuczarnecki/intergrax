# DIST-SCALE-X — Parent qualification evidence (P7 / L9 dossier)

| Field | Value |
| --- | --- |
| **SAC** | `SAC-DIST-SCALE-X-v1` |
| **START_HEAD** | `ddb18890a216b0c51abc9e463efc31f73dcfe359` |
| **EXECUTION_SHA (P5/P6 physical)** | `9ac74ca3849013e729dbc8c893f64427ef588f94` |
| **AUDIT_BASELINE_SHA** | `6f4193fa9a06204b32fab76a9774511ef0a6ad72` |
| **Disposition** | **READY FOR INDEPENDENT AUDIT** — WL-7 **14400 s** physical **deferred** ([deferral](DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md)) |
| **G-06 handoff** | [DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md](DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md) |
| **Parent CLOSED** | **FORBIDDEN** until G-06 + G-07 |

## L9 exit matrix (16 rows) — T-L9-01

| ID | Requirement | Evidence | Phys | EXECUTION_SHA | Disp |
| --- | --- | --- | --- | --- | --- |
| 1 | WL-1..7 specs | [WL specs](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md); WL-7 harness | PG+Rabbit | `9ac74ca…` | **PARTIAL** (4 h pending) |
| 2 | Production adapters MB-1..5 | P2–P5 reuse + P6 | LKW+PG+Rabbit | Multiple cited | **ALREADY PROVEN** + P5/P6 slices |
| 3 | Multi-worker | P2 WL-1 | N≥2 | `ee5a8371…` | **ALREADY PROVEN** |
| 4 | Overload | P2/P3 | — | P2/P3 SHAs | **ALREADY PROVEN** |
| 5 | Retry storm | WL-6 | Rabbit M | `e6020efc3` | **ALREADY PROVEN** |
| 6 | Hot key | WL-3 | 4 proc | `8656fb7d8` | **ALREADY PROVEN** |
| 7 | Crash/restart | P4 G/H | PG | `f0c49fd3…` | **ALREADY PROVEN** |
| 8 | Partial failure | P4 F + P5 I | PG+HTTP | `f0c49fd3…` + `9ac74ca…` | **PASS** |
| 9 | Tenant stress | WL-2 K | multi-tenant | `8656fb7d8` | **ALREADY PROVEN** |
| 10 | Resource exhaustion L | P5 L + P3 reuse | admission | `9ac74ca…` | **PASS** |
| 11 | Soak M | WL-7 | mixed load | `9ac74ca…` harness | **BLOCKED** (14400 s) |
| 12 | Chaos N | P6 matrix | seeded | `9ac74ca…` | **PASS** |
| 13 | Zero invariant violations | manifests | — | per run | **PASS** on accepted runs |
| 14 | Capacity envelope | [P7 capacity](DIST_SCALE_X_CAPACITY_ENVELOPE.md) | — | `9ac74ca…` | **PASS** (ENVELOPE_ONLY) |
| 15 | Failure envelope | [P7 failure](DIST_SCALE_X_FAILURE_ENVELOPE.md) | — | same | **PASS** (ENVELOPE_ONLY) |
| 16 | Blockers = 0 | this matrix | — | — | **BLOCKED** (row 11) |

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

**AGENT STOP** — dossier complete for **G-06 documentary audit** (WL-7 4 h physical excluded per deferral). No G-07; no parent CLOSED; no FRZ PASS; no EBH-5.
