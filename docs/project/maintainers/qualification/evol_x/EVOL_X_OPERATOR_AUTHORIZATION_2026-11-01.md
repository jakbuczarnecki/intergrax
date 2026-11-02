# EVOL-X — Operator authorization package (2026-11-01)

**Recorded in repo:** agent session implementing W0 register alignment + Batch B–F execution (Batch G excluded).

## Plan anchors

| Field | Value |
|-------|-------|
| Plan authority | [`EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md`](../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md) |
| PLAN_PUBLICATION_SHA | `b4fca4bc413a650b2e5a8dd8826669a9025864c5` |
| Implementation baseline | `8848857741f4b04e5b476dca3c47f7d0a60bf672` |

## Scoped approval

- **In scope:** integrated Batch **B–F** (W2–W6) implementation, tests, evidence preparation, commits on `development` with enterprise gates.
- **Out of scope:** Batch **G** / W7 FRZ promotion, B6 formal closure without independent audit, canonical roadmap § status edits, Docker, new production paths, FRZ-* checklist manufacture, test bypass, self-certification.
- **Thresholds:** canonical qualification thresholds — **no lowering** for PASS.

## Operator decisions (signed 2026-11-01)

| ID | Disposition |
|----|-------------|
| **OD-EVOLX-01** | APPROVED — EVOL-X stage entry for Batch B–F execution |
| **OD-EVOL-01** | APPROVED — Option C placement (`testing_support/change_safety/` + `tests/qualification/evol_x/adapters/`) |
| **OD-EVOL-02** | APPROVED — workflow policy (i): VERSIONED_DEPLOY + pin; at-rest MIGRATE; in-flight **EXPLICITLY_UNSUPPORTED** |
| **OD-EVOL-05a** | APPROVED — full EVOL deploy assessor scope in qualification harness |
| **OD-EVOL-05b** | **Option 1** — qualification-only FRZ-07 scope; **no** W4-04 production rollback port |
| **OD-EVOL-05c** | APPROVED — NA-01..04 out-of-scope register with evidence |
| **OD-EVOL-06** | APPROVED — revision facts via sidecar `schema_snapshot@revision` manifest |
| **OD-EVOL-09** | APPROVED — exclusions X-01,02,04,05; W2 proofs required for X-03,06,07 |
| **OD-EVOL-10** | APPROVED — CI/collect authority per register (manifest-backed CONSERVATIVE_FULL; `evol_x_p3_l3` profile) |
| **N4-A (N-04 OD)** | APPROVED — graph semver in migration docs only until optional N4-B |

## Deferred (remain PENDING — later W0 gates)

| ID | Blocks |
|----|--------|
| **OD-EVOL-03** | W6-01 metric thresholds (W0-07) |
| **OD-EVOL-07** | P-H historical commit promotion (W0-07) |
| **OD-EVOL-08** | M-09 SLA ratification (W0-07) |
| **OD-EVOL-11** | W7-01 independent auditor (W0-08) |
| **OD-EVOL-12** | W7-03 FRZ-08/10 checklist wording (Batch G — not authorized this package) |
| **OD-EVOLX-08** | W7-04 parent closure (Batch G — not authorized this package) |

## Addendum 2026-11-01 (operator message: "zatwierdzam wszystko - wykonaj caly plan")

| ID | Disposition |
|----|-------------|
| **OD-EVOL-03** | APPROVED - M thresholds canonical, no lowering |
| **OD-EVOL-07** | APPROVED - P-H promotion |
| **OD-EVOL-08** | APPROVED - M-09 SLA |
| **OD-EVOL-11** | APPROVED as appointment of an operator-named independent auditor; **name not supplied** - operator must record it (agent MUST NOT self-appoint) |
| **OD-EVOL-05b** | unchanged: **Option 1**, no W4-04 |
| **OD-EVOL-12**, **OD-EVOLX-08** | remain PENDING (W7-03/W7-04; require independent audit record) |

**Batch G constraint:** no B6 PASS, no FRZ checklist PASS edits, no roadmap closure without an independent audit record.

