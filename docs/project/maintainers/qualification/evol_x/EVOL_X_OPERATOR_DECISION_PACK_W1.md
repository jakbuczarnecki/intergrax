# EVOL-X W1 — Operator decision pack

**Status:** Prepared for operator — **not self-approved**.

## OD-EVOL-01 — Placement

| | |
|--|--|
| **Facts** | `pyproject.toml` ships only `intergrax/`; COMPAT/QUAL live under `tests/qualification/`; existing gate forbids `testing_support` import from production, not `tests` |
| **Recommendation [R]** | **Option C:** `testing_support/change_safety/` + `tests/qualification/evol_x/adapters/` |
| **Alternatives** | A/B under `intergrax/` (rejected in ADR — shipping/import risk); D scripts-only |
| **Production impact** | **None** for Option C |
| **Tests** | W2-02 import boundary gates |
| **Acceptance** | Signed ADR in capabilities pair |
| **Blockers** | None for W2 start once approved |

## OD-EVOL-02 — Workflow evolution policy

| | |
|--|--|
| **Facts** | Pinning + checkpoint verify exist; no in-flight migration executor; N-04 graph_version gap |
| **Recommendation [R]** | Policy **(i):** VERSIONED_DEPLOY + pin; at-rest MIGRATE; in-flight migration **EXPLICITLY_UNSUPPORTED** |
| **Production impact** | None unless N-04 resolved with Class B contract change |
| **Blockers** | N-04 decision before claiming WM rows involving `GraphSpecMigration` |

## OD-EVOL-05 — Deploy qualification split

| | |
|--|--|
| **Facts** | N-03 rollback gap; N-06 adapters; PROD-Q live OPEN |
| **Recommendation [R]** | (a) full assessor scope; (b) **Option 2** precondition port behind composition flag; (c) NA-01..04 with evidence |
| **Production impact** | **Option 2 only** — `intergrax/agent_distribution/` additive port |
| **Tests** | Existing AD tests + S7 + new port unit tests |
| **Blockers** | Operator must approve Option 2 before W4-04 |

## OD-EVOL-06 — Revision state facts source

| | |
|--|--|
| **Recommendation [R]** | Sidecar manifest `schema_snapshot@revision` (no `RuntimeRevision` field change in W1) |
| **Alternative** | Add fields to `RuntimeRevision` (Class B) |
| **Blockers** | Choose before W4-03 fact injection |

## OD-EVOL-09 — Exclusions

| | |
|--|--|
| **Recommendation [R]** | Accept X-01,02,04,05; complete X-03,06,07 proofs in W2 |
| **Blockers** | CW-04 fails if proofs missing |

## OD-EVOL-10 — CI / collect / workflow

| | |
|--|--|
| **Facts** | N-18: 32 collect errors on local `-m "gate and not no_ci"` without full extras; qual_x collect OK (85) |
| **Recommendation [R]** | Authorize: (1) W2 edit `unit-tests.yml` profile `evol_x_p3_l3`; (2) define CONSERVATIVE_FULL = pytest set from **CI manifest** (dev-ci sync), not ad-hoc local marker collect; (3) sync workflow to `main` before GHA dispatch |
| **Blockers** | W2-09, W6-04 blocked until signed |

## OD-EVOL-12 — FRZ-EVOL-08/10 promotion

| | |
|--|--|
| **Recommendation [R]** | Option A scoped PASS at W7 — see `EVOL_X_FRZ_08_10_CLOSURE_POLICY_W1.md` |
| **Blockers** | W7-05 checklist edit |

## OD-EVOLX-01 — Stage entry

| | |
|--|--|
| **Recommendation** | Approve EVOL-X entry @ START_HEAD |
| **Blockers** | W1 docs committed; independent architecture audit (AF-8) |
