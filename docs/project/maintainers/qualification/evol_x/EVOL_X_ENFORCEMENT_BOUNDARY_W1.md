# EVOL-X W1 — Enforcement boundary (Investigation A)

For each safety class, EVOL-X **must not** introduce parallel release/governance/execution authority.

| Decision / transition | Advisory assessment (Q tooling) | Qualification gate (EVOL tests) | Existing production enforcement | Missing production enforcement | S3 / S7 note |
|----------------------|--------------------------------|--------------------------------|--------------------------------|-------------------------------|--------------|
| Unknown change impact | `ImpactAssessmentResult` CONSERVATIVE | M-04, selection REFUSED/CONSERVATIVE_FULL | Push gate / diff_scoped (heuristic only) | Semantic impact graph | Qualification only until W2+ |
| In-flight wrong revision | Workflow matrix WM-* | S3 experiments | **YES:** pin inherit, checkpoint fingerprint verify, missing pin errors — `execution_pinning.py` | In-flight **definition migration** executor | **Production rejects** stale/mismatch pins; migration in-flight = unsupported (no executor) |
| Unsafe rollback (durable state) | `deployment_safety` assessor | S7 DM-04 arm | **PARTIAL:** AD rollback checks artifact/serving/SUPERSEDED/readiness — **not** durable schema | **YES:** durable forward-only rollback | **S7:** unsafe rollback can succeed at AD layer today → FRZ-EVOL-07 needs assessor + optional **OD-EVOL-05b** port |
| Contract breaking change | `contract_diff` | S2/S5 | COMPAT-X policy gates (static) | Change-time consumer closure | Qualification + reuse COMPAT policy |
| Mandatory regression bypass | Selection oracle | S6, M-05 | QUAL-X P5 manifest integrity | Impact-driven selection | W3+ |
| Core edit without OD | Extension classifier | S8, M-10 | Governance Class A/B/C (human) | Machine frozen-core registry | W2-03 JSON projection |

**S3 conclusion:** Current implementation **does** reject unsafe pin/checkpoint/stale-definition cases via existing profile-resolution enforcement. **Does not** adopt new workflow definitions in-flight — absence of executor = **EXPLICITLY_UNSUPPORTED** (not silent pass).

**S7 conclusion (FRZ-EVOL-07 — F-03):**

| Layer | What it proves | What it does **not** prove |
|-------|----------------|----------------------------|
| **S7 controlled assessor PASS** (qualification) | Harness + `deployment_safety` returns correct **read-only** verdicts (e.g. DM-04 REFUSE) on scripted durable-state scenarios | **Production** `ActivationService.rollback` will refuse unsafe rollback |
| **Existing production AD rollback** | Artifact revalidation, serving pointer, SUPERSEDED prior, instance readiness | Durable schema forward-only compatibility, in-flight pin safety vs demoted revision |
| **OD-EVOL-05b Option 2 port (`default off`)** | *If approved and implemented in W4* — optional **mandatory** precondition wired into **existing** AD rollback path | **Not** a second deployment authority; **default off = no new mandatory production enforcement** until composition explicitly enables port |

**FRZ-EVOL-07 scoped acceptance (checklist wording):** At EVOL-X closure, PASS may cite **qualification evidence**: S7 arms + assessor + typed `DeploySafetyAssessment` + explicit unsafe-refusal cases — **without** claiming live PROD-Q or mandatory runtime rollback hardening. Checklist: rollback not safe from “binary restart alone”; durable state + contract compatibility required — satisfied at **qualification** layer first.

**Mandatory runtime enforcement (out of scope W1 / this batch):** Requires separate operator authorization **OD-EVOL-05b Option 2**, domain plan row, implementation in W4-04, production composition flag **on**, and regression proof — still **no** parallel executor or governance path.

**Parallel authority prohibition:** Unchanged — assessment DTOs remain non-mutating; AD remains sole rollback mutator.
