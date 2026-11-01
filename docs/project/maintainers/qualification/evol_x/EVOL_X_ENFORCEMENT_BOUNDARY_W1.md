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

**S7 conclusion:** Current `ActivationService.rollback` **does not** enforce durable-state compatibility. Enforcement today is **qualification assessor + harness**; **production precondition port** is a **sanctioned optional delta** (OD-EVOL-05b Option 2) — **STOP production change until operator approves**.

**Parallel authority prohibition:** No new `approve`/`authorize`/`execute` fields on assessment DTOs; AD/ActivationService remain sole mutators.
