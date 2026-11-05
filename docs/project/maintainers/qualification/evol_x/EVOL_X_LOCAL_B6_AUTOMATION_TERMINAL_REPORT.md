# EVOL-X — LOCAL B6 AUTOMATION — TERMINAL REPORT

**Campaign ID:** `EVOL-X-LOCAL-B6-AUTOMATION`  
**Baseline HEAD:** `b669139d1acfb4aca37a3502775f3ac2b2ea5525`  
**Disposition:** **TECHNICAL PASS — FORMAL B6 PENDING_AUTHORITY** (not CLOSED)

---

## A. Git baseline

Verified @ start: `b669139d1…` == `origin/development`, clean tree. Immutable TES/FINAL/R2 preserved.

## B. Formal B6 authority

| Gate | Result |
|------|--------|
| Mode A+B technical (reuse R2) | **PASS** |
| Local attestor unsigned payload | **READY** (when R2 report present + env) |
| Production signed attestation | **Requires custodian login + external key** |
| Governance `IN_FORCE` | **NO** |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | **unset** |
| Path A OD-EVOL-11 | **auditor_id still missing** |

## C. Local attestation architecture

Implemented: separate **processes** (verifier subprocess, attestor CLI, custodian CLI), Ed25519 verify via existing contract, replay ledger **outside repo**, fail-closed guards. **Not claimed:** OS-level separation until admin runs `Install-IntergraxEvolB6Custody.ps1` + dedicated users.

## D. Governance activation package

[`EVOL_X_LOCAL_B6_ACTIVATION_AND_CUSTODY_PACKAGE.md`](./EVOL_X_LOCAL_B6_ACTIVATION_AND_CUSTODY_PACKAGE.md) — maintainer, security, custody still **pending**; no forged signatures.

## E. Sections 5–8 (closure chain)

**Not executed** — blocked on formal B6. W7-03 Outcome A remains **prepared** from prior campaign; FRZ rows **OPEN**; OD-EVOLX-08 **blocked**; no `FORMAL_CLOSURE_PUBLICATION_SHA`.

## F. Validation

| Suite | Result |
|-------|--------|
| B6 attestation contract + local custody guards + FRZ matrix | **35 passed** |
| `igx_gov ticket` | **PASS** |
| 290 pytest replay | **Skipped** (pinned evidence unchanged) |

Log: `.tmp/session/evol-x-local-b6-automation/`

## G. Publication HEAD

*Set after commit — `LOCAL_B6_AUTOMATION_PUBLICATION_SHA` on ladder optional row.*

## H. Exact authorization package (single unblock chain)

1. **Administrator:** custody directory + separate Windows principals ([`scripts/qualification/local_b6/README.md`](../../scripts/qualification/local_b6/README.md)).  
2. **ECAP maintainer:** signed publication act.  
3. **Security reviewer:** signed [`EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md`](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md).  
4. **Custodian:** key ceremony + trust root + replay ledger.  
5. **OD-EVOL-11:** record `auditor_id` (Path A) **or** valid AUTO-B6 custodian record when Path B live.  
6. Attestor → custodian sign → evaluator `ELIGIBLE` → authorized Protocol 03 publication commit.

**Ref:** [`EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md`](./EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md) · Ticket **T-IGX-011-D**
