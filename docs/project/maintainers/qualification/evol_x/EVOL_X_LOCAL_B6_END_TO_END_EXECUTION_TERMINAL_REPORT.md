# EVOL-X — LOCAL B6 END-TO-END EXECUTION — TERMINAL REPORT

**Campaign ID:** `EVOL-X-LOCAL-B6-END-TO-END-AUTOMATION`  
**Baseline START_HEAD:** `e8de0cb034505ad045c336da435fcaccaec1a49c`  
**Ticket:** [`T-IGX-012-A.md`](../../../governance/tickets/T-IGX-012-A.md)  
**Disposition:** **EVOL-X — LOCAL B6 DEPLOYMENT PARTIAL — EXACT ENVIRONMENT OR AUTHORITY BLOCKERS**

Formal EVOL-X closure, B6 acceptance SHA, FRZ promotion, and W7-05 remain **blocked** (fail-closed). Technical mechanics and consolidated tests **pass** under fixture/synthetic scope; production OS-level custody **not** established in this session.

---

## A. Git baseline and publication SHA

| Check | Result |
|-------|--------|
| Branch | `development` |
| START_HEAD | `e8de0cb034505ad045c336da435fcaccaec1a49c` (matches expected) |
| Ancestor of START_HEAD | **yes** |
| Working tree @ campaign start | **clean** |
| `END_TO_END_CAMPAIGN_PUBLICATION_SHA` | `0dab7bfb8db67b431581bd840b094e046335523e` |

Immutable pins preserved: TES `3c4a641b3e48a13f437e830a0562e2a80b03aabb` · FINAL `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` · CI `41337077334` · R2 `167583ad1d33d9eb6e82589661e6d18aee15fa92`.

---

## B. Windows installation results

| Step | Result |
|------|--------|
| `Install-IntergraxEvolB6Custody.ps1` | **BLOCKED** — `#Requires -RunAsAdministrator`; session `desktop-icfd1hr\xps` **not elevated** |
| `C:\ProgramData\Intergrax\evol-b6-custody\` | **absent** (not created without admin) |
| Script review | **PASS** — creates custody tree, denies write/delete to invoking admin user; **does not** generate keys, users, or ECAP activation |
| Script repair (campaign) | Added `ledger` + `archive` subdirs to match custody package topology |

**Environmental gate:** Run PowerShell **as Administrator** once, then create `IntergraxEvolAttestor` / `IntergraxEvolCustodian` and ACL grants per [`scripts/qualification/local_b6/README.md`](../../../scripts/qualification/local_b6/README.md).

---

## C. Actual Windows principal separation

| Principal | Present | Notes |
|-----------|---------|-------|
| Implementer (`desktop-icfd1hr\xps`) | yes | Cursor agent session |
| `IntergraxEvolAttestor` | **no** | `Get-LocalUser` empty |
| `IntergraxEvolCustodian` | **no** | — |

**Label:** `NOT_INDEPENDENT` — attestor CLI ran under implementer with env flags only; **not** ECAP-P4 organizational/OS separation.

---

## D. NTFS ACL verification

No production custody tree to inspect. Install script design: inherit deny **Write,Delete,ChangePermissions,TakeOwnership** on root for provisioning admin; custodian/attestor ACLs **not applied** (install not run).

---

## E. Key custody and trust-root status

| Item | Status |
|------|--------|
| Production Ed25519 private key | **absent** |
| Public trust root @ ProgramData | **absent** |
| Replay ledger @ custody | **absent** |

No substitute production key generated (fail-closed).

---

## F. Attestor execution result

| Field | Value |
|-------|-------|
| R2 report reused | `.tmp/session/evol-x-r2-final-reconciliation/run-2026-11-04/independent_verification_report.json` |
| Mode A / B (from report) | **PASS** / **PASS** (reuse) |
| Unsigned payload | **READY** @ `.tmp/session/evol-x-local-b6-end-to-end/pending/unsigned.json` |
| Principal | **implementer** + `EVOL_B6_ATTESTOR_*` env — **not independent** |

Log: `.tmp/session/evol-x-local-b6-end-to-end/attestor.log`

---

## G. Custodian signing result

**Not executed.** `evol_x_b6_custodian_sign.py` refused without `EVOL_B6_CUSTODIAN_MODE=1` and without custodian login/key path (expected). No production signature.

Log: `.tmp/session/evol-x-local-b6-end-to-end/custodian.log`

---

## H. Evaluator result

**No production signed attestation** to evaluate. Contract tests (`test_evol_x_b6_attestation_contract.py`) confirm: valid crypto + `NOT_IN_FORCE` governance → **`NOT_ELIGIBLE`**; synthetic `ELIGIBLE` only under test fixtures with explicit in-force governance — **not** claimed for production.

---

## I. Security review result

Automated negative/positive coverage via pytest (ACL guards, custodian mode, replay, revocation, policy downgrade, implementer attestor rejection). Unsigned maintainer record remains: [`EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md`](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md) — **no independent security signature** in this session.

Technical evidence bundle (unchanged): [`b6_technical_qualification/EVOL_X_AUTO_B6_SECURITY_REVIEW_TECHNICAL_EVIDENCE_UNSIGNED.md`](./b6_technical_qualification/EVOL_X_AUTO_B6_SECURITY_REVIEW_TECHNICAL_EVIDENCE_UNSIGNED.md).

---

## J. ECAP-P4 activation status

From [`EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json`](./EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json):

| Field | Value |
|-------|-------|
| `ecap_amendment_activation` | `PENDING_REQUIRED_APPROVALS` |
| `normative_effect` | **`NOT_IN_FORCE`** |
| `required_approvals_present.security_review` | **false** |
| `required_approvals_present.independent_key_custody` | **false** |
| `required_approvals_present.ecap_maintainer_publication` | **false** |

Proposed amendment does **not** become effective through this campaign.

---

## K. Formal Protocol 03 B6 acceptance status

**`PENDING_AUTHORITY`** — no signed Protocol 03 PASS; `B6_ACCEPTANCE_PUBLICATION_SHA` **unset**; OD-EVOL-11 `auditor_id` still missing.

---

## L. FRZ-EVOL-01..10 status

**Not promoted** — all mandatory formal rows remain **OPEN** pending B6. W7-03 Outcome A package remains **prepared only** (unsigned).

---

## M. OD-EVOL-12 / OD-EVOLX-08 status

**Blocked** — packages exist from prior campaign; no new signed operator decisions in this session.

---

## N. W7-05 / W7-06 result

**Not executed.** No `FORMAL_CLOSURE_PUBLICATION_SHA`. ENT-AUDIT-X formal handoff pin **blocked**.

---

## O. Tests and independent validation

| Suite | Result |
|-------|--------|
| B6 contract + custody guards + FRZ matrix + P03 integrity | **87 passed** (~232s) |
| 290-test VL3 replay | **skipped** (TES pin unchanged; no verifier/policy drift) |

Log: `.tmp/session/evol-x-local-b6-end-to-end/pytest.log`

`igx_gov ticket`: run @ publication (§Q).

---

## P. Final SHA ladder

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| R2_GLOBAL_AUDIT_RECONCILIATION_PUBLICATION_SHA | `167583ad1d33d9eb6e82589661e6d18aee15fa92` |
| B6_ACCEPTANCE_PUBLICATION_SHA | **— (pending)** |
| FORMAL_CLOSURE_PUBLICATION_SHA | **— (pending)** |
| END_TO_END_CAMPAIGN_PUBLICATION_SHA | `0dab7bfb8db67b431581bd840b094e046335523e` |

---

## Q. Commit/push confirmation

| Item | Value |
|------|-------|
| Publication commit | `0dab7bfb8db67b431581bd840b094e046335523e` |
| Branch | `development` |
| Remote | `origin/development` (push @ campaign close) |
| `igx_gov ticket` | **PASS** (pre-push @ `e8de0cb0…` working tree) |
| Working tree | clean post-commit |

Checkpoint log: `.tmp/session/evol-x-local-b6-end-to-end/igx_gov.log`

---

## R. Exact remaining external-authority gates

1. **Administrator (UAC):** run `Install-IntergraxEvolB6Custody.ps1`; create Windows users; ACL subfolders for attestor/custodian only.  
2. **Custodian login:** Ed25519 key ceremony + public trust root + replay ledger under custody.  
3. **Attestor login:** read-only clone; run `evol_x_b6_local_attestor.py` → `pending\unsigned.json`.  
4. **Custodian:** sign → `signed\attestation.json`; evaluator **`ELIGIBLE`** only when governance **`IN_FORCE`**.  
5. **ECAP maintainer** publication act + **independent security review** signature.  
6. **OD-EVOL-11:** record qualified `auditor_id` (Path A).  
7. Authorized **Protocol 03 PASS** commit → `B6_ACCEPTANCE_PUBLICATION_SHA` → FRZ/W7/closure chain.

**Ref:** [`EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md`](./EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md)
