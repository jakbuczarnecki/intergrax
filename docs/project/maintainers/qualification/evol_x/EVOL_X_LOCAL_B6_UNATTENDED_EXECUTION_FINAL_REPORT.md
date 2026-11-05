# EVOL-X — LOCAL B6 UNATTENDED EXECUTION — FINAL REPORT

**Campaign:** `EVOL-X-LOCAL-B6-UNATTENDED-EXECUTION`  
**Ticket:** [`T-IGX-012-B.md`](../../../governance/tickets/T-IGX-012-B.md)  
**Baseline @ start:** `bffd689d3b127bbcd753f0cf6f2c863c4db4a032` (`origin/development`)  
**Disposition (terminal @ 2026-11-05):** **`TECHNICAL_PIPELINE_OK_PENDING_AUTHORITY`** — key ceremony, attestor, custodian **completed** under SAM principals; evaluator **`NOT_ELIGIBLE`** (governance **`NOT_IN_FORCE`**). **Not** B6 formal acceptance, **not** ECAP `IN_FORCE`.

---

## Superseded runs (historical)

| Run | Outcome | Notes |
|-----|---------|-------|
| Launcher **EXIT=5** @ ~07:35 UTC | **Superseded** | 15 min UAC/wait timeout; partial ACL; key ceremony **103** (uv Python RX for SAM users — fixed) |
| Multiple parallel elevated windows | **Superseded** | Same-user elevation without Secure Desktop prompt; **`icacls` Invalid parameter** — fixed (per-grant + SID grants) |
| Continuation @ morning | **Superseded** | `PIPELINE INCOMPLETE — UAC` narrative replaced by successful pass @ `2026-11-05T08:00:08Z` |

Evidence (session, gitignored): `.tmp/session/evol-x-local-b6-unattended/elevated-result.json`, `elevated-launch.json`.

---

## Administrative execution

| Item | Result |
|------|--------|
| Mechanism | `Invoke-IntergraxEvolB6ElevatedLaunch.ps1` → `Start-Process -Verb RunAs` → `Invoke-IntergraxEvolB6UnattendedBootstrap.ps1` |
| Agent session admin | **No** (non-elevated implementer) |
| UAC / RunAs | Same-user elevation; **no** separate Secure Desktop prompt required on host after first approval |
| Structured result | `.tmp/session/evol-x-local-b6-unattended/elevated-result.json` |
| Launch meta | `.tmp/session/evol-x-local-b6-unattended/elevated-launch.json` |

---

## Windows account provisioning

| SAM account (≤20 chars) | Status |
|-------------------------|--------|
| `IntergraxEvolAttst` | **Present**, enabled (logical **IntergraxEvolAttestor**) |
| `IntergraxEvolCustn` | **Present**, enabled (logical **IntergraxEvolCustodian**) |

Note: canonical names exceed Windows SAM 20-character limit; bootstrap documents mapping in `sam_account_note`.

---

## Principal isolation and ACL (technical, local)

| Check | Result |
|-------|--------|
| Separate local SAM users | **Yes** |
| Attestor/custodian runs | **`Start-Process -Credential`** under SAM accounts (elevated bootstrap) |
| `attestor_can_read_private_key` (elevated probe) | **`false`** |
| `implementer_can_read_private_key` (launcher, non-elevated) | **`false`** |
| Implementer read `private\*.ed25519` | **Denied** (open + `Test-Path` unauthorized) |
| Implementer read `signed\attestation.json` | **Denied** |
| **Caveat — not full organizational custody** | Bootstrap creates accounts and passwords on **one workstation**; **Administrators** / elevated implementer can change ACLs; SAM isolation **≠** HSM or independent security org; do **not** infer ECAP Path A production independence from NTFS alone |

---

## Custody ACLs

| Path | Status |
|------|--------|
| `C:\ProgramData\Intergrax\evol-b6-custody\` | **Exists** |
| Subdirs | `private`, `public`, `pending`, `signed`, `ledger`, `audit`, `archive` |
| ACL configuration | **`CONFIGURED`** (@ `2026-11-05T08:00:08Z`) |
| Repo / uv execute grants | Marker **`repo-acl-v1.done`** — targeted RX (no repeated `/T` walk) |

---

## Key creation / trust root / replay

| Item | Status |
|------|--------|
| Production key `k-prod-local-1.ed25519` | **Created** (ceremony exit **0**) |
| `public\trust_root.v1.json` | **Published** |
| `ledger\replay_ledger.v1.json` | **Created** (custodian path) |
| Module | `scripts/qualification/local_b6/evol_x_b6_local_key_ceremony.py` |

---

## Attestor / custodian signer / evaluator

| Stage | Exit | Notes |
|-------|------|-------|
| Key ceremony (custodian) | **0** | Trust root @ `public\trust_root.v1.json` |
| Attestor (isolated) | **0** | Pending → signed pipeline |
| Custodian sign | **0** | `signed\attestation.json` |
| Evaluator (attestor credential) | **1** | **`formal_acceptance_eligibility`: `NOT_ELIGIBLE`** — expected under **`NOT_IN_FORCE`**; disposition still **`TECHNICAL_PIPELINE_OK_PENDING_AUTHORITY`** (crypto path completed; formal gate blocked) |
| Evaluator artifact | `C:\ProgramData\Intergrax\evol-b6-custody\audit\evaluator-result.json` | ProgramData (implementer may lack list/read) |

---

## Governance

| Item | Status |
|------|--------|
| `EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json` | **`NOT_IN_FORCE`** |
| Evaluator eligibility | **`NOT_ELIGIBLE`** (not forged `IN_FORCE`) |

---

## ECAP activation / Protocol 03 / FRZ / W7

| Gate | Status |
|------|--------|
| ECAP-P4 `IN_FORCE` | **No** |
| Maintainer / security signatures | **Not forged** |
| Protocol 03 B6 PASS | **Not published** |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | **unset** |
| FRZ-EVOL-01..10 | **OPEN** |
| OD-EVOL-12 / OD-EVOLX-08 / W7-05 / W7-06 | **Not executed** |

---

## Tests

| Suite | Result |
|-------|--------|
| B6 contract + custody guards + FRZ matrix | **35 passed** (@ terminal commit) |
| 290 VL3 replay | **Skipped** (TES pin unchanged) |
| `igx_gov ticket` | **PASS** (@ terminal commit, `PYTHONPATH` unset) |

Logs: `.tmp/session/evol-x-local-b6-unattended/pytest-final.log`, `igx-gov-ticket.log`

---

## Git publication

| Item | Value |
|------|-------|
| `LOCAL_B6_UNATTENDED_EXECUTION_PUBLICATION_SHA` | **See [`EVOL_X_SHA_LADDER.md`](./EVOL_X_SHA_LADDER.md)** (terminal row @ this report) |
| Branch | `development` → `origin/development` |

---

## Remaining gates (formal only)

1. ECAP amendment **`IN_FORCE`** + maintainer / independent security signatures (OD-EVOL-11 `auditor_id`, etc.).  
2. Authorized **Protocol 03** B6 publication → set **`B6_ACCEPTANCE_PUBLICATION_SHA`**.  
3. **FRZ-EVOL-*** promotion and **W7-05 / W7-06** closure chain.  
4. Organizational acceptance of local Path A custody (beyond NTFS + SAM on one host).

**Ref:** [`EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md`](./EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md)
