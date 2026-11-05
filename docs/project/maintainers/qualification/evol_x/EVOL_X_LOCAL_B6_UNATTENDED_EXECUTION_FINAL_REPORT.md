# EVOL-X — LOCAL B6 UNATTENDED EXECUTION — FINAL REPORT

**Campaign:** `EVOL-X-LOCAL-B6-UNATTENDED-EXECUTION`  
**Ticket:** [`T-IGX-012-B.md`](../../../governance/tickets/T-IGX-012-B.md)  
**Baseline @ start:** `bffd689d3b127bbcd753f0cf6f2c863c4db4a032` (`origin/development`)  
**Disposition (continuation @ 2026-11-05):** **PIPELINE INCOMPLETE — UAC SECURE DESKTOP NOT APPROVED IN AGENT WINDOW** — custody/users/ACL partial; **no production key/signature**; formal gates unchanged.

---

## Continuation run (same campaign, no new ticket)

| Action | Result |
|--------|--------|
| RunAs launcher | **Multiple invocations** + 3 min UAC wait window; **no fresh `elevated-result.json`** without interactive Secure Desktop approval |
| Bootstrap fixes | Python arg join; `Start-Process -Credential` + wrapper scripts (replaces flaky scheduled-task race **267011**); launcher **UAC timeout** (exit 5); targeted repo/python RX grants |
| `sudo.exe` | **Disabled** on host (cannot use Developer sudo path) |
| Implementer NTFS probe (non-elevated) | **Write denied** on `private\k-prod-local-1.ed25519`; key **absent**; `elevated-launch.json` **`implementer_can_read_private_key: false`** |
| Isolation caveat | **Separate SAM accounts exist**; **full attestor→custodian pipeline not proven** until elevated bootstrap completes; **Administrators** can override ACL during elevated sessions — not claimed as organizational custody |

---

## Administrative execution

| Item | Result |
|------|--------|
| Mechanism | `Invoke-IntergraxEvolB6ElevatedLaunch.ps1` → `Start-Process -Verb RunAs` → `Invoke-IntergraxEvolB6UnattendedBootstrap.ps1` |
| Agent session admin | **No** (non-elevated implementer) |
| UAC / RunAs | **Invoked repeatedly**; succeeds when operator approves elevation |
| Structured result | `.tmp/session/evol-x-local-b6-unattended/elevated-result.json` |
| Launch meta | `.tmp/session/evol-x-local-b6-unattended/elevated-launch.json` (when completed) |

**Non-automatable gate:** Windows UAC consent for `#Requires -RunAsAdministrator` bootstrap (cannot be bypassed or forged).

---

## Windows account provisioning

| SAM account (≤20 chars) | Status |
|-------------------------|--------|
| `IntergraxEvolAttst` | **Present**, enabled (logical **IntergraxEvolAttestor**) |
| `IntergraxEvolCustn` | **Present**, enabled (logical **IntergraxEvolCustodian**) |

Note: canonical names exceed Windows SAM 20-character limit; bootstrap documents mapping in `sam_account_note`.

---

## Principal isolation

| Check | Result |
|-------|--------|
| Separate local users | **Yes** |
| Attestor/custodian runs | **`Start-Process -Credential`** wrapper under SAM accounts (elevated bootstrap caller) |
| Full pipeline under isolated principals | **Incomplete** — last elevated pass failed key ceremony (**267011** scheduled-task race, since fixed); **re-run blocked on UAC** in agent session |

---

## Custody ACLs

| Path | Status |
|------|--------|
| `C:\ProgramData\Intergrax\evol-b6-custody\` | **Exists** (install OK) |
| Subdirs | `private`, `public`, `pending`, `signed`, `ledger`, `audit`, `archive` |
| ACL configuration | **CONFIGURED** (icacls; last elevated result @ 2026-11-05T06:17:57Z) |
| Implementer private-key probe | **Deferred** to non-elevated launcher (admin token skew) |

---

## Key creation / trust root / replay

| Item | Status |
|------|--------|
| Production key `k-prod-local-1.ed25519` | **Not present** @ campaign end |
| `public\trust_root.v1.json` | **Not published** |
| `ledger\replay_ledger.v1.json` | **Not created** |
| Module | `scripts/qualification/local_b6/evol_x_b6_local_key_ceremony.py` (custodian mode only) |

---

## Attestor / custodian signer / evaluator

| Stage | Status |
|-------|--------|
| Attestor (isolated) | **Not completed** (blocked on key/trust root) |
| Custodian sign | **Not completed** |
| Evaluator | **Not run** on production signed artifact |
| Governance | **`NOT_IN_FORCE`** → evaluator would return **`NOT_ELIGIBLE`** even if crypto OK |

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
| B6 contract + custody guards + FRZ matrix | **35 passed** |
| 290 VL3 replay | **Skipped** (TES pin unchanged) |

Log: `.tmp/session/evol-x-local-b6-unattended/pytest.log`

---

## Git publication

| Item | Value |
|------|-------|
| `UNATTENDED_EXECUTION_PUBLICATION_SHA` | `0356554cf8065a10a00b925afbdab4327375beaf` |
| Branch | `development` → `origin/development` |

---

## Remaining gates (ordered)

1. **UAC approval** for one elevated bootstrap completion (or persistent admin session for agent).  
2. Re-run `Invoke-IntergraxEvolB6ElevatedLaunch.ps1` (idempotent) to finish key → attestor → custodian → evaluator under isolated principals.  
3. ECAP maintainer + independent security review + OD-EVOL-11 auditor_id.  
4. Authorized Protocol 03 publication → FRZ / W7 closure chain.

**Ref:** [`EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md`](./EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md)
