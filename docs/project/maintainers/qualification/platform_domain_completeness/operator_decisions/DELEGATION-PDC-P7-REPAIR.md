# DELEGATION-PDC-P7-REPAIR — Operator decisions on the repair scope after the P7 domain audits

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-PDC-P7-REPAIR |
| **Stage** | PDC-X P7 (repair loop 1), P8 preparation |
| **Basis** | IIAC Amendment 01, rules R-1..R-6 (as summarized in `governance/tickets/README.md` §Approved); `DELEGATION-PDC-STANDING.md`; `DELEGATION-PDC-WAVE2.md` (decisions 6 and 8); `OD-PDC-IIAC-SCOPE.md` (verdict vocabulary) |
| **Date** | 2026-11-15 |

## Commission (transcription)

The eight independent domain audits IA-D1..IA-D8 (audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, reports `PDC_P7_D<n>_AUDIT_REPORT.md`) rated D5 FAIL and the other seven domains PARTIAL. Roughly fifteen findings follow one pattern (a component exists but has no production caller or composition, dossier finding F-04) and eight claim a feature that has no implementation. The orchestrator asked the Operator two questions; the answers below are English renderings of the Operator's Polish choices.

1. **Repair scope.** The Operator chose **option B** ("B: real defects + hygiene"). Repair the HIGH and MEDIUM findings that are real defects in existing code (bugs, tenant-scope gaps, fail-open paths, missing validation, product bugs behind red cited suites, stale or red cited tests, overstating documentation) and test hygiene.
2. **Hotspots and the TENANT-X pin.** The Operator answered **yes**: repair tickets may change files under `intergrax/applications/_shared/` and `intergrax/contracts/` when the ticket names the exact files, and the TENANT-X carrier-count gate (`tests/qualification/tenant_x/test_tenant_x_p0_current_head.py`, count 831 against pin 819) may be re-pinned once, after the last domain repair.

## Effect of option B

- Findings of the "no production caller or composition" family and findings of the missing-feature-versus-claim kind (for example F-D7-03, F-D5-05, F-D2-05, F-D2-07, F-D3-03, F-D3-07, F-D6-02 and the AUDIT_EVIDENCE producer part of F-D8-01) are **not repaired in PDC-X**. They are recorded as declared, labelled exceptions ("not wired" or "not implemented", deferred) with a reopen condition in the exception register of the P9 closure package. The affected requirement rows are rated PASS_SCOPED by this Operator authorization (verdict vocabulary of `OD-PDC-IIAC-SCOPE.md`); no row is silently upgraded.
- LOW and INFO findings go to the debt register of the P9 closure package unless a repair ticket closes them as a by-product.
- Gates are only strengthened, classified or re-pinned with classification; never weakened or deleted.

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>` and `Delegation: <this file>@<commit sha>` in the tickets **T-IGX-041-A .. T-IGX-041-I** only (A = D1, B = D2, C = D3, D = D4, E = D5, F = D6, G = D7, H = D8, I = TENANT-X re-pin), after confirming for each ticket:
  - the plan digest matches, and the plan matches its batch row in `PDC_IMPLEMENTATION_WBS.md` (write sets equal);
  - every hotspot path in the write set is named exactly (no directory wildcards);
  - the plan lists the in-scope findings and every excluded finding with its reason and reopen condition;
  - no gate is weakened or deleted (strengthening, classification or re-pin only, each with negative tests);
  - the plan adds no registry, scheduler, engine, trust, policy or audit authority, and changes nothing under `scripts/governance/` or `.github/`;
  - the plan states it is not a certification verdict; implementer and signer are different sessions.
- **Merge gate.** Product pull requests of these tickets merge under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation staleness after product changes until the P8-04 rebind; the EBH-7 identity gate until the final re-pin), head equals the verified SHA.
- **Loops.** Up to two audit and repair loops per domain (`DELEGATION-PDC-WAVE2.md` decision 6). A fresh auditor re-audits each corrected domain at the corrected SHA; a third loop returns to the Operator.

## Not delegated (unchanged)

OD signatures; FRZ promotion and the PDC-X closure line; lifting C10/C15; write-set amendments of any ticket above; changing the meaning of a gate beyond strengthening, classification or the one TENANT-X re-pin; the fail-closed rule for a revoked revision without a trusted rollback target (F-D1-04) and the MVP promotion gate semantics (F-D5-02) unless the plan only strengthens them.

## Separation

The implementer session and its subagents never write `Approved:` or activation lines. A delegate session does not implement or audit what it signs. An auditor never certifies and never fixes what it audits.
