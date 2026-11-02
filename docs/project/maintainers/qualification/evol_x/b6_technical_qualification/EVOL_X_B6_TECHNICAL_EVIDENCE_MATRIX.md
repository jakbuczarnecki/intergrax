# EVOL-X B6 — technical evidence matrix (G-xx)

**Profile:** `AUTOMATION_FIRST_SCOPED` · **TES:** `3c4a641b…` · **FINAL:** `bbf3e33b…`

| Gate | Normative ref | Evidence ref | Verified | Technical | Formal | Blocker | Next action |
|------|---------------|--------------|----------|-----------|--------|---------|-------------|
| G-01 | W6-04, CI record | [EVOL_X_CI_QUALIFICATION_RECORD.md](../EVOL_X_CI_QUALIFICATION_RECORD.md), run `41337077334` | Yes | **PASS** | **PASS** (qualified CI) | F-002 local JUnit copy | Optional artifact refresh |
| G-02 | OD-EVOL-11 | Operator register W1 | Yes | **FAIL** | **PENDING** | No named auditor | Record identity |
| G-03 | OD-EVOL-13 / human drill | M-09H NOT_MEASURED @ FINAL | Yes | **N/A (scoped PASS)** | **PASS (scoped)** | — | — |
| G-04 | P03 FINAL publication | FINAL SHA, MANIFEST | Yes | **PASS** | **PASS** | — | — |
| G-05 | Mode A integrity | `a2-mode-a.log`, bundle digests | Yes | **PASS** | **PENDING** auditor attestation | G-07 | Independent Mode A sign-off |
| G-06 | Mode B replay | `a2-mode-b-v2.log`, pytest v2 | Yes | **PASS** (v2) | **PENDING** | G-07 | Independent Mode B sign-off |
| G-07 | B6 acceptance record | — | Yes | **READY** | **PENDING** | No signature / no B6 SHA | W7-01 draft → signed record |
| G-08 | W7-03 FRZ-EVOL-07 | Part X, FRZ dossier | Yes | **READY** | **PENDING** | Outcome A/B | Independent + operator |
| G-09 | OD-EVOL-12 | Register | Yes | **BLOCKED** | **PENDING** | OD open | W7-03 |
| G-10 | OD-EVOLX-08 | Register | Yes | **BLOCKED** | **PENDING** | OD open | W7-04 |
| G-13 | SHA ladder | [EVOL_X_SHA_LADDER.md](../EVOL_X_SHA_LADDER.md) | Yes | **PASS** | **PASS** | B6 SHA empty by design | Set @ W7-01 |
| G-14 | Semantic review (B6 template §6) | M-09A cases, FRZ negatives, experiments | Yes | **PASS (scoped)** | **PENDING** | G-07 | Auditor semantic attestation |

**G-11 / G-12:** Not defined as standalone gates in published EVOL-X B6 package; covered by W7-01 composite and G-14.

## Legend

- **Technical PASS:** Cursor-agent reconciliation @ E3 only.
- **FORMAL ACCEPTED:** Requires named OD-EVOL-11 auditor + published signed record.
- **PENDING_OPERATOR_OR_AUDITOR:** Formal track only.
