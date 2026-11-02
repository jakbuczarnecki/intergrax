# PROPOSED (NOT APPROVED) — ECAP amendment: automated organizational B6 for EVOL-X scoped path

**Status:** **DRAFT / NOT OPERATOR APPROVED / NOT IN FORCE**  
**Purpose:** Record the minimum ECAP change that would be required to treat a **deterministic verifier + custody chain** as satisfying **organizational** independent B6 — **only if** the operator explicitly adopts it.

**This document is not approval.** OD-EVOL-13 does **not** authorize automated B6 sign-off. OD-EVOL-11 still requires a **named** eligible auditor ≠ implementer ≠ Cursor agent.

## Problem statement

Current binding authority (ECAP-P4, master plan W7-01, OD-EVOL-11):

| Requirement class | Current rule |
|-------------------|--------------|
| Technical independence | Separate replay @ exact SHAs; Mode A/B; digest chain — **automatable** (see `evol_x_b6_independent_verification.py`) |
| Organizational independence | Human or explicitly appointed **non-implementer** auditor signs Protocol 03 PASS |
| Immutable acceptance record | `EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` @ `B6_ACCEPTANCE_PUBLICATION_SHA` |
| FRZ promotion | Operator OD-EVOL-12 + W7-03 after formal B6 |

Cursor multi-agent review and implementer-run verification **do not** satisfy organizational independence.

## Proposed policy text (ECAP § amendment — inactive)

> **ECAP-P4-EVOL-SCOPE-AUTO-B6 (proposed):** For **EVOL-X** closure under **`certification_profile=AUTOMATION_FIRST_SCOPED`** and **OD-EVOL-13** only, organizational B6 acceptance MAY be satisfied by a **custody-separated automated auditor service** that: (1) executes in an environment with **no write access** to `development` and **no** agent implementer credentials; (2) emits a **signed attestation** bound to `test_execution_sha`, `evidence_final_publication_sha`, and Mode A/B logs; (3) uses a **key custody owner** distinct from EVOL-X implementers; (4) **does not** promote `FRZ-*` or declare EVOL-X CLOSED — operator ODs W7-03..05 remain mandatory.

## Authority required to activate

1. Operator decision row (new **OD-EVOL-14** or amendment to **OD-EVOL-11** procedure) — **SIGNED**.
2. ECAP maintainer publication of in-force **ECAP-P4-EVOL-SCOPE-AUTO-B6** (not this draft).
3. Security review of attestation key custody (out of scope for Cursor agent).

## Without amendment — permitted today

- Automated **technical** B6 preparation (E3): dossier, Mode A/B replay, findings register.
- `scripts/qualification/evol_x_b6_independent_verification.py` — mechanical verification only.
- Operator names auditor → human/eligible party signs PASS record → `B6_ACCEPTANCE_PUBLICATION_SHA` → E4 W7-03..06.

## Decision dossier pointer

Formal authority blockers and path to CLOSED: [b6_technical_qualification/EVOL_X_FULL_AUTOMATION_CLOSURE_WAVE_REPORT.md](./b6_technical_qualification/EVOL_X_FULL_AUTOMATION_CLOSURE_WAVE_REPORT.md).
