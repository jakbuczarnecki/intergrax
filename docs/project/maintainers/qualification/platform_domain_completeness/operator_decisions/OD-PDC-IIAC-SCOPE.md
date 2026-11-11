# OD-PDC-IIAC-SCOPE — Independent certification process for PDC-X

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-IIAC-SCOPE |
| **Stage** | PDC-X |
| **Status** | **PENDING** |
| **Policy** | `IIAC-2026-11-05.1` (activation record `IN_FORCE`; protocol §Purpose names EVOL-X and ENT-AUDIT-X only) |

## Question

Which internal independent certification process applies to PDC-X and its eight workstreams?

## Recommended option (A)

- Extend `IIAC-2026-11-05.1` to PDC-X through a per-stage activation decision (the MCP-SERVER-X precedent). The IIAC protocol text stays as it is.
- Roles:
  - One independent auditor session per domain (`IA-D1`..`IA-D8`). Each is a clean session and never the implementer of that domain's batches.
  - One certification/governance agent (`CERT`), separate from the auditors.
  - One new global auditor session for PDC-P8.
- `IMPLEMENTER != INDEPENDENT AUDITOR != CERTIFIER`, checked by `verify_iiac_role_separation.py` and `verify_iiac_audit_manifest.py`.
- ACT-N11 activation at `INTEGRATED_BASELINE_SHA` (PDC-P7-00) and a rebind at the global audit SHA (PDC-P8-04).
- Evidence goes under `docs/project/maintainers/qualification/iiac_evidence/PDC-X/<Dn|global>/`.
- Verdict vocabulary: PASS, PASS_SCOPED (only for Operator-authorized exclusions), PARTIAL, FAIL.

## Alternate option (B)

- A separate certification process with equivalent separation of roles. This needs its own protocol ticket and validator coverage.

## Open sub-question

- Should the IIAC protocol header wording (`NOT_IN_FORCE` at lines 4 and 91, finding PDC-BL-F01) be corrected inside the next MCP-SERVER-X or PDC activation batch? Either way the correction needs a rebind, because the file is a material governance path.

## Operator record

_Pending._
