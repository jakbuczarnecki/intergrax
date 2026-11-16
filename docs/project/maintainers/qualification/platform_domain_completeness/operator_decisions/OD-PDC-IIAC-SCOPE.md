# OD-PDC-IIAC-SCOPE — Independent certification process for PDC-X

| Field | Value |
| --- | --- |
| **Decision ID** | OD-PDC-IIAC-SCOPE |
| **Stage** | PDC-X |
| **Status** | **APPROVED — Option A** |
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

## Session plan (refined in PDC-P3-04)

| Session | Role | Scope | Separation |
| --- | --- | --- | --- |
| IA-D1 … IA-D8 | Independent auditor (one per domain) | Requirement rows of that domain in `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, at `Dn_AUDITED_SHA` | New clean session; never an implementer of that domain's P5 batch; at most 3 concurrent sessions (test process budget) |
| CERT-PDC | Certification / governance agent | Consolidates D1–D8 verdicts into `PDC_DOMAIN_CERTIFICATION_MATRIX`; checks manifests and role separation | Not an auditor and not an implementer |
| IA-GLOBAL | Independent global auditor (PDC-P8) | Cross-domain matrix, FRZ mapping, SHA ladder, ECAP §2.C run | New session; not used in D1–D8 audits |
| Signing delegate(s) | Operator delegate (Amendment 01) | OD signatures only where the Operator commissions them | Never implementer, auditor or certifier of the same item |

Audit manifests follow `schemas/iiac_audit_session_manifest.v1.schema.json`. Evidence goes under `iiac_evidence/PDC-X/<D1..D8|global>/`. ACT-N11 activation is bound at `INTEGRATED_BASELINE_SHA` (PDC-P7-00) and rebound at the global audit SHA (PDC-P8-04).

## Open sub-question

- Should the IIAC protocol header wording (`NOT_IN_FORCE` at lines 4 and 91, finding PDC-BL-F01) be corrected inside the next MCP-SERVER-X or PDC activation batch? Either way the correction needs a rebind, because the file is a material governance path.

## Operator record

Approved: operator @ 2026-11-12T07:11:25+01:00

**Selected variant:** Option A (recommended).

Provenance: the Operator answered the P3-gate question "Accept the recommended option (A) for all P3-gate ODs?" in chat with the words "Accept all A". This record was transcribed on the Operator's instruction under T-IGX-036-E.

**Still open:** the sub-question on the IIAC protocol header wording (PDC-BL-F01) is not decided by Option A; it is resolved in the next activation batch.
