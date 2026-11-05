# ENT-AUDIT-X + EVOL-X — Cross-stage decision matrix

**Status:** **DRAFT — NOT SIGNED — NOT ACCEPTED** (formal B6/OD chain) · **EVOL-X internal closure @ 2026-11-05** · **Internal certification @ OD-IC-01 (2026-11-04)**  
**START_HEAD @ preparation:** `0e538269a7188075003b9ff2c7cce2545d8f1ccb`

**IIAC path (NOT IN FORCE @ Session A):** After activation, formal B6/closure uses [INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md](./INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md) — closure string `CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`. GitHub admin remains **external infra**, not a blocker to IIAC CLOSED when compensating OD-ENT-02 is accepted.

| Gate | Stage | Required before | Owner | Evidence | State | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| EVOL internal closure | EVOL-X | OPERATOR-DIRECTED INTERNAL CLOSURE CAMPAIGN | Internal auditor session | [EVOL_X_INTERNAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md](./evol_x/EVOL_X_INTERNAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md) | **INTERNALLY CLOSED — AUTOMATION_FIRST_SCOPED** | Formal B6 when authorized |
| EVOL → ENT internal handoff | EVOL-X → ENT | Internal parent close | Maintainer docs | [EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md](./evol_x/EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md) | **INTERNAL HANDOFF COMPLETE** | Formal W7-06 pin still OPEN |
| Internal certification (EVOL) | EVOL-X | Operator OD-IC-01 | Operator | TES/FINAL/CI + OD-EVOL-17..19 | **INTERNALLY CERTIFIED — OPERATOR ACCEPTED (SCOPED)** | Superseded in disposition by internal closure row above |
| Internal certification (ENT) | ENT-AUDIT-X | Operator OD-IC-01 | Operator | TES/B5/re-audit + W8 disposition | **INTERNALLY CERTIFIED — OPERATOR ACCEPTED (TECHNICAL SCOPE)** | Formal Protocol 03 B6 when authorized |
| EVOL B6 | EVOL-X | OD-EVOL-11 / IIAC handoff when active | Independent auditor (IIAC session after activation) | TES `3c4a641b…`, FINAL `bbf3e33…`, CI `41337077334` | **OPEN** | IIAC manifest + `B6_ACCEPTANCE_PUBLICATION_SHA` (legacy human Protocol 03 until IN_FORCE) |
| EVOL FRZ | EVOL-X | EVOL B6 ACCEPT | Operator | FRZ-EVOL-01..10 checklist | **OPEN** | Operator FRZ promotion after B6 |
| EVOL W7-06 handoff (formal) | EVOL-X | W7-05 formal closure publication | Operator / maintainer | Handoff record formal pin | **OPEN** | Publish `FORMAL_CLOSURE_PUBLICATION_SHA` |
| EVOL W7-06 handoff (internal) | EVOL-X | Internal closure campaign | Internal auditor session | Handoff record @ internal SHA | **COMPLETE** | ENT intake may use internal package |
| EVOL formal CLOSED | EVOL-X | B6 + FRZ + W7 + §5 | Operator + auditor | Reconciliation + terminal campaign | **PARTIAL** (technical + R2; authority blocked) | [`EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md`](./evol_x/EVOL_X_FORMAL_CLOSURE_AUTHORITY_BLOCKED_DOSSIER.md) |
| OD-ENT-00 | ENT-AUDIT-X | EVOL disposition documented | Operator | [EVOL_X_FORMAL_CLOSURE_RECONCILIATION_PACKAGE_DRAFT.md](./evol_x/EVOL_X_FORMAL_CLOSURE_RECONCILIATION_PACKAGE_DRAFT.md) | **OPEN** | Accept EVOL formal status for ENT dependency |
| ENT B6 | ENT-AUDIT-X | B5 @ TES; technical re-audit | Independent auditor (IIAC after activation) | TES `89c76eb7…`, EVID `30085953…`, re-audit `0e538269…` | **OPEN** (tech rec.) | IIAC pass record + manifest (legacy human B6 until IN_FORCE) |
| OD-ENT-01 | ENT-AUDIT-X | ENT B6 ACCEPT | Operator + IIAC certifier when active | Trust chain L6 | **OPEN** | Promote `ACCEPTED_BASELINE_SHA` |
| OD-ENT-02 | ENT-AUDIT-X | ENT B6 path | GitHub admin **or** IIAC compensating (Option B) | LIVE: `verify_ent_audit_branch_protection.py` · COMPENSATING: [`OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md`](./ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md) · [operator packet](./ent_audit_x/OD_ENT_02_OPERATOR_PACKET.md) | **OPEN** | LIVE preferred; compensating without equivalence claim |
| Branch protection | Platform | OD-ENT-02 | GitHub admin | CI runbook | **EXTERNAL** | Admin applies rules |
| Accepted baseline | ENT-AUDIT-X | OD-ENT-01 | Operator | Ladder | **OPEN** | Record + promote CLI |
| Normal FREEZE | ENT-AUDIT-X | Accepted baseline | Operator / qual | PL-08 | **OPEN** | Post-B6 qual path |
| FRZ-AUD-01..10 | ENT-AUDIT-X | ENT B6 + evidence | Operator | Freeze checklist | **OPEN** | Promote after OD-ENT-08 path |
| OD-ENT-08 | ENT-AUDIT-X | B6 + baseline + FRZ path | Operator | ECAP closure | **OPEN** | Formal ENT stage closure |
| ENT formal closure SHA | ENT-AUDIT-X | OD-ENT-08 | Operator | Roadmap §5 | **OPEN** | Publication commit |
| Roadmap §5 | Program | Stage closure acts | Operator | [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) | **OPEN** | Ledger update after signatures |
| ARCH-FREEZE handoff | Program | ENT formal CLOSED + predecessors | Operator | ENT handoff doc | **OPEN** | Post ENT closure |
| BLK-ROADMAP-ENT-DEP | ENT-AUDIT-X | OD-ENT-00 | Operator | Cross-stage matrix | **OPEN** | Do not mark CLOSED without OD |

## Authority separation

| Actor | May | May not |
| --- | --- | --- |
| Cursor Agent | Prepare drafts, fix doc drift, validate mechanics | Sign B6, OD, FRZ, CLOSED |
| Independent auditor | Protocol 03 attestation within mandate | Operator baseline promotion |
| Operator | OD records, FRZ promotion, roadmap §5 | Substitute for auditor B6 |
| GitHub admin | Branch protection, required checks | Protocol 03 attestation |

## Recommended formal signature order

1. **OD-EVOL-11** (auditor appointment) — if not already complete  
2. **EVOL Protocol 03 B6** → `B6_ACCEPTANCE_PUBLICATION_SHA`  
3. **EVOL W7-05..06** → `FORMAL_CLOSURE_PUBLICATION_SHA` + handoff pin  
4. **Operator OD-ENT-00** (EVOL dependency for ENT)  
5. **ENT Protocol 03 B6** → ENT `B6_ACCEPTANCE_PUBLICATION_SHA`  
6. **OD-ENT-01** → **OD-ENT-02** (admin) → normal FREEZE → **OD-ENT-08** + FRZ-AUD  

**Parallel allowed:** ENT B6 **technical** dossier review may proceed while EVOL B6 pending; **OD-ENT-00** and formal ENT closure remain gated on EVOL reconciliation per canonical blockers.
