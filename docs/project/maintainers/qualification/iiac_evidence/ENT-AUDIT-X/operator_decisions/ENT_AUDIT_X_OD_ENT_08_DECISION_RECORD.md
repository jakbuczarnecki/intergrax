# ENT-AUDIT-X OD-ENT-08 — formal stage closure authorisation

**Decision ID:** OD-ENT-08  
**Stage:** ENT-AUDIT-X formal closure (W5/W6)  
**Policy version:** IIAC-2026-11-05.1  
**Record origin:** operator-delegate (signing session `ent-sign-w5-od-08-2026-11-11`)

## Operator disposition

Authorise **W5/W6 formal closure publication only** under **COMPENSATING (SCOPED)** disposition. **Not** ARCH-FREEZE closure. **Not** a waiver of OPEN-BP-01, F-W3-001, Path B / FRZ-PRD OPEN, or `process_enforced` assurance limits.

**Authoritative development HEAD @ authorisation:** `8fdbe7c037bb545f7cbfd7587589ac7f9ddd4d5a`

## Accepted chain (PASS_SCOPED — no upgrade to unscoped PASS)

| Item | Disposition |
|------|-------------|
| W3 independent audit | **PASS_SCOPED** @ `fd5728955605ef2ec013d69ec9290736f0970419` |
| W4 certification | **PASS_SCOPED** @ `5f178174e2462b038d544b61192a666d5b0ca79f` |
| OD-ENT-02 | **SUFFICIENT_COMPENSATING_SCOPED** · `od_ent_02_mode=compensating` |
| FRZ-AUD-01 … 09 | **PASS** |
| FRZ-AUD-10 | **PASS — COMPENSATING (SCOPED)** |
| Matrix C | **75/75 QUALIFIED** |
| Population E | **PASS_SCOPED** |

## Scoped limitations (explicit acceptance)

1. **OPEN-BP-01** — live branch protection on `development` not enabled.  
2. **F-W3-001** — FREEZE B5 directory naming strictness.  
3. **Path B / FRZ-PRD** — live qualification slice **OPEN**.  
4. **`process_enforced`** — procedural independence model only.

## Mechanical closure target (roadmap Gate B)

IIAC preflight headline (exact): `ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED`

Operator disposition label (SCOPED): `ENT-AUDIT-X — CLOSED — INDEPENDENTLY ACCEPTED (SCOPED) — PUBLISHED` — limitations above remain in force.

## Signature (S-2)

Approved OD-ENT-08: operator-delegate @ 2026-11-11T07:20:00Z  
Scope: ENT-AUDIT-X formal stage closure under IIAC PASS_SCOPED chain @ development HEAD 8fdbe7c037bb545f7cbfd7587589ac7f9ddd4d5a  
Authorises: W5 closure phase 1/2, roadmap §5 headline, `FORMAL_CLOSURE_PUBLICATION_SHA` pin, Gate B verification, ARCH-FREEZE handoff (W6) — not ARCH-FREEZE CLOSED

Delegation: docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/operator_decisions/ENT_AUDIT_X_O1_SIGNING_DELEGATION_RECORD.md@fb5c7801eea73f0930160af9ec51b5ade28875a3

## Assurance

`process_enforced` — operator chat commission 2026-11-11 transcribed per IIAC Amendment 01; no cryptographic proof of operator identity.
