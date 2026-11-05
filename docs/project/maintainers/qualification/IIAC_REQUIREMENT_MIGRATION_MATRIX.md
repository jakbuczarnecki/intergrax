# IIAC requirement migration matrix

**Policy document:** `IIAC-2026-11-05.1` · **Activation:** `NOT_IN_FORCE` until Session B.

| Old requirement | New IIAC requirement | Preserve substance? |
|-----------------|----------------------|---------------------|
| Human Protocol 03 B6 PASS file | `*_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` + manifest + certifier publication | Yes — Mode A/B/C semantics |
| OD-EVOL-11 human `auditor_id` | `auditor_session_id` + handoff record | Yes — independence |
| Operator OD-ENT-08 stage closure alone | Certifier + `FORMAL_CLOSURE_PUBLICATION_SHA` + preflight | Yes |
| Operator FRZ row mass promotion | Certifier gate + per-row evidence | Yes |
| OD-ENT-02 live branch protection only | LIVE **or** compensating Option B ([`OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md`](./ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md)) | Risk explicit; **OPEN-BP-01** under compensating |
| OD-IC-01 internal cert | Historical evidence only | No closure authority |
| ECAP-P4-EVOL-SCOPE-AUTO-B6 | Superseded by IIAC | Archive pointer only |
| GitHub admin / PROD-Q live | Unchanged external gates | Yes |
| EBH-6 / DUP-X human B6 records | Category 6 historical | No retroactive rewrite |

## FRZ-AUD-10 dual disposition

| Mode | FRZ-AUD-10 status | ENT IIAC CLOSED | ARCH-FREEZE |
|------|-------------------|-----------------|-------------|
| LIVE | PASS with live `gh` proof | Allowed | Preferred |
| COMPENSATING | `PASS — COMPENSATING (SCOPED)` + limitations | Allowed if Session D accepts package | **OPEN-BP-01** until LIVE or explicit gate |

## Matrix C six rows (ENT)

| Row | Old gate | IIAC gate |
|-----|----------|-----------|
| C001 | Operator after OD-ENT-01 | Accepted baseline manifest + auditor replay |
| C002 | Operator trust | IIAC trust-chain step |
| C003 | Operator-only | Auditor replay of trust boundaries |
| C042 | Operator | IIAC manifest-cited evidence |
| C057 | Operator | IIAC manifest-cited evidence |
| C065 | Operator | IIAC manifest-cited evidence |

## G0 cross-reference

Automated inventory: [`IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`](./IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json) — category 1 paths require manual review before activation.
