# ENT-AUDIT-X — ARCH-FREEZE handoff (W6, post–formal closure)

**Status:** **HANDOFF READY** — ENT-AUDIT-X formal stage **CLOSED (IIAC PASS_SCOPED)**; **ARCH-FREEZE not CLOSED**; **not** production/live qualification closure.

## Formal ENT disposition

| Field | Value |
|-------|-------|
| Roadmap headline | `ENT-AUDIT-X — CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` |
| Verdict chain | IIAC **PASS_SCOPED** (W3 audit + W4 certifier) |
| Operator | **OD-ENT-08** @ `2026-11-11T07:20:00Z` |
| Assurance | `process_enforced` |

## Exact SHA pins (ENT)

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `32a9633440007eb4e8f00a825bbf2e399192b346` |
| EVIDENCE_PUBLICATION_SHA | `481b2cc2c10a407fd2b93cf9c69714e8894aeccf` |
| B6_ACCEPTANCE_PUBLICATION_SHA | `add6a6ce2015ebaa4bc38b7fadf3936f100a1522` |
| ACCEPTED_BASELINE_SHA | `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` |
| W3 independent audit pub | `fd5728955605ef2ec013d69ec9290736f0970419` |
| W4 certification pub | `5f178174e2462b038d544b61192a666d5b0ca79f` |
| W2 normal FREEZE evidence pub | `bc4b5f61666a8fea638064f50809fff2089b1b8b` |
| FORMAL_CLOSURE_PUBLICATION_SHA | see `ENT_AUDIT_X_SHA_LADDER.md` |
| Audit execution | `dc778d125fd7e7205edd30fbbf144ca12320b0ee` |

## IIAC evidence paths

- Manifest: `iiac_evidence/ENT-AUDIT-X/certification/iiac_audit_session_manifest.ENT-AUDIT-X.json`
- Pass record: `ent_audit_x/ENT_AUDIT_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md`
- W3 tree: `iiac_evidence/ENT-AUDIT-X/audit/w3_dc778d/`
- W4 tree: `iiac_evidence/ENT-AUDIT-X/certification/w4_dc778d/`
- Operator decisions: OD-ENT-00, OD-ENT-01, OD-ENT-02, **OD-ENT-08**, O-1 delegation under `iiac_evidence/ENT-AUDIT-X/operator_decisions/`

## FRZ promotions (ENT scope)

| Family | Status |
|--------|--------|
| FRZ-AUD-01 … 09 | PASS |
| FRZ-AUD-10 | PASS — COMPENSATING (SCOPED) |
| FRZ-FRZ-11 | PASS — SCOPED (ENT independently CLOSED) |

## Enforcement route (OD-ENT-02)

- **Mode:** `compensating` (`od_ent_02_mode=compensating`)
- **Controls:** `OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md`, verifier `verify_od_ent_02_compensating_control.py`
- **OPEN-BP-01:** live GitHub branch protection on `development` **not** enabled — owner **repo admin / ARCH-FREEZE**; trigger: required ENT checks live on `development`

## Scoped limitations (do not waive at ARCH-FREEZE entry)

1. **OPEN-BP-01** — compensating ≠ LIVE protection  
2. **F-W3-001** — FREEZE B5 directory naming  
3. **Path B / FRZ-PRD-*** — live qualification slice **OPEN** (PROD-Q)  
4. No cryptographic independence claim — **process_enforced** only  

## EVOL predecessor pins

| Role | SHA |
|------|-----|
| EVOL formal closure | `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca` |
| EVOL → ENT handoff | `7a4f279fd8c0986431941c77283d872ea5b0c1ba` |
| EVOL IIAC B6 | `1dd769331506149a1ee2f411c2bc62e85e5bbf25` |

## Explicit non-claims

- ENT closure **does not** close ARCH-FREEZE  
- **Does not** claim global FRZ PASS or PROD-Q live readiness  
- **Does not** claim LIVE branch protection  
- FRZ-EVOL-07 qualification-layer limitation (from EVOL handoff) unchanged  

## MCP-SERVER-X gate (2026-11-11)

Formal **ARCH-FREEZE** stage closure now requires **`MCP-SERVER-X` = CLOSED** (IIAC) between ENT and ARCH-FREEZE per roadmap §3. ENT CLOSED remains valid; **Mode A** change-impact for the future MCP public contract is recorded in [`mcp_server_x/MCP_SERVER_X_INDEX.md`](../mcp_server_x/MCP_SERVER_X_INDEX.md) — ENT accepted baseline `480550d80b2515fc8cbd2a0ba6f79d1ee1c44541` is **not** invalidated by P0 documentation.

## ARCH-FREEZE entry (operator/architect)

Prerequisites now satisfied for **ENT-AUDIT-X independently CLOSED** (FRZ-FRZ-11). **MCP-SERVER-X** and then **PDC-X** (roadmap §3.2) must close before ARCH-FREEZE stage closure. Remaining ARCH-FREEZE obligations: full freeze checklist, T-B7-* WBS, independent admissions per plan — **execute only under ARCH-FREEZE campaign**, not in this handoff.

**Terminal report:** [`ENT_AUDIT_X_FORMAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md`](./ENT_AUDIT_X_FORMAL_CLOSURE_CAMPAIGN_TERMINAL_REPORT.md)
