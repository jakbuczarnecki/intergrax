# OD-MCP-P8-CLOSE-01 — conditional formal closure of MCP-SERVER-X (scoped)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P8-CLOSE-01 |
| **Status** | **REQUESTED** (unsigned; conditional) |
| **Tickets** | `governance/tickets/T-IGX-035-H.md`, `governance/tickets/T-IGX-035-I.md` |
| **Policy version** | IIAC-2026-11-05.1 |

## Question

Does the operator authorize the formal scoped closure of MCP-SERVER-X, **taking effect only if every condition below is mechanically true and attested by an independent terminal verification session**? If any condition is false, this decision is void and the orchestrator stops.

## Conditions (all must hold at the time of use)

1. **C1** MCP-ACC-01..20 are green on one `TEST_EXECUTION_SHA` (or have a justified N/A) in the exact-SHA package.
2. **C2** The independent audit verdict is `PASS` or `PASS_SCOPED`; open mandatory blockers = 0; every finding has a disposition.
3. **C3** The IIAC certification manifest (`iiac-v1`, `IIAC-2026-11-05.1`) passes `verify_iiac_audit_manifest.py` and `verify_iiac_role_separation.py --strict-product-diff`; implementer, auditor and certifier are three different sessions.
4. **C4** The FRZ counter is 225, unchanged; no FRZ PASS promotion; the FRZ validator is green.
5. **C5** The evidence (`EVIDENCE_PUBLICATION_SHA`) is published on `origin/development` and the required CI of the publication merges has finished green.
6. **C6** `verify_iiac_policy_activation --require-in-force` exits 0 at the head, and no ACT-N11 product path changed after the audit pin (or a separate signed activation decision covers it).
7. **C7** PROD-Q Path B is untouched; `REQ-MCP-OBS-01`, `LIM-RESEARCH-STRICT-FACTORY-MEMORY` and OPEN-BP-01 are recorded as limitations in the final report.
8. **C8** Findings F-MCP-P5-GOV-001, F-MCP-P6-GOV-001/002/003 and every P7/P8 process deviation remain on record, not waived.
9. **C9** `MCP_SERVER_X_ARCH_FREEZE_HANDOFF.md` is published.
10. **C10** An independent terminal verification session (not the implementer, auditor or certifier) has published a report with the verdict `P8_TERMINAL_VERIFIED` that checks C1–C9 against git and `gh`.

## What this decision authorizes (only when C1–C10 hold)

The roadmap status of MCP-SERVER-X `CLOSED — INDEPENDENTLY AUDITED (INTERNAL AGENT CERTIFICATION) — PUBLISHED` with the scoped limitations stated in the text, the `FORMAL_CLOSURE_SHA` pin, the final closure report and the ARCH-FREEZE handoff. Limitations remain in force; the label is scoped, not an unscoped PASS.

## Operator line (required; not written by an agent)

```text
Approved: OD-MCP-P8-CLOSE-01 @ ________________
```

**Explicit non-actions:** does not close ARCH-FREEZE, does not promote any FRZ criterion, does not claim production readiness, does not touch PROD-Q Path B, does not waive any finding, and is void if any condition fails.
