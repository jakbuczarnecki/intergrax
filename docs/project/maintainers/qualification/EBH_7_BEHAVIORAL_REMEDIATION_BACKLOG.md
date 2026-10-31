# EBH-7 — Behavioral remediation backlog

**Status:** Semantic fail-closed audit restored @ disposition SSOT.  
**Semantic coverage:** TEN **18/26 EXACT**, **8 OPEN**; XDM **16/34 EXACT**, **18 OPEN**.  
**Mechanical pytest:** `tests/qualification/ebh_7/` green with **2× B6 readiness XFAIL** (EXACT 60/60, RPL semantic).

## Priority remediations

| ID | Verdict | Corrective action |
| --- | --- | --- |
| EBH7-TEN-E2E-04 | BLOCKED | Exec→Tool deny on `ToolRuntime.invoke` — ADR or implementation |
| EBH7-TEN-E2E-03,05,12–14,17–19,21–23 | PARTIAL | Replace platform proof delegations with boundary-faithful E2E |
| EBH7-XDM-007,012,014–019,022–031 | PARTIAL / STRUCTURAL | Cross-domain integrated proofs (not single-program reuse) |
| RPL Package C (51 concerns) | PARTIAL semantic | Alternate backend switch or FIXED substrate ADR |

**Stage CLOSED blockers (operator/external):** OD-EBH7-01, Protocol 03 **B6**, OD-EBH7-08, roadmap §3.
