# MCP-SERVER-X P6 — independent G6 audit dossier (post-publication)

**Role:** `independent-quality-auditor` (organizational separation from implementer)  
**Ticket:** `governance/tickets/T-IGX-035-G.md`  
**PR:** #459 (merged)  
**Protocol:** `docs/audit_results/AUDIT_PROTOCOL.md` v2.2 (scoped stage audit)  
**Auditor session id:** `77e4a696-c2e5-4a47-94dc-58aff28ccb8d` (Cursor independent G6 session 2026-11-11)  
**Date (UTC):** 2026-11-11

## PIN block

| Field | SHA / value |
| --- | --- |
| P6_BASELINE_SHA | `e48106740a4a243204cadccb6a85db9a8710d6af` |
| P6_IMPLEMENTATION_SHA | `5ff8606b089828b334ebac133518ba47c9f6f7ce` |
| PR HEAD @ audit | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| MERGE_SHA / P6_PUBLICATION_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` |
| AUDIT_EXECUTION_SHA (replay) | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| G6_EVIDENCE_PUBLICATION_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| Plan digest | `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7` |

## GOVERNANCE_RESULT

| Check | Result | Evidence |
| --- | --- | --- |
| Merge timestamp | `2026-11-11T17:08:34Z` | `gh pr view 459` |
| Pre-merge independent G6 on record | **NO** | Ticket AC-P6-08; handoff § sequencing |
| Pre-merge `governance-fast` | **FAIL** (ACT-N11) | PR checks @ `9b8c19d75` |
| Pre-merge IIIC ACT-N11 rebind | **NO** | `IIAC_POLICY_ACTIVATION_RECORD.json` pin `8fff85e3b` |
| `product_paths` between impl and PR HEAD | **docs only** | `5ff8606b0..9b8c19d7` empty product Δ |

**Finding F-MCP-P6-GOV-001 (CRITICAL, governance):** PR #459 merged **before** independent G6 evidence publication, IIIC ACT-N11 P6 rebind, and mandatory `governance-fast` green — same class as F-MCP-P5-GOV-001 (prospective corrective, no retroactive approval).

**Finding F-MCP-P6-GOV-002 (HIGH):** Ticket remained `quality_gate` with G6/ACT-N11 pending at merge.

**Policy resolution:** Post-publication independent replay @ `9b8c19d75` satisfies **scoped P6 technical acceptance** (this dossier). Operator disposition + IIIC rebind remain mandatory before ticket `closed`.

## Scoped technical acceptance

| AC | Result |
| --- | --- |
| AC-P6-01..07 | **PASS** @ `AUDIT_EXECUTION_SHA` |
| AC-P6-08 | **PASS** (this dossier) |

Replay: `pytest tests/qualification/mcp_server_x/test_mcp_server_x_p6_qualification_batch.py` exit **0** (~225s auditor replay).

## INDEPENDENT_AUDIT_RESULT

**`ACCEPT_SCOPED`**

Scoped P6 product criteria (MCP-P6-01..04) satisfied @ audit tree. Merge governance violations recorded; ACT-N11 recovery tracked separately under `act_n11_mcp_p6/`.

## Operator boundary

Auditor did **not:** edit activation record, sign operator OD, merge, promote FRZ, or close MCP-SERVER-X stage.
