# T-IGX-035-G — MCP-SERVER-X P6: public API, reference server, migration, DX docs

## Scope

Batch **MCP-P6** only (MCP-P6-01..04): consolidated developer public API, P6 reference qualification server (L3), eight-application compatibility/migration matrix, developer and migration documentation, deterministic P6 acceptance suite.

**Parent:** [`T-IGX-035-F.md`](T-IGX-035-F.md) — P5 closed (scoped) @ `c59892829`; P6 entry **OD-MCP-P5-P6-01** @ 2026-11-11T16:24:00+01:00.

**Baseline SHA:** `e48106740a4a243204cadccb6a85db9a8710d6af` (`origin/development` @ P6 cut)

**Tasks:** MCP-P6-01, MCP-P6-02, MCP-P6-03, MCP-P6-04; P6 qualification batch (requirement-to-test mapping); G6 handoff package (separate session).

## Plan digest

Plan digest: 5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-035-A.md`](plans/PLAN_T-IGX-035-A.md) — WBS § MCP-P6-01..04.

## Approved

Approved: operator @ 2026-11-11T16:53:00+01:00

Operator directive (chat 2026-11-11): approve MCP-SERVER-X P6-01..04 implementation and qualification scope per canonical plan digest `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7`.

## Test contract

```text
.venv/Scripts/python.exe -m pytest tests/qualification/mcp_server_x/test_mcp_server_x_p6_qualification_batch.py -q
```

Expected: exit **0** (consolidated P6 batch).

## P6 acceptance criteria (pre-implementation)

| # | Criterion | Verification |
| --- | --- | --- |
| AC-P6-01 | Typed `McpServerDefinition` and validators are the supported contract surface (`intergrax.contracts.mcp_server_definition` + `mcp_developer_api`). | L1 import + contract tests |
| AC-P6-02 | Tier-3 facades compose via `compose_application_mcp_server` / public API only — no catalog bridge or engine private imports. | Import policy test + eight-app matrix |
| AC-P6-03 | Reference qualification server under `tests/qualification/mcp_server_x/reference_app/` exercises MCP-ACC-01 scenarios on **streamable HTTP** L3. | Dedicated L3 pytest |
| AC-P6-04 | Migration matrix documents every application: legacy → canonical → change → test → result. | SSOT markdown + regression |
| AC-P6-05 | Developer guide matches executable behavior (@ implementation SHA). | Doc review + example tests |
| AC-P6-06 | P5/P4 regression subset green in P6 batch. | Selection batch |
| AC-P6-07 | `LIM-RESEARCH-STRICT-FACTORY-MEMORY` preserved as cross-stage PROD-Q dependency (no fabricated vendor qual). | Research test contract unchanged |
| AC-P6-08 | Independent G6 audit **before** ticket `closed` (no self-certification; F-MCP-P5-GOV-001 sequencing). | Separate audit session |

## Implementation contract

- Invariant: single MCP composition authority (`compose_mcp_server_from_definition`); no second registry or execution path
- Reference server is test qualification asset only — not a ninth product application
- Resources/Prompts remain UNSUPPORTED_BY_DESIGN v1
- No FRZ promotion; no MCP-SERVER-X stage CLOSED; no P7/P8 certification claims

## Out of scope

P7 MCP-ACC exact-SHA package @ publication, P8 IIIC stage close, FRZ PASS, stdio/SSE transport, operator IIAC override.

## Lifecycle state

**closed** — scoped P6 acceptance (`OD-MCP-P6-TICKET-01` @ `2026-11-12T07:35:55+01:00`); **F-MCP-P6-GOV-001, F-MCP-P6-GOV-002 and F-MCP-P6-GOV-003 preserved** on record. MCP-SERVER-X stage **not CLOSED**; no FRZ promotion; P7 entry per `OD-MCP-P6-P7-01` (effective only after P6 terminal verification is published).

## Governance note (preserved)

PR #459 merged @ `a695c9c83` **before** independent G6 publication, before the ACT-N11 rebind and while `governance-fast` was red (F-MCP-P6-GOV-001; ticket stale at merge, F-MCP-P6-GOV-002). Recovery PR #460 merged @ `8ed2bdcee` with `governance-fast` failing and before the activation record was rebound (F-MCP-P6-GOV-003). Operator `OD-MCP-P6-GOV-01` (option A) and `OD-MCP-P6-GOV-02` (option A) do **not** retroactively approve either merge. See [`GOVERNANCE_RECONCILIATION.md`](../../docs/project/maintainers/qualification/iiac_evidence/MCP-SERVER-X/p6_g6_closeout/GOVERNANCE_RECONCILIATION.md) and [`MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md`](../../docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md).

## Checkpoint

| Field | Value |
| --- | --- |
| Branch (historical) | `mcp-server-x-p6` (from `e48106740`) → merged PR #459 |
| P6 entry OD | OD-MCP-P5-P6-01 APPROVED @ 2026-11-11T16:24:00+01:00 |
| PARENT publication | P5 `c59892829` |
| P6_BASELINE_SHA | `e48106740a4a243204cadccb6a85db9a8710d6af` |
| P6_IMPLEMENTATION_SHA | `5ff8606b089828b334ebac133518ba47c9f6f7ce` |
| P6_MERGE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` (PR #459 product merge; formerly labelled `P6_PUBLICATION_SHA`) |
| P6_RECOVERY_EVIDENCE_PR460_MERGE_SHA | `8ed2bdcee6d244c128f3c5d95be3d7817418433b` (PR #460) |
| P6_TEST_EXECUTION_SHA | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| P6_G6_AUDIT_SHA | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| P6_IIIC_AUDIT_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| P6_EVIDENCE_PUBLICATION_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| P6_IIIC_CERTIFICATION_PUBLICATION_SHA | `276c64305799fa4d7f1a25018cc1cccf061fbafd` |
| P6_ACTIVATION_SHA | `92ad3555775649742299826e3680ec8ce703cc8a` |
| P6_ACTIVATION_PHASE2_SHA | `19c617365cda85d88b474575e77b7d083739b529` |
| P6_PUBLICATION_SHA | `66dcd1bfe260b33a41881c7b9911b2a2abb752f9` (activation PR merge commit) |
| P6_FORMAL_CLOSURE_SHA | `pending (pinned after merge)` |
| P6_TERMINAL_CLOSURE_SHA | `pending (pinned after merge)` |
| LOCAL_TEST_EXECUTION | P6 batch **PASS** @ `5ff8606b0` |
| `igx_gov ticket` | **PASS** @ `5ff8606b0` (re-run on the closure commit per W6) |
| CI | `governance-fast` was **FAIL** on #459/#460 (ACT-N11); required green on activation PR head and on the closure PR head before merge |
| G6 | **`ACCEPT_SCOPED`** — dossier + `p6_g6_closeout/` @ `2d3eceb6c` (post-merge) |
| ACT-N11 IIIC | **`PASS_SCOPED`** published; certifier preflight **PASS**; activation record two-phase `92ad3555775649742299826e3680ec8ce703cc8a` / `19c617365cda85d88b474575e77b7d083739b529` |
| Operator ODs | IIAC activation OD AUTHORIZED @ 2026-11-12T06:35:45+01:00; OD-MCP-P6-GOV-01 APPROVED option A @ 2026-11-12T06:35:45+01:00; OD-MCP-P6-GOV-02 APPROVED option A @ 2026-11-12T06:46:49+01:00; OD-MCP-P6-TICKET-01 @ `2026-11-12T07:35:55+01:00`; OD-MCP-P6-P7-01 @ `2026-11-12T07:35:55+01:00` |
| Findings | F-MCP-P6-GOV-001 (CRITICAL), F-MCP-P6-GOV-002 (HIGH), F-MCP-P6-GOV-003 (CRITICAL) — preserved, not waived |
| Residuals | OPEN-BP-01 (no required status checks on `development`); `LIM-RESEARCH-STRICT-FACTORY-MEMORY` OPEN non-blocking; roadmap §MCP-SERVER-X description row stale ("Engine not started") — P8 roadmap close |
| P6 terminal verification | pending (W7); P7 `READY` only after it is published |
| Ticket `closed` | **scoped** — AC-P6-01..08 per `OD-MCP-P6-TICKET-01` |
