# MCP-SERVER-X P6 — terminal dossier (final, scoped)

**Ticket:** `T-IGX-035-G` — **`closed` (scoped)**  
**Plan digest:** `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7`  
**Disposition:** **P6 CLOSED (scoped)** — G6 `ACCEPT_SCOPED`, IIIC ACT-N11 `PASS_SCOPED`, operator `OD-MCP-P6-TICKET-01` @ `2026-11-12T07:35:55+01:00`; F-MCP-P6-GOV-001/002/003 preserved. **MCP-SERVER-X stage not CLOSED. FRZ not promoted.** P7 `READY` only after the independent P6 terminal verification is published (`OD-MCP-P6-P7-01`).

## SHA ladder (this batch; canonical copy in [`MCP_SERVER_X_SHA_LADDER.md`](./MCP_SERVER_X_SHA_LADDER.md))

| Role | Value |
| --- | --- |
| P6_BASELINE_SHA | `e48106740a4a243204cadccb6a85db9a8710d6af` |
| P6_IMPLEMENTATION_SHA | `5ff8606b089828b334ebac133518ba47c9f6f7ce` |
| P6_MERGE_SHA (PR #459 product merge; formerly labelled `P6_PUBLICATION_SHA`) | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` |
| P6_RECOVERY_EVIDENCE_PR460_MERGE_SHA | `8ed2bdcee6d244c128f3c5d95be3d7817418433b` |
| P6_TEST_EXECUTION_SHA / P6_G6_AUDIT_SHA | `9b8c19d75d5df0f4b7c87d2707811f3715805102` |
| P6_IIIC_AUDIT_SHA / P6_EVIDENCE_PUBLICATION_SHA | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| P6_IIIC_CERTIFICATION_PUBLICATION_SHA | `276c64305799fa4d7f1a25018cc1cccf061fbafd` |
| P6_ACTIVATION_SHA (phase 1) | `92ad3555775649742299826e3680ec8ce703cc8a` |
| P6_ACTIVATION_PHASE2_SHA | `19c617365cda85d88b474575e77b7d083739b529` |
| P6_PUBLICATION_SHA (activation PR merge) | `66dcd1bfe260b33a41881c7b9911b2a2abb752f9` |
| P6_FORMAL_CLOSURE_SHA | `pending (pinned after merge)` |
| P6_TERMINAL_CLOSURE_SHA (closure PR merge) | `pending (pinned after merge)` |
| FINAL_REMOTE_HEAD | `pending (pinned after terminal verification)` |

## Gates

| Gate | Status |
| --- | --- |
| G6 independent audit | **`ACCEPT_SCOPED`** — dossier [`MCP_SERVER_X_P6_INDEPENDENT_AUDIT_DOSSIER.md`](./MCP_SERVER_X_P6_INDEPENDENT_AUDIT_DOSSIER.md), `p6_g6_closeout/` @ `2d3eceb6c`; operator disposition `ACCEPT_SCOPED_WITH_GOVERNANCE_FINDING` (`OD-MCP-P6-GOV-01` option A) |
| IIIC ACT-N11 P6 | **`PASS_SCOPED`** @ `act_n11_mcp_p6/` |
| Certifier | preflight **`CERTIFIER_PREFLIGHT_PASS`** @ `276c64305`; activation certifier PASS @ `19c617365cda85d88b474575e77b7d083739b529` |
| IIAC `IN_FORCE` rebind | bound to `2d3eceb6c`; `verify_iiac_policy_activation --require-in-force` exit 0 on the activation PR merge `66dcd1bfe260b33a41881c7b9911b2a2abb752f9` and on `origin/development` `dcdb35601305a2ad9a3ee22894b26c29800a9640` (orchestrator re-run 2026-11-12); to be re-run on the closure merge in W7 |
| `governance-fast` and mandatory CI | red on #459 and #460 (preserved, F-MCP-P6-GOV-001/003); green on activation PR #464 head `03b246990c0f359c6f91f026799cd5cf98475eb0` (6 passing, 0 failing) and on the `development` push run at `dcdb35601305a2ad9a3ee22894b26c29800a9640` (`ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `governance-policy-pin` success); closure PR checks and the closure merge are verified in W7 |
| Terminal verification (W7) | pending — `P6_TERMINAL_VERIFIED` report not yet published; see [`MCP_SERVER_X_P7_HANDOFF.md`](./MCP_SERVER_X_P7_HANDOFF.md) |

## Operator decisions

- IIAC activation OD `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` — AUTHORIZED @ 2026-11-12T06:35:45+01:00 (designated signing session).
- `OD-MCP-P6-GOV-01` — option A @ 2026-11-12T06:35:45+01:00 (designated signing session); covers F-MCP-P6-GOV-001/002.
- `OD-MCP-P6-GOV-02` — option A @ 2026-11-12T06:46:49+01:00 (designated signing session); covers F-MCP-P6-GOV-003.
- `OD-MCP-P6-TICKET-01` — `APPROVED` @ `2026-11-12T07:35:55+01:00`.
- `OD-MCP-P6-P7-01` — `APPROVED` @ `2026-11-12T07:35:55+01:00`; effective only at the `origin/development` SHA where the P6 terminal verification PASS is published.
- Details and provenance: [`MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md`](./MCP_SERVER_X_P6_OPERATOR_CLOSURE_RECORD.md).

## Governance (immutable)

- **F-MCP-P6-GOV-001** (CRITICAL): merge before ACT-N11 activation — [`GOVERNANCE_RECONCILIATION.md`](../iiac_evidence/MCP-SERVER-X/p6_g6_closeout/GOVERNANCE_RECONCILIATION.md). PR #459 merged before independent G6 publication, before the ACT-N11 rebind and while `governance-fast` was red.
- **F-MCP-P6-GOV-002** (HIGH): ticket stale at merge of PR #459.
- **F-MCP-P6-GOV-003** (CRITICAL, governance, recorded 2026-11-12): recovery PR #460 was merged as merge commit `8ed2bdcee6d244c128f3c5d95be3d7817418433b` (2026-11-12T05:26:38Z, head `b0b415e92082f306d2ee7de698c5f2e25822f04f`) with `governance-fast` failing and before `IIAC_POLICY_ACTIVATION_RECORD.json` was rebound (still pin `8fff85e3b`). Effect: `origin/development` stayed red until the activation PR #464 landed (merge `66dcd1bfe260b33a41881c7b9911b2a2abb752f9`; push run at `dcdb35601305a2ad9a3ee22894b26c29800a9640` green); the evidence and unsigned drafts are published. No retroactive approval; activation continued on branch `mcp-p6-gov-signing` (base `bc9ef52bd690738e71638a599eec64c76ac356c9`). The signed `OD-MCP-P6-GOV-01` (option A, 2026-11-12T06:35:45+01:00) does not textually cover this finding; it is covered by the separate decision `OD-MCP-P6-GOV-02`.
- No finding is waived, back-dated or retroactively approved.

**MCP-SERVER-X stage:** not CLOSED. **FRZ:** not promoted.

## Recovery verification notes (W1)

- **Pin-only certifier edits:** the diff `2d3eceb6c..276c64305` on auditor-authored files (IIIC dossier, IIIC findings, G6 manifest, G6 dossier) changes only `PENDING` placeholders to the evidence publication SHA `2d3eceb6cf7e93dccd4d96919319a6c969b4444b`. Independence of G6 auditor, IIIC auditor and preflight certifier is a `process_enforced` attestation; the activation certifier is a separate new session.
- **Inventory erratum:** the cumulative ACT-N11 inventory is **17** paths (P4: 16, plus `intergrax/applications/_shared/mcp_developer_api.py`), recomputed from `11ac8ad85..a695c9c83` and equal to `audit_scope.json`. Earlier P6 drafts say "19" / "16 + 3". Auditor files are not edited.
- **Minor:** the G6 dossier states ~225 s replay; the stored log shows 97.57 s; auditor logs carry no command/SHA header (P7 logs must carry command + SHA).
- **Structural:** the `development` ruleset requires no status checks; mandatory CI is policy-only (cross-reference OPEN-BP-01).
- **Activation OD:** the draft `IIIC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` was superseded by the gate-compliant `IIAC_OD_ACTIVATION_MCP_ACT_N11_P6_2026-11-11.md` (now AUTHORIZED, see above).

## Residuals (open, carried forward)

| Residual | Disposition |
| --- | --- |
| `LIM-RESEARCH-STRICT-FACTORY-MEMORY` / F-MCP-P5-MEM-001 | PROD-Q cross-stage, OPEN, non-blocking; no fabricated vendor qualification |
| OPEN-BP-01 | No required status checks on the `development` ruleset (FRZ-AUD-10 compensating); operator settings choice, not changed in P6 |
| Roadmap §MCP-SERVER-X description row (~line 495) | Stale text ("Engine **not started** until OD-MCP-01..07"); only the status-table row was updated in P6; fix at P8 roadmap close |
| REQ-MCP-OBS-01 | Dedicated MCP invocation trace id gap remains (P7) |
| PF-06 | G6 replay-duration note and missing command/SHA headers in auditor logs |

## Non-claims

P6 closed **scoped** to the audited P6 inventory only. MCP-SERVER-X not CLOSED; no FRZ status change; no P7 artifact; no IIIC terminal (P8) claim; MCP-ACC rows remain scoped evidence until the P7 exact-SHA package.
