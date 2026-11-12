# OD-MCP-RC-01 — decisions for the MCP-SERVER-X residual remediation campaign (MCP-RC)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-RC-01 |
| **Status** | **RECORDED (operator chat answers; not signed `Authorized:` lines)** |
| **Related** | R-P8-01, R-P7-01..06, F-P8-07/08, `OD-MCP-P8-RESIDUAL-01`, `OD-MCP-P7-P8-AUTH-01`, `OD-PDC-SEQ-01`, `OD-PDC-IIAC-SCOPE` |
| **Policy version** | IIAC-2026-11-05.1 |
| **Plan** | `agent-env/plans/pasted-content-id-703a-integrax-prancy-fiddle.md` (plan mode output, 2026-11-12) |

## Operator record

The operator answered the decision pack D-1..D-8 of the plan one question at a time in chat on 2026-11-12, selecting the recommended option each time. The questions and the selected answers are listed below as English renderings written by the orchestrator; the Polish option labels are the operator's selections. No agent-composed timestamp and no signature line.

| ID | Question (summary) | Selected answer |
| --- | --- | --- |
| D-1 | Path for the MCP-RC product changes | **A** — one PDC-P7-00 window: all MRC batches ride in the PDC-X integration window with one independent delta audit and one two-phase rebind at `INTEGRATED_BASELINE_SHA`. Extends `OD-MCP-P8-RESIDUAL-01` from R-P8-01 to R-P7-01..06 and the hardening findings. |
| D-1b | Merge target for product PRs | Branch `mcp-rc`; merged to `development` only together with the rebind, after audit and certification, with the five required CI contexts green on the pinned head. |
| D-1c | Re-grant of the authorizations that lapsed with PR #480 (`OD-MCP-P7-P8-AUTH-01`) | **Yes, for the duration of the MCP-RC campaign**: branches, worktrees, independent sessions, PRs, merges only after the five green contexts. No ruleset, FRZ or P8-history change. |
| D-2 | Bounded diagnostic pytest under the C10/C15 freeze | **Yes**: the 3 R-P7-02 files (29 tests) plus the mandatory MCP contract (164 + 14 + 141), one pytest process at a time, Windows environment unchanged, logs under `.tmp/session/mcp-rc/`. No Docker, no new vendors, no production configuration. |
| D-3 | MCP-RC auditor lane in `OD-PDC-IIAC-SCOPE`; signing delegates | **Yes**; delegates as before (new dedicated non-implementing sessions on the `DELEGATION-PDC-WAVE-*` pattern, Amendment 01 R-1..R-6). Evidence under `iiac_evidence/PDC-X/MCP-RC/`. |
| D-4 | Breaking-change classification | **Versioned**: MCP contract v1 → v1.1 (security hardening) with ADR-MCP-RC-01, a migration note per facade and `LOCAL_DEV_EXPLICIT` as the dev path. |
| D-5 | R-P7-01 trace correlation | **Implement** `mcp_invocation_id` (additive correlation key, not an execution authority). |
| D-6 | Advertised-but-empty capabilities (logging, prompts, resources, UI) | **Documentation only** plus a snapshot test; no protocol behaviour change. |
| D-7 | ARCH-FREEZE blocking set | Classification of the plan section C approved: mandatory = G-ID, G-ERR-03, G-RB, G-DL, G-TQ; G-TR-01 recommended and non-blocking. FRZ counter stays 225; no promotion. |
| D-8 | Ticket series and WBS/DAG amendment | Series `T-IGX-038-A..F` (037 is taken by SFX) and an amendment registering MCP-RC as an additional writer of the `intergrax/applications/_shared/` hotspot in an explicit wave. |

D-9 (PDC-BL-F01 protocol header wording) stays with PDC-P7-00; D-10 (why PRs #475, #476, #478 were closed) is informational and open.

## Explicit non-actions

This record does not edit any P8 record, any IIAC activation record, any FRZ status or any GitHub ruleset, and it does not approve any ticket plan. Ticket `Approved:` lines and the IIAC activation line are written only by the operator or by a commissioned delegate session, never by the implementer. OPEN-BP-01 stays deferred (`OD-MCP-P8-RESIDUAL-01`). PROD-Q, MEM and production readiness are untouched.
