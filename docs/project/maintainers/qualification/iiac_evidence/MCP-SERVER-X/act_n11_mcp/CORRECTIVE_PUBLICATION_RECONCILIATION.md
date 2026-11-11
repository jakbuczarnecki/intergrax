# F-MCP-ACT-N11-001 — corrective publication reconciliation (PR #454)

| Field | Value |
|---|---|
| Ticket | `governance/tickets/T-IGX-035-E.md` |
| Prior PR | [#454](https://github.com/jakbuczarnecki/intergrax/pull/454) — **MERGED** (do not re-merge) |
| PR #454 merge commit | `7562a0692d7f33ce4423baf76988ced008e870fa` @ `2026-11-11T11:36:52Z` |
| PR #454 branch tip @ merge | `7c401fd98cf2b42b9b2fd5860485bac34a8c2df3` (ACT-N11 governance evidence only) |
| `origin/development` HEAD | `7562a0692d7f33ce4423baf76988ced008e870fa` |
| Corrective branch | `mcp-server-x-p4` |
| **`corrective_candidate_sha`** | `0320df94f00458b724445902051d7b51b610f57b` (publication HEAD; product fix @ `27e19f1f`) |
| Corrective PR | [#455](https://github.com/jakbuczarnecki/intergrax/pull/455) |

## What merged vs what did not

PR #454 delivered P4 product scope through `7c401fd98` (P4 facades + ACT-N11 recovery **governance** artifacts). That merge **did not** include the product fix for **F-MCP-ACT-N11-001** (`host_custom` FastMCP registration).

The fix landed **after** merge on `mcp-server-x-p4` as commits:

| SHA | Summary |
|-----|---------|
| `e03092849ce028db275fc6fd4e472babf6421fa4` | `fix(mcp): register host_custom tools with explicit FastMCP signatures` |
| `76ed01ff34d59f7e7ee3ba64e875b53bc2ff432b` | evidence pin (superseded by HEAD) |
| `27e19f1feb31522c3103e6fce6f15a8c9395f380` | regression: missing required MCP args |

Verification: `git merge-base --is-ancestor 27e19f1 origin/development` → **false**; `git show origin/development:intergrax/applications/_shared/mcp_server_engine.py` has **no** `_bind_custom_host_tool_for_fastmcp`.

## Governance context @ PR #454 close

Last CI on the pre-merge branch @ `7c401fd98` included **Regression gate FAILURE** (run `43012460938`, 2026-11-11T11:30:14Z). The **`governance-fast` ACT-N11** failure class persisted (expected pre-activation). Merge completed while independent IIIC **PARTIAL_EXACT_BLOCKERS** and ACT-N11 gate remained red — corrective path is a **follow-on PR**, not a re-open of #454.

## Corrective publication intent

1. New PR: **development** ← `mcp-server-x-p4` @ `corrective_candidate_sha` (three commits only vs `7562a069`).
2. Scope: `F-MCP-ACT-N11-001` product fix + focused tests + qualification handoff updates.
3. **No** IIAC activation record edit; **no** operator signature; **no** FRZ promotion.
4. **No** self-PASS — independent IIIC re-audit @ `corrective_candidate_sha` required before Operator activation.

## Residual (unchanged by corrective)

**`LIM-RESEARCH-MOUNT-MEMORY-ADMISSION`:** `test_research_application_exposes_mcp_mount` → `MemoryProviderAdmissionError` (`qualification_evidence_missing`). Ticket contract **28 pass / 1 fail** on corrective HEAD.
