# OD-MCP-RC-02 — the MCP server-side call bound is a transport observer trigger, not a deadline authority

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-RC-02 |
| **Status** | **AUTHORIZED (operator chat answer, recorded as an `Authorized:` line at the operator's explicit request on 2026-11-13; see the end of this file)** |
| **Related** | R-P7-04, `OD-MCP-RC-01`, master plan sections D.3 and E (row MRC-3.1), `T-IGX-038-C` |

## Question

The delegate signer held `T-IGX-038-C` because the plan wording had been changed after its first review: the master plan said a configured `max_call_seconds` "feeds the canonical resolver", while the code shows that the canonical Execution deadline authority (`ExecutionDeadlineAuthorityResolver`, `intergrax/runtime/execution/deadline_authority/resolver.py`, read in `intergrax/runtime/execution/runtime.py`) is composed by the host through the run budget (`max_wall_time_seconds`); MCP can neither set nor override it. How should R-P7-04 be closed?

## Operator record

Operator chat, 2026-11-13, answer selected from two options (Polish option label: "Wyzwalacz transportowy MCP (Recommended)"). English rendering: "MCP transport trigger (recommended): `max_call_seconds` only triggers the existing governed cancel, like a client disconnect; the canonical deadline stays with the host." Recorded here as the operator's instruction; no agent-composed timestamp and no signature line.

## Disposition

1. `max_call_seconds` (optional, default none) is an MCP **transport observer trigger**: on expiry it requests the existing governed cancel (`governed_cancel_active_task`, cooperative cancel owned by `CancellationCoordinator`) exactly as a client disconnect does, applies the existing bounded drain and supervision handoff, and returns `MCP_DEADLINE_EXCEEDED` with an explicit outcome-unknown detail. It is not a deadline authority and not a timer lifecycle.
2. The canonical root Execution deadline stays with the host (`ExecutionDeadlineAuthorityResolver`, run budget). A host that needs a hard run wall time sets that budget; MCP never creates, configures or overrides it.
3. Master plan section E, row MRC-3.1, acceptance is amended accordingly: "server work bounded by the MCP transport trigger and the host-owned canonical run budget; outcome unknown never reported as success".
4. Non-cooperative synchronous work (for example a catalog tool thread) cannot be terminated in-process; this stays a documented, accepted limit with a bounded wait.

## Explicit non-actions

No product change, no signature, no ticket approval. The `Approved:` line of `T-IGX-038-C` is written only by a commissioned delegate session after it re-verifies the plan against this record.

## Authorization record

On 2026-11-13 the operator answered an explicit chat question whether this decision may be recorded as a signed `Authorized:` line quoting their earlier answer. English rendering of the operator's answer (Polish option label: "Tak, zapisz z cytatem (Recommended)"): "Yes, record it with the quotation." The quoted earlier answer is the one in the section "Operator record" above ("Wyzwalacz transportowy MCP (Recommended)"). The line below is written by the orchestrator session on that instruction, not by the operator's own hand; it signs nothing beyond this decision (no ticket approval, no IIAC activation, no product change).

Authorized: operator @ 2026-11-13T14:09:10+01:00
