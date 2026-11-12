# OD-SFX-16 — Execution arrangements: adapters, signing route, independent sub-sessions

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-16 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **RECORDED — SIGNATURE PENDING** (decision given by the operator in the planning chat, 2026-11-12) |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

Three questions asked by the orchestrator on 2026-11-12: (1) how the SFX tickets' Approved lines are signed; (2) whether the regenerated `.claude/skills` and `.cursor/` adapters may ship before ARCH-FREEZE as part of the agent-env delivery (C-OD-01); (3) who performs the generalization-set authoring and the PR reviews.

## Decision (English rendering of the operator's chat)

(1) Option chosen: "A delegate in a separate session" — see DELEGATION-SFX-PRE-FREEZE. (2) Option chosen: "Yes, with agent-env"; `scripts/governance/sync_agent_env.py` and `governance/agent-control-plane/role_registry.yaml` stay untouched until W3. (3) "Run them as sub-sessions of the current session, so these sub-sessions can report progress to the board."

## Operator line

_Pending. To be written by the operator, or by a dedicated signing session the operator commissions (IIAC Amendment 01). The orchestrator, its implementer subagents and reviewers never write this line._
