# OD-SFX-15 — Pre-freeze execution sequencing

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-15 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **RECORDED — SIGNATURE PENDING** (decision given by the operator in the planning chat, 2026-11-12) |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

When does the SFX campaign start, and what is authorized before ARCH-FREEZE?

## Decision (English rendering of the operator's chat)

Start the SCENARIO-FACTORY-X campaign immediately, in parallel with MCP-SERVER-X, PDC-X and ARCH-FREEZE. Authorized before ARCH-FREEZE: SFX-P0 roadmap and governance registration; SFX-P1 research and competitive analysis; SFX-P2 capability audit, gap analysis, proposed architecture and ADRs; SFX-P3 UX design, mock-ups and contract designs; SFX-P4 only an allowed, proposal-only IDE toolkit without changes to protected platform mechanisms unless separately authorized. Execute batch-first, with separate worktrees, independent reviews and canonical DevOS. STOP before SFX-P5 and any product implementation. SFX-P5..P13 may start only after formal ARCH-FREEZE = CLOSED, acceptance of the required contracts and architecture decisions, and GATE-SFX-ENTRY. SEF-X and SCENARIO-READINESS-X remain independent, parallel stages; SFX neither blocks nor replaces them. Do not prematurely change frozen contracts, start a production runtime, or bypass SCENARIO-GATE / PROD-Q. Pre-freeze goal: bring SFX-P0..P4 to PRE_IMPLEMENTATION_READY, with a complete, verified product plan and exactly specified gates remaining before implementation starts. Work on a separate branch or worktree; this session is the orchestration session; tasks run through subagents attached to it, as parallel as possible, with a board monitoring progress.

## Operator line

_Pending. To be written by the operator, or by a dedicated signing session the operator commissions (IIAC Amendment 01). The orchestrator, its implementer subagents and reviewers never write this line._
