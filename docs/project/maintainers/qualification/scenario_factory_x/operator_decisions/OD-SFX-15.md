# OD-SFX-15 — Pre-freeze execution sequencing

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-15 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **APPROVED** |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

When does the SFX campaign start, and what is authorized before ARCH-FREEZE?

## Decision (English rendering of the operator's chat)

Start the SCENARIO-FACTORY-X campaign immediately, in parallel with MCP-SERVER-X, PDC-X and ARCH-FREEZE. Authorized before ARCH-FREEZE: SFX-P0 roadmap and governance registration; SFX-P1 research and competitive analysis; SFX-P2 capability audit, gap analysis, proposed architecture and ADRs; SFX-P3 UX design, mock-ups and contract designs; SFX-P4 only an allowed, proposal-only IDE toolkit without changes to protected platform mechanisms unless separately authorized. Execute batch-first, with separate worktrees, independent reviews and canonical DevOS. STOP before SFX-P5 and any product implementation. SFX-P5..P13 may start only after formal ARCH-FREEZE = CLOSED, acceptance of the required contracts and architecture decisions, and GATE-SFX-ENTRY. SEF-X and SCENARIO-READINESS-X remain independent, parallel stages; SFX neither blocks nor replaces them. Do not prematurely change frozen contracts, start a production runtime, or bypass SCENARIO-GATE / PROD-Q. Pre-freeze goal: bring SFX-P0..P4 to PRE_IMPLEMENTATION_READY, with a complete, verified product plan and exactly specified gates remaining before implementation starts. Work on a separate branch or worktree; this session is the orchestration session; tasks run through subagents attached to it, as parallel as possible, with a board monitoring progress.

## Operator line

**Provenance:** signed by an operator-designated signing session under `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-OD-SIGNING.md@7dab0e14c71c0a6f42a0c7bed5f4cb9d52355ad1` (IIAC Amendment 01, commission given in chat 2026-11-12; operator answer "Sesja podpisująca"). The line below is in the plain form because it is written by that session; it is not claimed to be operator-written. **Transcription check:** compared with the operator's pasted message "Operator decision — SFX execution sequencing" and the operator's follow-up paragraph in the same message, in the SFX planning chat transcript (session "Scenario Factory-X campaign planning", 2026-11-12), read by this session through the local session transcript (2026-11-12T12:43Z). Every clause (P0..P4 scope, batch-first rules, STOP before SFX-P5, entry conditions for P5..P13, SEF-X/READINESS independence, prohibitions, PRE_IMPLEMENTATION_READY goal, orchestration with attached subagents and a board) is present in the Polish original; judged faithful, with no widening. Consistent with DELEGATION-SFX-PRE-FREEZE item 1.

```text
Approved: OD-SFX-15 @ 2026-11-12T14:10:14+01:00
```

**Explicit non-actions:** does not authorize SFX-P5 or later, does not accept any ADR, does not change FRZ, does not satisfy GATE-SFX-ENTRY, and does not bypass SCENARIO-GATE or PROD-Q.
