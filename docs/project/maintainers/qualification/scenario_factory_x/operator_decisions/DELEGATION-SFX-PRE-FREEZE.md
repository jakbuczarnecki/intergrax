# DELEGATION-SFX-PRE-FREEZE — Operator commission for plan-approval signing (SFX-P0..P4)

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-SFX-PRE-FREEZE |
| **Stage** | SCENARIO-FACTORY-X (SFX) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6, as summarized in `governance/tickets/README.md` §Approved. Precedents: `DELEGATION-PDC-WAVE-A.md` and `DELEGATION-PDC-WAVE-B.md`. |
| **Date** | 2026-11-12 |
| **Provenance** | The orchestrator session transcribed this record. It is not operator-signed. Assurance: `process_enforced`. |

## Commission (transcription; English rendering of the Polish planning chat)

1. **Execution sequencing decision (OD-SFX-15).** The operator wrote: "Start the SCENARIO-FACTORY-X campaign immediately, in parallel with MCP-SERVER-X, PDC-X and ARCH-FREEZE."
   - The operator authorized SFX-P0 through SFX-P4 before ARCH-FREEZE:
     - P0: roadmap and governance registration;
     - P1: research and competitive analysis;
     - P2: capability audit, gap analysis, proposed architecture and ADRs;
     - P3: UX design, mock-ups and contract designs;
     - P4: only an allowed, proposal-only IDE toolkit, with no change to protected platform mechanisms unless separately authorized.
   - Execution rules: work batch-first, in separate worktrees, with independent reviews, following canonical DevOS.
   - The operator required a STOP before SFX-P5 and before any product implementation. SFX-P5..P13 start only after formal `ARCH-FREEZE = CLOSED`, acceptance of the required contracts and architecture decisions, and `GATE-SFX-ENTRY`.
   - SEF-X and SCENARIO-READINESS-X stay independent parallel stages.
   - Frozen contracts must not be changed prematurely. No production runtime may be started. SCENARIO-GATE and PROD-Q must not be bypassed.
   - The pre-freeze target is `PRE_IMPLEMENTATION_READY`.
2. **Signing route.** The orchestrator asked how the `Approved:` lines of the SFX tickets should be written. The options were a delegate in a separate session, or the operator. The operator chose **"Delegat w osobnej sesji"** (English rendering: "A delegate in a separate session").
3. **C-OD-01 (OD-SFX-16).** Asked whether the regenerated `.claude/skills` and `.cursor/` adapters may ship before ARCH-FREEZE as part of the agent-env delivery, the operator chose **"Tak, z agent-env"** (English rendering: "Yes, with agent-env"). `sync_agent_env.py` and `role_registry.yaml` stay untouched.
4. **Independence.** Asked who performs the generalization-set authoring and the PR reviews, the operator answered (English rendering): "Run them as sub-sessions of the current session, so these sub-sessions can report progress to the board."

## Scope of the delegation

- A dedicated delegate session may write `Approved: operator-delegate @ <ISO8601>`, followed on the next line by `Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@<commit sha>`. This is allowed in these tickets only: **T-IGX-037-A, T-IGX-037-B, T-IGX-037-C, T-IGX-037-D and T-IGX-037-E**.
- The delegate decides independently and may refuse. Before writing a line, it confirms all of the following for that ticket:
  - The `Plan digest:` equals `scripts/governance/plan_digest.py` run on the referenced plan.
  - The plan matches the SFX master plan R1 units for W0–W2: SFX-P0..P4, with P4 limited to SFX-P4-A01/A02.
  - The plan changes no path under `intergrax/`, `agents/`, `applications/` or `scripts/`. T-IGX-037-E additionally allows `agent-env/`, `.claude/` and `.cursor/` adapters and nothing under `governance/agent-control-plane/`.
  - The plan does not authorize SFX-P5 or later, does not accept any ADR, does not change FRZ, and does not weaken any governance invariant.
- **Not delegated:**
  - OD signatures;
  - ADR acceptance;
  - IIAC activation;
  - FRZ promotion;
  - merges;
  - GATE-SFX-ENTRY;
  - any SFX-P5+ ticket;
  - any other ticket.

## Separation

- The orchestrator session, its implementer subagents and its reviewer sessions never write these lines.
- The delegate session does not implement or review any of the five tickets.
