# OD-SFX-16 — Execution arrangements: adapters, signing route, independent sub-sessions

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-16 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **APPROVED** |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

Three questions asked by the orchestrator on 2026-11-12: (1) how the SFX tickets' Approved lines are signed; (2) whether the regenerated `.claude/skills` and `.cursor/` adapters may ship before ARCH-FREEZE as part of the agent-env delivery (C-OD-01); (3) who performs the generalization-set authoring and the PR reviews.

## Decision (English rendering of the operator's chat)

(1) Option chosen: "A delegate in a separate session" — see DELEGATION-SFX-PRE-FREEZE. (2) Option chosen: "Yes, with agent-env"; `scripts/governance/sync_agent_env.py` and `governance/agent-control-plane/role_registry.yaml` stay untouched until W3. (3) "Run them as sub-sessions of the current session, so these sub-sessions can report progress to the board."

## Operator line

**Provenance:** signed by an operator-designated signing session under `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-OD-SIGNING.md@7dab0e14c71c0a6f42a0c7bed5f4cb9d52355ad1` (IIAC Amendment 01, commission given in chat 2026-11-12; operator answer "Sesja podpisująca"). The line below is in the plain form because it is written by that session; it is not claimed to be operator-written. **Transcription check:** compared with the SFX planning chat transcript (session "Scenario Factory-X campaign planning", 2026-11-12), read by this session through the local session transcript (answers at 2026-11-12T12:45Z): (1) the operator selected "Delegat w osobnej sesji (Recommended)"; (2) the operator selected "Tak, z agent-env (Recommended)"; (3) the operator wrote the quoted free-text answer, rendered faithfully. Note on (2): the words "until W3" are not the operator's; the question stated only that `sync_agent_env.py` and `role_registry.yaml` stay untouched. This session accepts them as the orchestrator's contextual gloss matching `governance/tickets/plans/SFX-MASTER-PLAN.md` §15.1 (P4-B01/B02 moved to W3). This signature confers no authority to change either file; any W3 change needs GATE-SFX-ENTRY and its own ticket approval.

```text
Approved: OD-SFX-16 @ 2026-11-12T14:10:14+01:00
```

**Explicit non-actions:** does not authorize SFX-P5 or later, does not accept any ADR, does not change FRZ, does not satisfy GATE-SFX-ENTRY, and does not bypass SCENARIO-GATE or PROD-Q.
