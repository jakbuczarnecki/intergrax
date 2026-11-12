# PLAN T-IGX-037-A — SCENARIO-FACTORY-X P0: roadmap registration and SFX documentation baseline (docs only)

Parent: SCENARIO-FACTORY-X (SFX) master campaign plan, **Revision R1**. R1 is the corrective review of R0, and its approval scope covers W0–W2 only. The operator approved both revisions in the planning session on 2026-11-12. This batch publishes R1 as `governance/tickets/plans/SFX-MASTER-PLAN.md`, including its `CORRECTION_MATRIX` (WBS rows SFX-P0-01..03, wave W0). Approval of this plan does not authorize any post-freeze implementation.

Reference revision: `origin/development` = `eadb1c2116feb0a2090b234a36fd4182bd6ace8d`.

This batch changes documentation and governance records only. It touches no product path (`intergrax/`, `agents/`, `applications/`) and no `scripts/` path. IIAC ACT-N11 is therefore unaffected. The batch is allowed while ARCH-FREEZE is OPEN (OD-SFX-04: pre-freeze scope P0–P4 is docs, research, design and agent-env only).

## Deliverables

1. **Operator decision records** under `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/`:
   - `OD-SFX-01.md`: placement and stage registration
   - `OD-SFX-02.md`: Scenario Studio stack, a React + TypeScript + Vite SPA in the Tier-3 host
   - `OD-SFX-03.md`: IDE toolkit model, CLI + agent-env skills/rules/agents + optional MCP live mode, one factory core
   - `OD-SFX-04.md`: pre-freeze scope
   - `OD-SFX-15.md`: pre-freeze execution sequencing. The campaign starts at once, in parallel with MCP-SERVER-X, PDC-X and ARCH-FREEZE. SFX-P0..P4 are authorized. Work STOPs before SFX-P5. The pre-freeze target is `PRE_IMPLEMENTATION_READY`.
   - `OD-SFX-16.md`: resolution of C-OD-01. The regenerated `.claude/` and `.cursor/` adapters may ship with the agent-env sources in SFX-P4. Ticket approvals are delegated to a dedicated delegate session, and independent work runs in linked sub-sessions that report to the orchestrator board.

   Each record holds the operator's decision as a labelled English rendering of the Polish planning chat (2026-11-12) and a signature section. The implementer does not sign. Status stays `PENDING SIGNATURE` until the operator or a commissioned delegate signs.

2. **Roadmap amendment** in `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` (master plan R1 §2, A1–A10):
   - A1: header "Scenario gate" sentence. Generated scenarios stay proposal-only or sandbox; SFX closure admits no scenario to production.
   - A2: §2.0.2 FRZ linkage row. No new FRZ IDs; 225 / 26 unchanged.
   - A3: §2.0.3 predecessor row.
   - A4: §3 row `SCENARIO-FACTORY-X`, inserted between ARCH-FREEZE and SEF-X, status `[ ] PLANNED / MANDATORY / POST-ARCH-FREEZE PRIORITY`. The row names the two engineering modes `DETERMINISTIC_SCAFFOLD` and `AGENTIC_ENGINEERING`, and states that closure requires the agentic mode to be verified on Accepted and certified SEF-X mechanisms. The registry is described as a catalog sidecar, not a new registry. Product certification is distinct from scenario admission.
   - A5: ARCH-FREEZE "next stages" sentence.
   - A6: new §3.0.7b subsection. It records the one-way edge "SFX closure ← SEF-X GATE-SEF-FULL + SEF scoped B6" and the admission separation.
   - A7: chain blocks.
   - A8: §5 ledger. Adds the SFX row and the missing SEF-X row, and corrects ARCH-FREEZE "Next".
   - A9: §7 programme goal.
   - A10: amendment audit record.

   Unchanged:
   - the predecessors of SEF-X, SCENARIO-READINESS-X, SCENARIO-GATE and PROD-Q;
   - every FRZ status;
   - the stale MCP-SERVER-X row (RA-F-01, owned by the MCP session).

3. **SSOT plan** `governance/tickets/plans/SFX-MASTER-PLAN.md`: the approved master plan R1, edited only for publication. Concretely: links are made relative, and the plan-mode header is replaced by the SSOT header.

4. **Capability doc pair** (multi-layer feature topology):
   - `docs/project/capabilities/architecture/SCENARIO_FACTORY.md` (status Proposed)
   - `docs/project/capabilities/plan/SCENARIO_FACTORY.md`
   - a hub row in `docs/project/capabilities/README.md`

5. **Qualification dossier skeleton** `docs/project/maintainers/qualification/scenario_factory_x/`:
   - `SFX_INDEX.md`
   - `SFX_WBS.md`
   - `SFX_GAP_REGISTER.md`
   - `SFX_SHA_LADDER.md` (all SHAs unset except `P0_REGISTRATION_SHA`, which is set after merge)
   - `SFX_REQUIREMENTS_TRACEABILITY.json` (REQ-SFX-001..054 skeleton, each mapped to R1 task IDs)

## Acceptance

- Every roadmap view that lists the post-freeze tail contains SFX. None of them makes SFX a predecessor of SCENARIO-READINESS-X or SCENARIO-GATE.
- `tests/qualification/roadmap_replay_x` shows no new failures compared with `eadb1c21`. Pre-existing red tests are recorded in `PDC_REG_DIAGNOSIS.md`.
- No change under `intergrax/`, `agents/`, `applications/` or `scripts/`.
- No FRZ change. No SFX, ARCH-FREEZE or production-readiness CLOSED claim. ADR-SEF-001 stays Proposed.

## Test contract

```text
uv run pytest tests/qualification/roadmap_replay_x tests/unit/governance -q
uv run python scripts/governance/sync_agent_env.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product code, scripts, and agent-env changes. The IDE toolkit is SFX-P4.
- Competitive research findings (SFX-P1, ticket T-IGX-037-B).
- ADRs (SFX-P2).
- Fixes for RA-F-01, RA-F-03..06.
- FRZ changes, OD signatures by the implementer, and merges.
