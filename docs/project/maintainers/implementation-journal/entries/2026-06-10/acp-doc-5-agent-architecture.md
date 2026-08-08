---
id: IJ-2026-06-10-008
date: 2026-06-10
tiers:
  - tier-0
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-DOC.5
  - ACP-ADR.3
status: completed
commit: pending
adr: docs/project/technical/adr/entries/2026-06-11/ADR-AGENT-003.md
---

# Agent layer architecture canon §31–§36 and ADR-AGENT-003

## Operator request

Finalize agent-layer architecture documentation from audit conclusions: `run()` session API, `on_next_step` step loop, dual observability (agent trace vs application orchestration), per-step LLM routing, shared state visibility, and use-case catalog — aligned with agent + environment cooperation model.

## Summary

Extended `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` with §31–§36 (dual observability, step loop, LLM routing, state matrix, UC catalog, synthesis). Rewrote §13 and §29 for `on_next_step` / `AgentRunTrace`. Added ADR-AGENT-003. Updated plan register (ACP-STEP, ACP-OBS, ACP-LLM, ACP-STATE), AGENT_CREATION_GUIDE Appendix AC, audit prompt, ADR README, and architecture hub cross-refs.

## Project impact

Authors and auditors now have a single normative target: one `run()` per session, many `on_next_step` iterations, full `AgentRunTrace` on result, application orchestration journal separate, per-agent resources from environment merge — without removing Nexus for multi-agent production.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §13, §29–§36 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` ACP-DOC.5, ACP-STEP-*, ACP-OBS-* |
| ADR | `docs/project/technical/adr/entries/2026-06-11/ADR-AGENT-003.md` |
| Audit / gap | GAP-ACP-12..17 closed in docs; implementation rows remain Planned |

## Changed artifacts

- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — §31–§36 + §13/§29/§28/§45 updates
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-DOC.5, ACP-ADR.3, step/obs/llm/state rows
- `docs/project/technical/adr/entries/2026-06-11/ADR-AGENT-003.md` — new ADR
- `docs/project/technical/adr/README.md`, `docs/project/architecture/intergrax_runtime_architecture.md` — ADR-003 links
- `docs/project/technical/guides/AGENT_CREATION_GUIDE.md` — Appendix AC step loop + trace
- `docs/project/maintainers/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` — audit dimensions 28–34
- `scripts/audit/generate_domain_audit_prompts.py` — generator sync

## Verification

```bash
python scripts/audit/check_docs_domain_pairs.py
uv run python scripts/audit/generate_domain_audit_prompts.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (docs-only iteration; code implementation ACP-STEP/OBS remains Planned).

## Risks and follow-ups

- Code still on UAEPExecutor path until ACP-STEP-1..2 and ACP-OBS-1 land.
- `ApplicationRunSummary` requires Nexus/task host wiring (ACP-OBS-2).
