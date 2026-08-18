---
id: IJ-2026-06-12-004
date: 2026-06-12
tiers:
  - tier-1
  - tier-2
  - tier-3
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-CLOSE-LEG-5
status: completed
commit: pending
adr: docs/project/technical/adr/entries/2026-06-12/ADR-FLOW-005.md
---

# Retire Tier-1 RuntimeEngine pipeline stack (ACP-CLOSE-LEG-5)

## Operator request

Remove all legacy Tier-1 pipeline machinery (`RuntimeEngine`, `pipelines/`, `runtime_steps/`, engine planner stack). Agent logic lives in Tier-2 ACP implementations only; the old config-driven pipeline path will never be used again.

## Summary

Deleted the full RuntimeEngine pipeline stack and migrated surviving utilities (bounded tool loop, RAG/websearch/tools context invocation, context helpers) to `nexus/tools/` and `nexus/context/`. Removed `uaep_pipeline_bridge`, agent `steps/pipeline.py` files, pipeline-related `RuntimeConfig` fields, and legacy UAEP scaffold paths. Updated fleet agents, gate stubs, tests, and ACP-only scaffold (`--uaep` rejected). Regenerated domain audit prompts; scrubbed architecture/plan/guides/agent docs of pipeline references. Canon now states exclusive author loop control via `on_next_step` + `HarnessKernel`. Added ADR-FLOW-005 and plan row ACP-CLOSE-LEG-5.

## Project impact

- Single agent execution model: ACP `on_next_step` / UAEP shim for tests only.
- Smaller Tier-1 surface; audits no longer conflate Nexus task planner with retired engine planner.
- ToolRuntime and catalog dispatch keep shared context invocation without pipeline steps.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §13.5 |
| Plan | `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — `ACP-CLOSE-LEG-5` |
| ADR | `docs/project/technical/adr/entries/2026-06-12/ADR-FLOW-005.md` |

## Changed artifacts

- `intergrax/runtime/nexus/pipelines/` — removed
- `intergrax/runtime/nexus/runtime_steps/` — removed
- `intergrax/runtime/nexus/engine/runtime.py` — removed (`RuntimeEngine`)
- `intergrax/runtime/nexus/tools/tool_loop.py`, `plan_context_invocation.py` — preserved tool/context behavior
- `intergrax/agents/authoring/stub_llm.py`, `acp_stub_reflex.py` — fleet stub helpers without pipeline
- `agents/*/steps/pipeline.py` — removed across fleet
- `intergrax/scaffold/new_agent.py` — ACP-only scaffold
- `testing_support/uaep_gate_stubs.py` — handoff via `AgentDecision` without pipeline

## Verification

```bash
uv run pytest -m gate -q
python scripts/docs/check_docs_domain_pairs.py
```

Result: **1776 passed** (gate); domain pairs OK.

## Risks and follow-ups

- `check_harness_no_getattr.py` reports one grandfathered getattr in `acp_budget_reactions.py` (pre-existing).
- Historical plan register rows retain retired path names only in audit-ID cross-reference tables (superseded by ADR-FLOW-005).
