**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix E — Harness AI alignment traceability (Phase R)

**Source:** Harness AI philosophy audit (2026-06-01) — scaffold, harness+LLM=agent, tool vs skill, context engineering, subagents, policy.  
**Goal:** Step-by-step implementation readiness; every audit theme maps to Phase R deliverables.  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### E.1 Audit theme → Phase R mapping

| Audit theme | Intergrax today | Gap | Phase R IDs | Status |
|-------------|-----------------|-----|-------------|--------|
| Scaffold | `intergrax/scaffold` | No `new-skill` | R-Skill.7, R.0.4 | Done |
| Harness = Nexus + platform + app wiring | Tier-1 + Tier-0 + Tier-3 | Terminology not in glossary | R.0.2 §5.3 | Done |
| LLM separate from agent module | `llm_adapters` | “Runnable instance” undefined | R.0.2 §5.3 | Done |
| Tool = atomic operation | `ToolContract`, `ToolRuntime` | Doc said “tool/skill” | R.0.3, R.0.1 | Done |
| Skill = goal-oriented pack | Was missing (pre-R); **MVP Done** | Registry + importers + first-party packs | R-Skill.1–R-Skill.10 | Done |
| Option 1: skills = tools | — | **Rejected** — breaks LLM/MCP atomic model | R.0.1 ADR | Done |
| Option 2: Skill Library | — | **Adopted** | R-Skill.* | Done |
| Context engineering | §27–28, `MemoryView`, `TaskContextAssemblyOptions` | No central budget API | R-Context.* | Done |
| Subagents | `GraphExecutor`, handoff §42.15 | No isolated child namespace | R-Delegate.* | Done |
| Policy | Multiple engines | No single bundle narrative | R-Policy.* | Done |
| External skill compatibility | — | No importer | R-Skill.8 | Done |

### E.2 Four-layer capability model (canonical)

```text
Integration  →  vendor/backend Protocol (Postgres, Bing, Jira REST)
Tool         →  atomic LLM/MCP operation (rag.retrieve, jira.search_tasks)
Skill        →  composable pack: tool_ids + prompts + policy fragment + metadata
Agent        →  domain module: contract, UAEP steps, skill_ids[], local governance
Harness      →  Nexus + Tier-0 + Tier-3 wiring (orchestration, trace, policy enforcement)
```

### E.3 Phase R paydown log

| Date | R ID | Summary |
|------|------|---------|
| 2026-06-01 | R.0.1,R.0.2,R.0.3,R.0.4 | ADR Option 2; canon §5.3, §7.1.8, §28.1, §42.11.4, §42.14.3; ToolContract docstring; plan Appendix E |
| 2026-06-01 | R-Skill.1–R-Skill.9,R-Context.1,R-Delegate.1,R-Policy.1 | Skill Library MVP, legal pilot, ContextBudget, DelegationSpec, gate **422 passed** |
| 2026-06-01 | R-Skill.10,R-Context.2,R-Delegate.2–4,R-Policy.2 | Event recording, delegation memory, graph integration test, policy bundle wiring |
| 2026-06-01 | **Phase R (MVP)** | All R-* deliverables **Done** or **Won't fix**; gate **450 passed** |
| — | — | *(append row per merged PR)* |

**Coverage target:** 100% **Done** or **Won't fix** — **met** (2026-06-01). Phase S proceeds on this harness baseline.

---
