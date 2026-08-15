# Nexus Execution Flow — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/NEXUS_EXECUTION_FLOW.md`](../architecture/NEXUS_EXECUTION_FLOW.md) · [`plan/NEXUS_EXECUTION_FLOW.md`](../plan/NEXUS_EXECUTION_FLOW.md)  
**Audit map layers:** 8–10 · compact slice: [`audit_slices/NEXUS_EXECUTION_FLOW.md`](../guides/audit_slices/NEXUS_EXECUTION_FLOW.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with the repository available, but do not perform broad repository exploration. Read only the files listed in Context budget / Canonical reads, use path-filtered grep before opening files, and do not use semantic search, subagents, or full-repo scans unless the operator explicitly approves.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/audit/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: NEXUS_EXECUTION_FLOW
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Nexus Execution Flow (`NEXUS_EXECUTION_FLOW`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Nexus Execution Flow** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **end-to-end Nexus loop narrative** against NEXUS_EXECUTION_FLOW canon: step ordering, three planning planes, handoff/retry, final response composition, flow-level observability, and acceptance-scenario coverage.

## Key symbols and contracts

Task/TaskLifecycle/TaskResult · SharedTaskContext · AgentContextBundle · TaskContextAssemblyOptions · RuntimeRequest · AgentHandoff · ValidationResult · ExecutionNode

## Active plan phases (verify status vs code reality)

FLOW 18/18 Done · FLOW-CTL · FLOW-8 harness Done/product Deferred · H-APP-WIRING · COG-DEPTH cross-ref

## Known open gaps — re-validate every item (closed / still open / partial)

FLOW-GAP-20 hybrid daemon LKW · UC-6 research stubs · WAITING_FOR_RESOURCES/EXPIRED reserved v1 · production-ready = Partial without strict profile + W-OPS

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/NEXUS_EXECUTION_FLOW.md`](../guides/audit_slices/NEXUS_EXECUTION_FLOW.md) — compact slice (layers **8–10**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/NEXUS_EXECUTION_FLOW.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` — hub + one `plan/satellites/` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **Appendix I §I.2–I.6** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/NEXUS_EXECUTION_FLOW.md` — then inspect:

```text
intergrax/runtime/task/unified_task_runner.py
intergrax/runtime/nexus/nexus_loop.py
intergrax/runtime/nexus/tools/tool_loop.py · plan_context_invocation.py
intergrax/agents/agent_engine.py · authoring/acp_run.py · HarnessKernel
intergrax/runtime/nexus/agent_router.py
intergrax/runtime/nexus/context/context_manager.py
intergrax/runtime/nexus/handoff/coordinator.py
intergrax/runtime/nexus/retry/retry_engine.py
intergrax/runtime/nexus/response/final_response_composer.py
applications/_shared/nexus_factory.py · graph_spec_to_plan.py
tests/acceptance/agent_os/
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Three planning planes distinguished: Nexus planner / agent on_next_step / tool planner.
2. TaskClassifier does not mutate Task.state directly.
3. AgentRouter respects production_mode and registry constraints.
4. Handoff uses HandoffCoordinator — traced lineage.
5. FinalResponseComposer applies merge_strategy from orchestration.
6. FLOW-GAP register items closed in code or explicitly deferred with risk.
7. Cancel is cooperative at step boundaries.
8. Trace reconstructs full 'why did run stop' narrative.
9. DECISION_EMITTED on UAEP steps before side effects.
10. RAG poisoning defense active on catalog rag.retrieve path (cross-check RAG domain).
11. Reserved lifecycle states not used in production hosts.
12. Engine planner requires llm_adapter at bootstrap — fail-fast if missing.
13. Partial completion policy explicit when PARTIALLY_COMPLETED allowed.
14. Evaluation/critic hooks profile-driven — not hardcoded per agent.
15. Lab vs production matrix §1.4 respected in host configs.
16. Acceptance scenarios UC-1–UC-9 / S1–S7 have test evidence.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Acceptance 01–10 including mid-UAEP resume 05b.
- Parallel execution cap integration tests.
- Handoff + retry combined scenarios.
- Long-running loop with nested delegation.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

execution_mode strict|balanced · EvaluationProfile · CriticProfile · require_human_approval · graph_spec on profile · lab trace debug routes

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/project/maintainers/audit/README.md` §Shared production Harness checklist:

- Architecture & modularity
- Configuration & strategy selection
- Override & customization surfaces
- Observability, tracing & logging
- Security & governance
- Reliability & error handling
- Performance & scale
- Testing & verification
- Documentation alignment

---

## 7. Production baseline comparison

Compare against: **Agent OS acceptance suite · reference host presets · W-OPS SLO evidence for production claims**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Confusing three planning planes · agent-specific Nexus branches · undocumented partial completion · flow doc/code drift

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/acceptance/agent_os/ -q
uv run pytest tests/unit/runtime/nexus/ -q -k 'handoff or graph_spec'
python scripts/maintenance/check_harness_no_getattr.py
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- **O1 terse** checkpoint unless operator requests full report.
- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7–§8 for final write-up.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update plan/arch gap rows; **no code** unless operator requests separately.

Begin the audit now.

---END PROMPT---
