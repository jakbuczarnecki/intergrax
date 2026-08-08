# Reasoning and Cognition — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/REASONING_AND_COGNITION.md`](../../architecture/REASONING_AND_COGNITION.md) · [`plan/REASONING_AND_COGNITION.md`](../plans/REASONING_AND_COGNITION.md)  
**Audit map layers:** 7 · compact slice: [`audit_slices/REASONING_AND_COGNITION.md`](../../technical/guides/audit_slices/REASONING_AND_COGNITION.md)  
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

domain: REASONING_AND_COGNITION
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Reasoning and Cognition (`REASONING_AND_COGNITION`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Reasoning and Cognition** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **three cognition planes** (Nexus planning, agent on_next_step, tool planning): TaskClassifier, typed plans, DecisionRecord, planner strategies, reasoning failure taxonomy.

## Key symbols and contracts

TaskClassification · NexusPlan/PlanStep · StepOutcome · ToolPlanDecision · DecisionRecord (decision_record.v1) · IntentRoute · ReasoningProfile · OrchestrationProfile.planner_kind/classifier_kind

## Active plan phases (verify status vs code reality)

COG-DEPTH Done · COG-1..6 · COG-3.* classifier · ORCH-CONFIG.1 · COG-OBS residuals

## Known open gaps — re-validate every item (closed / still open / partial)

ReasoningFailureKind enum on trace (COG-6 target) · allow_dynamic_replan partial · retired RuntimeEngine engine planner (ACP-CLOSE-LEG-5)

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/REASONING_AND_COGNITION.md`](../../technical/guides/audit_slices/REASONING_AND_COGNITION.md) — compact slice (layers **7**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/REASONING_AND_COGNITION.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/REASONING_AND_COGNITION.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/project/maintainers/plans/REASONING_AND_COGNITION.md` — hub + one `plan/satellites/` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **Appendix I §I.4** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/REASONING_AND_COGNITION.md` — then inspect:

```text
intergrax/runtime/nexus/task_classifier.py
intergrax/runtime/nexus/planning/task_planner.py · EngineBackedNexusPlanner · nexus_llm_plan_builder.py
applications/_shared/graph_spec_to_plan.py
intergrax/runtime/nexus/tools/catalog_tool_planner.py · tool_planning_service.py · tool_selection.py
intergrax/agents/authoring/patterns/ (ReAct, plan_execute, …)
intergrax/contracts/decision_record.py
intergrax/prompts/registry/ (planner prompt ids)
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Classification precedes side-effectful execution.
2. Plans are typed (NexusPlan) — not free-text-only.
3. LLM planner falls back to TaskPlanner on parse failure.
4. DecisionRecord on UAEP steps (decision_record.v1 schema).
5. Nexus planning emits decision records (COG-4).
6. Prompt registry ids for planners — not inline strings (COG-2).
7. Tool planning plane separate from Nexus graph planning.
8. MULTI_AGENT semantics ≠ cross-role pipeline conflation.
9. Engine planner requires llm_adapter at bootstrap.
10. Reasoning failures classified separately from tool/runtime failures.
11. Planner LLM identity separable from producer LLM.
12. graph_spec seeding respects trigger_capabilities.
13. catalog_tool_planner single-pass — ReAct status cross-ref TOOL-ENG-6.
14. IntentRoute table maps orchestration tokens correctly.
15. DECISION_EMITTED gate regression (FLOW-12) green.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- research.pipeline 2-step planner.
- Engine planner with multiple routable agent_ids.
- Classifier fan-out on ambiguous intake.
- Replanning when allow_dynamic_replan enabled.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

OrchestrationProfile · ReasoningProfile.tool_planner_prompt_id · IntentRoute · graph_spec · classifier_kind=rules|llm

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

Compare against: **OpenAI/o1-style task decomposition · intent routers · Google ADK structured planners**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Free-text plan only · reasoning+tools in one agent method · no DecisionRecord · ad-hoc prompt strings for planners

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/nexus/planning/ -q
uv run pytest tests/unit/runtime/nexus/tools/test_tool_planning_constraints.py -q
uv run pytest tests/unit/runtime/nexus/tools/test_tool_selection_strategy.py -q
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
