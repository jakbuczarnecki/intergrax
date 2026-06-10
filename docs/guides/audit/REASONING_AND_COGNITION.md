# Reasoning and Cognition — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/REASONING_AND_COGNITION.md`](../architecture/REASONING_AND_COGNITION.md) · [`plan/REASONING_AND_COGNITION.md`](../plan/REASONING_AND_COGNITION.md)  
**Audit map layers:** 7 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with **full repository access**.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/generate_domain_audit_prompts.py`

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

Audit **three cognition planes** (Nexus planning, UAEP engine, tool planning): TaskClassifier, typed plans, DecisionRecord, planner strategies, reasoning failure taxonomy.

## Key symbols and contracts

TaskClassification · NexusPlan/PlanStep · EnginePlan · ToolPlanDecision · DecisionRecord (decision_record.v1) · IntentRoute · ReasoningProfile · OrchestrationProfile.planner_kind/classifier_kind

## Active plan phases (verify status vs code reality)

COG-DEPTH Done · COG-1..6 · COG-3.* classifier · ORCH-CONFIG.1 · COG-OBS residuals

## Known open gaps — re-validate every item (closed / still open / partial)

ReasoningFailureKind enum on trace (COG-6 target) · allow_dynamic_replan partial · engine vs Nexus planner bridge debt documented

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/REASONING_AND_COGNITION.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/REASONING_AND_COGNITION.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 7
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix I §I.4**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/runtime/nexus/task_classifier.py
intergrax/runtime/nexus/planning/task_planner.py · EngineBackedNexusPlanner · nexus_llm_plan_builder.py
applications/_shared/graph_spec_to_plan.py
intergrax/runtime/nexus/tools/catalog_tool_planner.py · tool_planning_service.py · tool_selection.py
intergrax/runtime/nexus/planning/engine_planner_orchestrator.py
intergrax/contracts/decision_record.py
intergrax/prompts/registry/ (planner prompt ids)
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Classification precedes side-effectful execution.
2. Plans are typed (NexusPlan/EnginePlan) — not free-text-only.
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

Apply **every** section in `docs/guides/audit/README.md` §Shared production Harness checklist:

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
| **niedoróbka** / missing wiring | … |

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

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/REASONING_AND_COGNITION.md` gap rows + `docs/architecture/REASONING_AND_COGNITION.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
