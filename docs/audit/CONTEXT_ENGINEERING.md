# Context Engineering Engine — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/CONTEXT_ENGINEERING.md`](../architecture/CONTEXT_ENGINEERING.md) · [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md)  
**Audit map layers:** 16 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: CONTEXT_ENGINEERING
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Context Engineering Engine (`CONTEXT_ENGINEERING`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Context Engineering Engine** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **Tier-1 context compiler engine**: plugin providers, budget/degradation, step-aware assembly, provenance, quality scoring, observability, and Tier-3 ContextProfile control plane — integrated with Harness.

## Key symbols and contracts

ContextEngine · ContextSourceProvider · ContextFragment · ContextAssemblyRequest · AssembledContext · ContextCompiler · ContextManager · AgentContextBundle · ContextBudgetPolicy · TaskContextAssemblyOptions · ContextDecisionProfile · ContextProfile · DegradationLadder

## Active plan phases (verify status vs code reality)

CE-LC Done · CE-DEPTH Done · §6.1av CE-MAINT closed (OTel, cost, baselines)

## Known open gaps — re-validate every item (closed / still open / partial)

GAP-CTX-12 adaptive ranking **Frozen** → AHI-MAINT-04 · CE-LC register closed

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/CONTEXT_ENGINEERING.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/CONTEXT_ENGINEERING.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 16
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix L**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/runtime/nexus/context/ (context_engine.py target, context_compiler.py, context_manager.py)
intergrax/context_engineering/ (ContextEngine · providers)
intergrax/runtime/architecture/context_engineering.py · context_regression_benchmark.py
intergrax/contracts/context_assembly.py
intergrax/context/ (target contracts + plugin registry)
applications/_shared/context_runtime_bridge.py · context_wiring.py
intergrax/runtime/events/context_skill_recording.py · payloads/canonical.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. ContextEngine.assemble() is the single target entry (CE-3) — no agent prompt concatenation.
2. ContextSourceProvider plugin catalog with register_context_plugin (CE-2).
3. Global token budget + DegradationLadder never-overflow (ADR-MEM-001 / ContextCompiler).
4. Provenance on every included/excluded fragment.
5. CONTEXT_ASSEMBLED / CONTEXT_TRIMMED events on all paths.
6. BEFORE_CONTEXT_BUILD / AFTER_CONTEXT_BUILD hooks wired.
7. ContextProfile drives Tier-3 presets (default, codebase, regulated_minimal).
8. Step-aware ContextAssemblyRequest (step_kind, objective) on UAEP path (CE-4).
9. Quality scoring integrated in DefaultContextRanker (CE-10).
10. OTel spans on assemble/collect/budget (CE-9).
11. RAG/Memory/Tool outputs enter via providers — not CE owning retrieval.
12. Codebase preset uses WorkspaceContextProvider — not full repo dump.
13. Context regression benchmark gates preset drift.
14. Forbidden: Tier-2 imports of Nexus context internals for assembly.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- 128k budget with multi-source fragments — degradation ladder trace.
- Graph node SUMMARY_ONLY tier under tight budget.
- Codebase 1k+ files — retrieval-first workspace provider.
- Delegation child explore preset — parent synthesis only.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ContextProfile · context_runtime_bridge · context_wiring · context_plugins[] · engine_preset · BEFORE_CONTEXT_BUILD hooks

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/audit/README.md` §Shared production Harness checklist:

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

Compare against: **Cursor-class context engine · Anthropic-style budgeting · LangGraph-style state injection patterns**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Agent-built prompts · silent fragment drop · string-heuristic source detection as final state · CE logic in Tier-2

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/nexus/context/ -m gate -q
uv run pytest tests/unit/applications/test_context_wiring.py -m gate -q
uv run pytest tests/acceptance/test_acceptance_context_compiler_long_session.py -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/CONTEXT_ENGINEERING.md` gap rows + `docs/architecture/CONTEXT_ENGINEERING.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
