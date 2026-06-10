# Memory and Context Engineering — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/MEMORY.md`](../architecture/MEMORY.md) · [`plan/MEMORY.md`](../plan/MEMORY.md)  
**Audit map layers:** 15–16 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: MEMORY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Memory and Context Engineering (`MEMORY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Memory and Context Engineering** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **memory stores**, scopes, lifecycle, **ContextManager** assembly, budgets, consolidation, and Knowledge-vs-LTM boundary — explicit, governed, observable, retrieval-first.

## Key symbols and contracts

MemoryProfile · MemoryKind · MemoryWritePolicy · PolicyScopedMemoryView · AgentContextBundle · ContextBudgetPolicy · TaskContextAssemblyOptions · MemoryConsolidationJob · DegradationLadder · ContextProfile

## Active plan phases (verify status vs code reality)

MEM Done · MEM-DEPTH Done · MEM-OBS.1 · ADR-MEM-001

## Known open gaps — re-validate every item (closed / still open / partial)

Procedural memory minimal · temporal fact validity (MEM-DEPTH-5.2) · LangMem/Zep parity gaps on entity graph · fragmented budgeting partially unified via ContextCompiler

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/MEMORY.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/MEMORY.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 15–16
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix L**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/memory/ (user_profile_memory.py, contracts/)
intergrax/runtime/nexus/session/ · intergrax/runtime/task_memory/
intergrax/runtime/nexus/context/context_manager.py · context_models.py
intergrax/runtime/organization/ · consolidation services
applications/_shared/memory_wiring.py · context_runtime_bridge.py · context_wiring.py
ContextCompiler (MEM-DEPTH)
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Memory types separated: STM, task KV, session, user LTM, tenant, procedural, shared context.
2. Agents do not write Redis/Postgres/vector DB directly.
3. Session vs checkpoint vs task KV stores distinct.
4. Every read/write scoped; subagent namespace isolation (task_id/delegation/{node_id}/).
5. MemoryWritePolicy + BEFORE_MEMORY_WRITE hooks enforced.
6. Context assembly via ContextProfile + ContextBudgetPolicy.
7. Provenance on every AgentContextBundle fragment.
8. ContextCompiler unified budget (MEM-DEPTH) — not siloed trims.
9. Retrieval-first for large history — SUMMARIZE_OLDEST/TRUNCATE_OLDEST ladders.
10. Knowledge (RAG) ≠ user LTM — graph RAG ≠ Zep-style entity memory.
11. Retention_days, FIFO session limits, LTM top_k enforced.
12. LTM logical delete tombstones vectors where applicable.
13. Org profile vs user profile separation.
14. Consolidation triggers configured in MemoryProfile.
15. RAG knowledge does not silently mutate user memory profile.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Long session exceeding FIFO — summarization path.
- Tight token budget with multi-source context.
- Delegation namespace isolation under parallel subagents.
- Large LTM corpus with vector search + dedup.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

MemoryProfile · ContextProfile · context_runtime_bridge · BEFORE_MEMORY_WRITE hooks · TaskMemoryViewBinding on ToolWiringContext

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

Compare against: **Mem0/Zep/Letta taxonomies · LangMem consolidation · Anthropic-style context budgeting**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **niedoróbka** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Global memory store · full chat history dump · graph RAG as user memory · unscoped writes · agents with DB drivers

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/nexus/context/ -q
uv run pytest tests/unit/memory/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/MEMORY.md` gap rows + `docs/architecture/MEMORY.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
