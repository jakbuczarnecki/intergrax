# Memory Platform — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/MEMORY.md`](../architecture/MEMORY.md) · [`plan/MEMORY.md`](../plan/MEMORY.md)  
**Audit map layers:** 15 · compact slice: [`audit_slices/MEMORY.md`](../guides/audit_slices/MEMORY.md)  
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

domain: MEMORY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Memory Platform (`MEMORY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Memory Platform** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **memory stores**, scopes, lifecycle, consolidation, and Knowledge-vs-LTM boundary — explicit, governed, observable, retrieval-first. Context assembly is audited under CONTEXT_ENGINEERING.

## Key symbols and contracts

MemoryProfile · MemoryKind · MemoryWritePolicy · PolicyScopedMemoryView · MemoryConsolidationJob · MemoryView · SharedTaskContext

## Active plan phases (verify status vs code reality)

MEM Done · MEM-DEPTH Done · MEM-OBS.1 · ADR-MEM-001

## Known open gaps — re-validate every item (closed / still open / partial)

MEMORY-LC Done · MEM-DEPTH Done · §6.1av depth closed (procedural/org/temporal) · MEM-MAINT-03 LangMem/Zep entity graph parity **backlog** (not Mem0 SaaS; no Phase K)

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/MEMORY.md`](../guides/audit_slices/MEMORY.md) — compact slice (layers **15**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/MEMORY.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/MEMORY.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/project/maintainers/plans/MEMORY.md` — hub + one `plan/satellites/` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **Appendix G** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/MEMORY.md` — then inspect:

```text
intergrax/memory/ (user_profile_memory.py, contracts/)
intergrax/runtime/nexus/session/ · intergrax/runtime/task_memory/
intergrax/runtime/organization/ · consolidation services
applications/_shared/memory_wiring.py · memory_runtime_bridge.py
EntityGraphMemoryStore · workspace_index_spike.py (RFC — CE owns production wiring)
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Memory types separated: STM, task KV, session, user LTM, tenant, procedural, shared context.
2. Agents do not write Redis/Postgres/vector DB directly.
3. Session vs checkpoint vs task KV stores distinct.
4. Every read/write scoped; subagent namespace isolation (task_id/delegation/{node_id}/).
5. MemoryWritePolicy + BEFORE_MEMORY_WRITE hooks enforced.
6. Retrieval-first for large history — consolidation not full dump.
7. Knowledge (RAG) ≠ user LTM — graph RAG ≠ Zep-style entity memory.
8. Retention_days, FIFO session limits, LTM top_k enforced.
9. LTM logical delete tombstones vectors where applicable.
10. Org profile vs user profile separation.
11. Consolidation triggers configured in MemoryProfile.
12. RAG knowledge does not silently mutate user memory profile.
13. Layer C context compiler spec lives in CONTEXT_ENGINEERING canon — not duplicated here.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Long session exceeding FIFO — summarization path.
- Delegation namespace isolation under parallel subagents.
- Large LTM corpus with vector search + dedup.
- Entity graph memory under concurrent writes.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

MemoryProfile · memory_runtime_bridge · BEFORE_MEMORY_WRITE hooks · TaskMemoryViewBinding on ToolWiringContext

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

Compare against: **Mem0/Zep/Letta taxonomies · LangMem consolidation**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Global memory store · graph RAG as user memory · unscoped writes · agents with DB drivers

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/memory/ -q
uv run pytest tests/unit/applications/test_memory_profile_runtime_bridge.py -m gate -q
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
