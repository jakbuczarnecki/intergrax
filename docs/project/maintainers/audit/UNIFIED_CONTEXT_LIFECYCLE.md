# Unified Context Lifecycle — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](../architecture/UNIFIED_CONTEXT_LIFECYCLE.md) · [`plan/UNIFIED_CONTEXT_LIFECYCLE.md`](../plan/UNIFIED_CONTEXT_LIFECYCLE.md)  
**Audit map layers:** UCL · compact slice: [`audit_slices/UNIFIED_CONTEXT_LIFECYCLE.md`](../guides/audit_slices/UNIFIED_CONTEXT_LIFECYCLE.md)  
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

domain: UNIFIED_CONTEXT_LIFECYCLE
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Unified Context Lifecycle (`UNIFIED_CONTEXT_LIFECYCLE`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Unified Context Lifecycle** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **Unified Context Lifecycle**: one durable conversation ledger (Memory/Session), one global input budget (Context Engineering), one transformation executor (Token Optimization), and Nexus as lifecycle coordinator for EPHEMERAL_ASSEMBLY and DURABLE_COMPACTION — no parallel compression engines and no application-local summary caches.

## Key symbols and contracts

ModelCallExecutionScope · OptimizationExecutionGuard · ContextOptimizationDecision · ArtifactLookupKey · ReusableOptimizationArtifact · OptimizationArtifactRepository · InMemoryOptimizationArtifactRepository · EPHEMERAL_ASSEMBLY · DURABLE_COMPACTION · PRIMARY_MODEL_CALL · INTERNAL_OPTIMIZATION_CALL

## Active plan phases (verify status vs code reality)

CTX-UCL-ARCH-1 through CTX-UCL-6D ACCEPTED/CLOSED · CTX-UCL-CLOSEOUT-1 ACCEPTED/CLOSED · ADR-UCL-001 Accepted · TOKEN-10E-1 READY_FOR_REVIEW

## Known open gaps — re-validate every item (closed / still open / partial)

TOKEN-10E-1 READY_FOR_REVIEW (durable safety contracts; candidate flow not implemented) · TOKEN-10E-2..4 Blocked · durable production OptimizationArtifactRepository adapter

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/UNIFIED_CONTEXT_LIFECYCLE.md`](../guides/audit_slices/UNIFIED_CONTEXT_LIFECYCLE.md) — compact slice (layers **UCL**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/UNIFIED_CONTEXT_LIFECYCLE.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/UNIFIED_CONTEXT_LIFECYCLE.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/project/maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md` — hub + one `plan/satellites/` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/UNIFIED_CONTEXT_LIFECYCLE.md` — then inspect:

```text
intergrax/runtime/context_lifecycle/ (contracts, repository, InMemoryOptimizationArtifactRepository)
intergrax/runtime/nexus/context/ucl_orchestration.py · context_engine.py
intergrax/runtime/token_optimization/message_sequence_artifact.py
intergrax/runtime/wiring/context_runtime_bridge.py
docs/project/technical/adr/entries/2026-08-01/ADR-UCL-001.md
tests/unit/runtime/context_lifecycle/
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. One durable ledger owner (Memory/Session) — compaction does not rewrite ConversationLedger.
2. One global input budget owner per model call (CE) — no second independent global budget.
3. One optimization executor (TO) — no parallel application compression engines.
4. Nexus coordinates; does not implement algorithms or storage.
5. Every PRIMARY_MODEL_CALL traverses the canonical UCL optimization decision point.
6. EPHEMERAL_ASSEMBLY never mutates active revision.
7. Durable compaction never mutates active revision in place.
8. Internal optimization calls do not recursively traverse full UCL for the same target.
9. Same-key concurrent artifact creation is single-flight coordinated.
10. Lookup-before-create: REUSE_ARTIFACT skips transform execution.
11. MessageSequenceArtifactExecutor only on CREATE_ARTIFACT.
12. Application host does not own a private summary cache or parallel compression engine.
13. Retention is not token optimization; LTM consolidation is a separate concern.
14. HistoryLayer independent summarizer disabled; legacy non-OFF strategies fail-closed.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Two sequential unchanged calls: first creates artifact, second reuses.
- Two concurrent unchanged calls: one summarizer invocation (single-flight).
- Different-key concurrency preserved.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ContextOptimizationPolicy via ApplicationEnvironmentProfile → RuntimeEnvironmentProfile · context_runtime_bridge · InMemoryOptimizationArtifactRepository (reference, not production fallback)

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

Compare against: **Single conversation-ledger + compiler budget + optional durable compaction with reuse-before-create — not per-application summarizer caches**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Second universal compression engine · dual global budgets · in-place active history overwrite · application-local summary cache · recursive full UCL on summarizer calls · Token Optimization-owned artifact persistence

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/context_lifecycle/ -q
uv run pytest tests/unit/runtime/nexus/context/ -q -k ucl
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
