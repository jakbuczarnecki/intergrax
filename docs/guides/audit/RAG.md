# RAG and Retrieval Engine — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/RAG.md`](../architecture/RAG.md) · [`plan/RAG.md`](../plan/RAG.md)  
**Audit map layers:** 14 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with repository access.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus`).
4. Output must follow [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: RAG
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — RAG and Retrieval Engine (`RAG`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **RAG and Retrieval Engine** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** Deep audit of the Tier-0 retrieval engine: ingest, chunking, indexing, retrieval modes, resilience, and production posture vs state-of-the-art RAG systems.

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/RAG.md` — current architecture canon
3. `docs/plan/RAG.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 14
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix K §K.5** — control-plane wiring

---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
intergrax/rag/, RetrievalService, IngestPipeline, rag.* catalog tools, RagStep
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

1. Single canonical path: RagProfile → RetrievalService → catalog tools / Nexus RagStep.
2. No agent direct vectorstore.query shortcuts.
3. Retrieval modes: vector, keyword, hybrid, fusion, graph, rerank, agentic, hierarchical — wired vs documented-only.
4. Ingest: parser catalog, chunking strategies, contextual enrich, dual-index / TOC for large docs.
5. Strategy selection: explicit Tier-3 policy vs autonomous (AHI deferred) — dead config flagged.
6. Short/medium vs multi-GB corpus behaviour — sync vs job orchestration.
7. Resilience: retry, fallback chains, circuit breakers on embedding/retriever paths.
8. Security: retrieval poisoning defence on **all** surfaces (Nexus + catalog tools).
9. Citations, tenant MetadataFilter, multi-tenant isolation with prod backends.
10. Observability: RetrievalTrace, parser trace, metrics, OTel on hot paths.
11. Golden retrieval tests, recall/MRR eval harness, load/soak gaps.

---

## 4. Workload and scale probes

Evaluate behaviour for:

Single-page doc, 100-page PDF, book-scale corpus, high QPS retrieve, poisoned chunks.

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

RagProfile, IntegrationProfile vector_store/document_parser/rerank_provider, rag_runtime_bridge.

Confirm overrides are **wired**, not documentation-only.

---

## 6. Cross-cutting checklist (mandatory)

Apply every item in `docs/guides/audit/README.md` §Shared production Harness checklist:

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

## 7. Production comparison

Compare the implementation to **production-grade systems** in this domain (commercial and open-source). State clearly:

- What Intergrax already matches at L3 production Harness OS level
- What is L2 or below with specific gaps
- What is intentionally deferred (design boundary) vs **niedoróbka** / missing wiring

---

## 8. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5:

```text
L0 — Fragmented
L1 — Operational MVP
L2 — Scalable Harness
L3 — Production Harness OS
L4 — Adaptive Agent OS
```

Report **score before**, **target for current milestone**, evidence, and **remaining risks**.

---

## 9. Verification commands

Run applicable checks; cite results:

```bash
uv run pytest -m gate -q
uv run pytest tests/unit/<relevant>/ -q
python scripts/check_harness_no_getattr.py
# plus domain-specific scripts discovered during inspection
```

---

## 10. Output and mode rules

- Follow output format in `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 (Audit Result template).
- End with §8 Completion Summary.
- `audit-only`: **no file edits**
- `audit-and-fix`: update `docs/plan/RAG.md` gap rows and `docs/architecture/RAG.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
