# RAG and Retrieval Engine — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/RAG.md`](../architecture/RAG.md) · [`plan/RAG.md`](../plan/RAG.md)  
**Audit map layers:** 14 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: RAG
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — RAG and Retrieval Engine (`RAG`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **RAG and Retrieval Engine** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Deep audit of the **Tier-0 retrieval engine** vs production RAG systems: ingest, chunking, indexing, retrieval modes, query routing, resilience, security (poisoning), tenancy, observability, evaluation — and honest L2.5/L3 posture with M-RAG-DEPTH gap closure status.

## Key symbols and contracts

RagProfile · RagStack · RetrievalService · RetrievalRequest/RetrievalResult · RetrievalTrace · IngestPipeline · QueryRouter · MetadataFilter · DualIndexStrategy · HierarchicalRetriever

## Active plan phases (verify status vs code reality)

M-RAG.1–M-RAG.22 Done · **M-RAG-DEPTH active** (M-RAG.23–M-RAG.37 ← GAP-RAG-01..23)

## Known open gaps — re-validate every item (closed / still open / partial)

GAP-RAG-21 no RAG load/soak gate · GAP-RAG-22 semantic chunking size guard · `pinecone`/`milvus`/`vespa` beta until ops soak

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/RAG.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/RAG.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 14
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix K §K.5**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/rag/profiles/rag_profile.py
intergrax/rag/bootstrap/rag_stack_bootstrap.py · create_default_rag_stack()
intergrax/rag/ingest/ingest_pipeline.py · ParserPipeline
intergrax/rag/retrieval/retrieval_service.py
intergrax/rag/retrievers/ (hybrid, fusion, graph_rag, hierarchical, multi_query, agentic, …)
intergrax/rag/rerankers/ · intergrax/rag/vectorstore/
intergrax/rag/evaluation/golden_harness.py
intergrax/rag/tracking/ (RetrievalTrace, metrics)
applications/_shared/rag_runtime_bridge.py
intergrax/runtime/nexus/runtime_steps/rag_step.py
intergrax/tools/providers/rag/
.github/workflows/rag-guard.yml · tests/fixtures/rag_golden/
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Single canonical path: RagProfile → RetrievalService → rag.* tools / Nexus RagStep.
2. Agents do not call vectorstore.query or vendor SDKs directly.
3. RagProfile fields wired — flag dead config (especially query_expansion, INTERGRAX_RAG_* env).
4. ParserPipeline + chunking strategies (5+) used on ingest — not raw text shortcut.
5. Retrieval modes: vector, keyword, hybrid, fusion, graph, rerank, agentic, hierarchical — wired vs doc-only.
6. DualIndexStrategy + HierarchicalRetriever wired in default bootstrap for book-scale (GAP-RAG-02/03).
7. Short/medium docs: sync ingest OK with explicit profile.
8. Multi-GB corpora: job orchestration / stream ingest — honest not-ready if missing.
9. Retrieval poisoning defense on **all** surfaces including perform_rag_retrieve catalog path.
10. MetadataFilter + tenant namespace enforcement with prod vector backends.
11. Resilience: embedding retry, retriever retry, fallback chains, circuit breakers — per canon.
12. RetrievalTrace + parser trace; OTel spans on retrieve/ingest hot paths.
13. Citations: chunk metadata + composer; formal Citation on RetrievalResult if canon requires.
14. Golden harness passes (retrieval, graph_rag, multi_hop, agentic scenarios).
15. agentic_enabled defaults safe (false) unless Tier-3 opts in.
16. Graph RAG (document graph) ≠ agent user memory (MEMORY boundary).
17. Integration slugs: vector_store, document_parser, rerank_provider resolved via IntegrationProfile.
18. Compare maturity table in architecture §Production readiness verdict — update if code changed.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Single-page HTML, 100-page PDF, book-scale TOC/hierarchical path.
- High QPS retrieve with reranker latency budget.
- Poisoned chunk injection attempt on Nexus + catalog paths.
- Semantic chunking O(n) embedding cost on large doc.
- Multi-tenant corpus isolation scenario.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

RagProfile + INTERGRAX_RAG_* env · IntegrationProfile vector_store/document_parser/rerank_provider · rag_runtime_bridge · ContextProfile.enable_rag · production_rag_profile()

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

Compare against: **LlamaIndex/Weaviate/Qdrant enterprise RAG · LangChain retrieval pipelines · multi-tenant vector stores · production ingest job queues**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **niedoróbka** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Dense-only retrieval · RAG logic inside agent · multiple uncorrelated RAG paths · missing citations · retrieve without tenant filter

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/rag/ -q
uv run pytest tests/integration/ -q -k rag
# .github/workflows/rag-guard.yml scenarios
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/RAG.md` gap rows + `docs/architecture/RAG.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
