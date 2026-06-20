# RAG — Implementation Plan

**Architecture (1:1):** [`architecture/RAG.md`](../architecture/RAG.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/plan/` satellites on demand).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/RAG_audit_history.md`](plan/plan/RAG_audit_history.md) | audit history |
| [`plan/plan/RAG_embedded_detail.md`](plan/plan/RAG_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read this hub + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — RAG gap register (layer 14)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6, §7.7  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Engine depth audit:** 2026-06-10 — full register in [`architecture/RAG.md`](../architecture/RAG.md) §Engine depth audit register

| ID | AUDIT § | Gap | Priority | Status | M-RAG |
|----|---------|-----|----------|--------|-------|
| AUDIT-IDEAL-14.1 | §14 RAG | Graph RAG production profile (shared with MEMORY) | P1 | **Done** | M-RAG.12 (stable) |
| AUDIT-IDEAL-14.3 | §14 RAG | Wire `RagProfile.query_expansion` to retrieval path | P0 | **Done** | M-RAG.23 |
| AUDIT-IDEAL-14.4 | §14 RAG | Dual-index + hierarchical retriever default bootstrap | P1 | **Done** | M-RAG.24 |
| AUDIT-IDEAL-14.5 | §14 RAG | Retrieval poisoning defense on `rag.retrieve` catalog path | P1 | **Done** | M-RAG.25 |
| AUDIT-IDEAL-14.6 | §14 RAG | Large-corpus async ingest (stream / job orchestration) | P1 | **Done** | M-RAG.26 |
| AUDIT-IDEAL-14.7 | §14 RAG | OpenTelemetry spans on RAG retrieve + ingest hot path | P2 | **Done** | M-RAG.27 |
| AUDIT-IDEAL-14.8 | §14 RAG · §3.7.1 | Universal GraphRAG platform — backend registry, lifecycle, retrieval hardening | P1 | **Done** (G1–G3; G4 optional) | M-RAG-GRAPH (M-RAG.38–M-RAG.47, M-RAG.48, M-RAG.52) |

**Note:** AUDIT-IDEAL-14.2 (retrieval poisoning on product hosts) is owned by [`plan/MEMORY.md`](MEMORY.md) + UAEP security wiring — Nexus `rag.retrieve` (catalog) path.

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR (when applicable) → update this table + master register → gate green. Additional GAP-RAG rows without AUDIT-IDEAL IDs use M-RAG.\* only.

**Engine audit (2026-06-13):** Maturity **L3 implementation / L3 control plane** — **Frozen**. Closeout: [Phase M-RAG-CONVERGE](#phase-m-rag-converge--doc--diagnostics-closeout-2026-06-13).

---
