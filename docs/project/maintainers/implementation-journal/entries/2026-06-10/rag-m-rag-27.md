---
id: IJ-2026-06-10-036
date: 2026-06-10
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.27
  - GAP-RAG-08
  - GAP-RAG-09
  - AUDIT-IDEAL-14.7
status: completed
commit: pending
adr: none — wires existing OpenTelemetry API on Tier-0 hot paths; no contract change
---

# OpenTelemetry spans on RAG retrieve + ingest (M-RAG.27)

## Operator request

Execute next Wave 3 step: OTel spans on `RetrievalService.retrieve` and `IngestPipeline.run` stages.

## Summary

Added `intergrax/rag/tracking/rag_spans.py` with canonical span registry, `rag_span()` context manager (tracer `intergrax.rag`), and env `INTERGRAX_RAG_OTEL_SPANS_ENABLED` (default on). Wired retrieve + ingest hot paths. Registered gate `scripts/maintenance/check_rag_otel_span_registry.py` in `check_observability_gates.py`.

Aggregated RAG metrics remain opt-in via `INTERGRAX_RAG_METRICS_ENABLED`; spans are on the default observability spine unless explicitly disabled.

## Project impact

RAG retrieve and ingest hot paths emit canonical OTel spans by default, aligning Tier-0 RAG with the platform observability spine and closing GAP-RAG-08/09 audit findings.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/RAG.md` GAP-RAG-08, GAP-RAG-09 closed |
| Plan | `docs/project/maintainers/plans/RAG.md` M-RAG.27 **Done** |
| Audit | AUDIT-IDEAL-14.7 |

## Changed artifacts

- `intergrax/rag/tracking/rag_spans.py` — span registry and `rag_span()` helper
- `intergrax/rag/retrieval/retrieval_service.py`, `intergrax/rag/ingest/ingest_pipeline.py` — span wiring
- `scripts/maintenance/check_rag_otel_span_registry.py` — observability gate registration

## Verification

```bash
uv run pytest tests/unit/rag/tracking/test_rag_otel_spans.py -m gate -q
uv run python scripts/maintenance/check_rag_otel_span_registry.py
uv run python scripts/maintenance/check_observability_gates.py
```

## Risks and follow-ups

- Span volume on high-throughput ingest may require sampling tuning in production deployments.
- Next Wave 3 item: **M-RAG.28** retriever fallback chain.
