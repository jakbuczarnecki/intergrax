---
id: IJ-2026-06-10-008
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
adr: none — wires existing OpenTelemetry API on Tier-0 hot paths; no contract change
---

# OpenTelemetry spans on RAG retrieve + ingest (M-RAG.27)

## Operator request

Execute next Wave 3 step: OTel spans on `RetrievalService.retrieve` and `IngestPipeline.run` stages.

## Summary

Added `intergrax/rag/tracking/rag_spans.py` with canonical span registry, `rag_span()` context manager (tracer `intergrax.rag`), and env `INTERGRAX_RAG_OTEL_SPANS_ENABLED` (default on). Wired retrieve + ingest hot paths. Registered gate `scripts/check_rag_otel_span_registry.py` in `check_observability_gates.py`.

Aggregated RAG metrics remain opt-in via `INTERGRAX_RAG_METRICS_ENABLED`; spans are on the default observability spine unless explicitly disabled.

## Verification

```bash
uv run pytest tests/unit/rag/tracking/test_rag_otel_spans.py -m gate -q
uv run python scripts/check_rag_otel_span_registry.py
uv run python scripts/check_observability_gates.py
```

## Next step

**M-RAG.28** — completed in IJ-2026-06-10-009.
