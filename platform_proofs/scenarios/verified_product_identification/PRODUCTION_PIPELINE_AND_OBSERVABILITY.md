# VPI production pipeline and observability (5C12)

## Public boundary

- `ProductIdentificationPipelineService.run(request)` — canonical production entry point.
- `build_product_identification_pipeline(...)` — scenario-owned composition root (constructor injection only).
- Contracts: `ProductIdentificationPipelineRequest` (field `query: ProductIdentificationQuery`), `ProductIdentificationPipelineResult`, `ProductIdentificationRunId`.

## Stage order

1. Typed query context observation  
2. Multi-channel retrieval (`MultiChannelRetrievalService`)  
3. Offer-level fusion (`OfferCandidateFusionService` + RRF strategy)  
4. Identity hypothesis formation (`ProductIdentityHypothesisService`)  
5. Identity evaluation / reranking (`IdentityHypothesisEvaluationService`)  
6. Verification and terminal decision (`ProductIdentificationVerificationService` / 5C10)  
7. Clarification requirement selection (5C11) when decision is not terminal for clarification  
8. Terminal observation (exactly once per successful business run)

Infrastructure stage failure short-circuits downstream business stages. Empty successful retrieval is not `NO_MATCH`.

## Query boundary today

**Authoritative query:** `ProductIdentificationQuery` (`verification_context` + optional `search_text`).

- **Verification semantics:** `ProductIdentificationQuery.verification_context` (`ProductIdentificationQueryContext`).
- **Lexical / vector semantics:** explicit `ProductIdentificationQuery.search_text` when present — not synthesized from constraints or identifiers.
- **Retrieval request:** derived inside the pipeline by `ProductIdentificationRetrievalRequestBuilder` (canonical: `DeterministicProductIdentificationRetrievalRequestBuilder`).
- **Current input origin:** pipeline observability always reports `TYPED_QUERY_CONTEXT` (caller does not select origin).
- **`RAW_QUERY`:** not implemented — reserved for future Query Understanding → `ProductIdentificationQuery` → pipeline.

Retrieval and verification cannot receive unrelated query semantics through the public pipeline API.

Full natural-language query understanding is **not** implemented in 5C12-R1.

## Application vs proof

- Application emits immutable `ProductIdentificationObservation` events on the **same** execution path as the business result.
- Proof/evaluator code consumes `ProductIdentificationPipelineResult` plus sink snapshots externally — no gold labels, expected outcomes, or benchmark truth in application contracts.

## Observability

- Typed payloads per stage (no `dict[str, object]`, no chain-of-thought).
- `run_id` + monotonic `sequence` per run; per-stage `duration_ns` via `execute_timed_stage`.
- Retrieval: per invoked channel, success vs failure vs empty, bounded offer refs.
- Fusion: `fusion_score` labeled as fusion fact, not verification confidence.
- Verification: full 5C10 provenance including `SourceIdentityFact`.
- Sink modes: `BEST_EFFORT` (NoOp default) and `REQUIRED` (canonical proof runs).

## Agent adapter

`application/agent.py` remains a non-canonical ReflexAgent skeleton. Business core is `ProductIdentificationPipelineService`; a thin typed adapter may be added later.

## Prohibitions

- No `cluster_id` as identity truth in pipeline/observability code.
- No proof-layer imports in `application/pipeline` or `application/observability`.
- No provider concrete dependencies in pipeline core.
