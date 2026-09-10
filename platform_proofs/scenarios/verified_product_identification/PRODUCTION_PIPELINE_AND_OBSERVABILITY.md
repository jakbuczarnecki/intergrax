# VPI production pipeline and observability (5C12)

## Public boundary

- `ProductIdentificationPipelineService.run(request)` — canonical production entry point.
- `build_product_identification_pipeline(...)` — scenario-owned composition root (constructor injection only).
- Contracts: `ProductIdentificationPipelineRequest`, `ProductIdentificationPipelineResult`, `ProductIdentificationRunId`.

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

Production runs start from **`ProductIdentificationQueryContext`** (`TYPED_QUERY_CONTEXT`). Full natural-language query understanding is **not** implemented in 5C12; `RAW_QUERY` is reserved for future Real E2E.

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
