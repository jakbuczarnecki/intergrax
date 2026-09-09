# VPI vector retrieval (5C6E)

Canonical production vector retrieval uses the Qdrant-backed adapter:

- `integrations/search_store/qdrant_vector_candidate_search_adapter.py`
- composition: `retrieval/composition.py` → `build_vector_candidate_search`

## Provider-neutral boundary

Application depends only on `VectorCandidateSearchPort`. Provider construction lives in
`retrieval/composition.py` (Qdrant today via platform `VectorStore`).

## Query embedding

- Same configured embedding identity as stored document vectors (`VpiEmbeddingConfiguration`).
- Query text is embedded **once** per search via `EmbeddingExecutionPort.embed_batch`.
- Document token-budget policy (`vpi-bge-m3-document-token-budget-768-v1`) applies to
  **document** embedding only; query paths remain unaffected (E5A evidence).
- Query vector dimension must equal `expected_dimension` — mismatch fails closed.

## Score semantics

`VectorChannelScore.cosine_similarity` carries **cosine similarity** in `[-1.0, 1.0]`.

Qdrant cosine metric returns provider-side top-K with cosine-compatible scores (no Python
re-ranking over the full corpus). Scores are not clamped, normalized, or converted to
verification confidence.

Vector similarity is **recall evidence only** — never product identity or verification.

## Source identity

Vector hits decode strict typed identity from storage bootstrap payload keys:

- `catalog_id`
- `offer_id`
- `source_revision` (optional)

No configured-catalog fallback. Missing or malformed identity → typed `INVALID_QUERY` failure.

## Runtime support

| Backend | Storage bootstrap | Runtime query |
| --- | --- | --- |
| Qdrant | qualified (5C5C) | **supported** via `VectorStore` |
| PostgreSQL + pgvector | qualified (5C5C2) | **PGVECTOR_RUNTIME_QUERY_GAP** — bootstrap writes scenario-owned tables; platform `VectorStore` does not query them |

## Index identity compatibility gate

Before the first provider vector query, `QdrantVectorCandidateSearchAdapter` runs a
lazy one-time compatibility gate (`VectorIndexCompatibilityGate`):

1. resolve actual index identity from Qdrant administration + typed payload probes;
2. compare against expected identity from VPI configuration and Data Pack/bootstrap
   contracts;
3. fail closed on any mismatch or missing required metadata.

**Expected identity source (authoritative):**

- `VpiEmbeddingConfiguration` for provider, model, dimension;
- `VPI_DATA_PACK_MANIFEST_PATH` manifest `embedding_identity` + `content_identity`, or
  `VPI_EMBEDDING_MODEL_REVISION` (+ optional `VPI_DATA_PACK_CONTENT_IDENTITY`);
- same `ExpectedVectorIdentity` helper as storage bootstrap (`expected_vector_identity_from_embedding_configuration`);
- target identity: `VectorIndexIdentity(logical_name=collection_name, tenant_id=qdrant tenant)`.

**Actual identity source (authoritative):**

- `VectorIndexAdministration.describe_index` — existence, reachability, dense dimension;
- Qdrant collection config — distance metric (cosine required);
- durable index metadata point (`vpi:__index_identity_metadata__`) written during
  storage bootstrap `prepare_target` when Data Pack manifest is bound;
- fallback probe: first stored vector payload embedding fields (no content identity).

**Compared fields:** target, provider, model, revision, dimension, metric, content identity
(when expected).

**Cache:** successful compatibility only; transient provider failures are not cached.

**UNKNOWN != COMPATIBLE** — missing required identity metadata fails closed.

## pgvector runtime gap

PostgreSQL + pgvector bootstrap remains qualified; runtime query support is still
**PGVECTOR_RUNTIME_QUERY_GAP** (unchanged in R1).

## Legacy

`integrations/search_store/platform_vector_search_adapter.py`

**LEGACY / REFERENCE ONLY** — thin delegate to the canonical adapter for proof-50 runtime.
Not wired by `build_vector_candidate_search`.
