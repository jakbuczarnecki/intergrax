# Reference Provider Decision

**Task:** VPI-IMPLEMENTATION-3  
**Audit revision:** `development` @ `d96c97dd766c038f68d7f57f5915c1462fdeaef1`  
**Scope:** provider selection only — no ingest, schema, indexes, adapters, or embeddings.

---

## Decision

**Reference deployment (canonical reproduction path):**

| Role | Provider |
| --- | --- |
| Immutable catalog / source truth | **PostgreSQL** |
| Lexical candidate retrieval (derived) | **Qdrant** |
| Semantic vector candidate retrieval (derived) | **Qdrant** |

**PgVector** remains a **supported alternative** for deployments that accept dense-only vector search and platform gaps at 3.77M scale (see below). It is **not** the canonical VPI reproduction path.

**Embedding model/provider:** selected in VPI-IMPLEMENTATION-4 — see [`EMBEDDING_PROVIDER_DECISION.md`](EMBEDDING_PROVIDER_DECISION.md) (`hf` + `BAAI/bge-m3` @ 1024, swappable via `VPI_EMBEDDING_*`).

---

## Requirements

Reference stack must be:

- real and runnable locally (Docker-friendly, no mandatory SaaS),
- reproducible by a public operator,
- credible for **3,770,377** selected WDC offers,
- aligned with existing Intergrax integration boundaries,
- provider-neutral at the VPI **application** layer (`application/ports/*`),
- bootstrapable automatically in a later task.

VPI keeps four **logical** retrieval channels (`EXACT`, `LEXICAL`, `STRUCTURED`, `VECTOR`). Native provider hybrid (e.g. Qdrant RRF) may exist as an optimization later; it must not replace per-channel attribution.

---

## Existing Intergrax capabilities

Audited on current `development` branch (code + live-qual where present):

| Provider | Intergrax status | Integration entry | Live qualification |
| --- | --- | --- | --- |
| PostgreSQL | **STABLE** (`catalog_manifests.POSTGRESQL`) | `create_postgresql_relational_store()` → `RelationalStore` (`execute` / `fetch_all`) | N/A (generic SQL facade) |
| MySQL | **BETA** (`mysql/manifest.py`) | `create_mysql_relational_store()` — same `RelationalStore` contract | N/A |
| DuckDB / SQLite | **STABLE** | Lab / single-file relational facades | N/A — not million-scale catalog targets |
| Qdrant | **STABLE** (`catalog_manifests.QDRANT`) | `QdrantVectorStore` + `LexicalHybridSupport` | Unit/integration tests; **no** dedicated live-qual gate like PgVector |
| PgVector | **STABLE** (`catalog_manifests.PGVECTOR`) | `PgVectorRagStore` | `tests/integration/rag/vectorstore/test_pgvector_live_qualification.py` (50-doc soak) |
| Weaviate | Registered (`bootstrap_extended`) | `WeaviateVectorStore` — native hybrid when client present | No VPI-scale qual |
| Chroma | Registered | `ChromaVectorStore` — **dense only**, no `LexicalHybridSupport` | No VPI-scale qual |

VPI application code (`application/domain`, `application/ports`, `application/catalog`) contains **no** `psycopg`, `qdrant`, or `pgvector` imports (enforced by unit tests).

---

## Provider comparison

Legend: ✓ capable · ~ partial · ✗ missing or blocked for VPI reference path.

| Criterion | PostgreSQL | Qdrant | PgVector | MySQL | DuckDB/SQLite | Weaviate | Chroma |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Integration status | STABLE | STABLE | STABLE | BETA | STABLE (lab) | extended | extended |
| B. Local reproducibility | ✓ Docker | ✓ localhost:6333 default | ✓ same DB as PG | ✓ | ✓ file/local | needs server | http/embedded |
| C. Bulk ingest suitability | ~ generic SQL; VPI schema TBD | ✓ batched upsert (`batch_size=256`) | ✗ row-by-row `INSERT` loop | ~ same as PG pattern | ✗ not canonical scale | ~ | ~ batch API |
| D. Million-scale indexing | ✓ with scenario indexes | ✓ ANN + sparse index support | ✗ no HNSW/IVFFlat; sequential distance sort | ✓ with indexes | ✗ | ~ | ~ |
| E. Exact lookup | ✓ (scenario-owned tables) | ✗ not source store | ~ payload equality only | ✓ (scenario-owned) | ~ dev only | ✗ | ✗ |
| F. Structured filtering | ✓ SQL + indexes | ~ payload equality + `IN` | ~ JSONB `@>` equality only; **rejects** `IN` | ✓ SQL | ~ | ~ | ~ metadata |
| G. Lexical / BM25 | ✗ not in platform PG provider | ✓ `LexicalHybridSupport` + optional `INTERGRAX_RAG_QDRANT_SPARSE` | ✗ no FTS/BM25/`query_hybrid` | ✗ | ✗ | ✓ native hybrid (ops cost) | ✗ dense only |
| H. Dense vectors | via PgVector only | ✓ `query()` cosine ANN | ✓ cosine `<=>` | separate vector backend | ✗ | ✓ | ✓ |
| I. Metadata filtering | N/A (relational) | ✓ equality + membership | ✓ equality; no membership | N/A | N/A | ✓ | ~ |
| J. ABI reuse for VPI | `RelationalStore` + scenario SQL | `VectorStore` / `VectorStoreRecord` / `MetadataFilter` | same vector ABI, gaps above | `RelationalStore` | lab only | vector ABI | vector ABI |
| K. Operational complexity | moderate | moderate (2nd service) | lower if single DB — but gaps | moderate (2 services) | low | higher | moderate |
| L. Application domain changes | none — adapters only | none — adapters only | none | none | none | none | none |
| M. Public bootstrap UX | ✓ compose-friendly | ✓ compose-friendly | ~ blocked by ingest/lexical/ANN gaps | ✓ if qualified | ✗ scale | ~ | ~ |

---

## Responsibility matrix

| Capability | Reference provider | Reason | Alternative |
| --- | --- | --- | --- |
| Immutable source truth (`record_json`) | PostgreSQL | Durable relational store; platform `RelationalStore` STABLE; scenario-owned schema next | MySQL + scenario schema (BETA catalog integration) |
| `SourceRecordRef` resolution | PostgreSQL | Exact primary-key / catalog identity lookup on source table | MySQL equivalent |
| Exact identifier lookup (GTIN/MPN/SKU/productID) | PostgreSQL | Typed indexes on normalized identifiers; not a vector-store concern | MySQL + same adapter pattern |
| Structured attribute search | PostgreSQL | SQL + B-tree/GiN on normalized attribute subset | MySQL + same pattern |
| Catalog metadata (version, counts, checksum) | PostgreSQL | Bootstrap validation and READY status | MySQL |
| Lexical candidate retrieval | Qdrant | `LexicalHybridSupport` BM25; durable sparse via `INTERGRAX_RAG_QDRANT_SPARSE` | PgVector: **not available**; PG FTS: scenario-built (extra work) |
| Semantic vector retrieval | Qdrant | Dense ANN, batched ingest, metadata filters | PgVector (dense only; scale/ingest gaps) |
| Identity / evidence logic | Application | Provider-independent (`SourceRecordFetchPort`, verifier) | — |
| Verification | Application | Reads source fields from PostgreSQL | — |
| Multi-channel fusion | Application | Four independent channels; not provider hybrid | — |

---

## Reference architecture

```text
WDC selected_offers.parquet
        │
        ▼
  [bootstrap ingest — next task]
        │
        ├──────────────────────────────┐
        ▼                              ▼
 PostgreSQL (source truth)         Qdrant (derived search)
  • record_json immutable           • dense vectors (semantic channel)
  • identifier tables               • sparse/BM25 lexical (lexical channel)
  • structured attr columns         • payload: SourceRecordRef + channel fields
  • catalog metadata
        │                              │
        └──────────┬───────────────────┘
                   ▼
         VPI application ports (provider-neutral)
   ExactIdentifierLookupPort ──► PostgreSQL adapter
   StructuredCandidateSearchPort ► PostgreSQL adapter
   LexicalCandidateSearchPort ──► Qdrant adapter (lexical path only)
   VectorCandidateSearchPort ───► Qdrant adapter (dense query only)
   SourceRecordFetchPort ───────► PostgreSQL adapter
                   ▼
         fusion / verification (application — later tasks)
```

**Platform vs scenario ownership:**

- **Platform:** PostgreSQL connection (`RelationalStore`), Qdrant vector store (`VectorStore` contract).
- **Scenario-owned (next task):** VPI table DDL, bootstrap scripts, port adapters, collection config (including embedding dimension placeholder).

---

## Provider-neutral boundary

Application depends only on:

- `ExactIdentifierLookupPort`
- `LexicalCandidateSearchPort`
- `StructuredCandidateSearchPort`
- `VectorCandidateSearchPort`
- `SourceRecordFetchPort`

Forbidden: `application/scenario.py` or `application/domain/*` importing `psycopg`, `qdrant_client`, or `pgvector`.

Provider choices live in **integrations**, **runtime composition**, and **scenario bootstrap/adapters** (not yet implemented).

---

## Reproduction model

Target operator flow (design — compose not implemented yet):

1. Obtain VPI dataset artifact (`selected_offers.parquet`).
2. Start local **PostgreSQL** + **Qdrant** (eventual Docker Compose).
3. Run scenario bootstrap CLI.
4. Bootstrap creates scenario PostgreSQL schema + Qdrant collection(s).
5. Bootstrap ingests ~3.77M offers (source → PG; derived → Qdrant).
6. Bootstrap validates counts / version / checksum → status **READY**.
7. Scenario runs against full catalog.

Environment knobs (existing): `INTERGRAX_POSTGRESQL_*`, `INTERGRAX_QDRANT_*`, `INTERGRAX_RAG_QDRANT_SPARSE`, embedding dimension TBD.

---

## Alternatives

| Configuration | When valid | Caveats |
| --- | --- | --- |
| **PostgreSQL + Qdrant** (reference) | Canonical public reproduction | Two services; scenario adapters required |
| PostgreSQL + PgVector | Dense-only deployments; smaller corpora | No lexical on provider; no ANN index; row ingest; not qualified at 3.77M |
| MySQL + Qdrant | Operator prefers MySQL catalog | MySQL integration **BETA**; same Qdrant derived layer |
| PostgreSQL + PgVector + scenario FTS | Single-database ops priority | Extra scenario work for lexical; PgVector scale extensions still needed |

Alternatives are valid only when they implement the same **port contracts**. Do not claim capabilities the repo does not ship.

---

## Known gaps / risks

| Gap | Owner | Impact |
| --- | --- | --- |
| VPI PostgreSQL schema / indexes / bootstrap | Scenario (next task) | Blocks ingest |
| VPI Qdrant collection bootstrap + adapters | Scenario (next task) | Blocks retrieval |
| Embedding provider + dimension | **Resolved (config)** — bootstrap enforcement next | Qdrant collection size fixed at bootstrap |
| Qdrant lexical at 3.77M without sparse | Bootstrap config | Enable `INTERGRAX_RAG_QDRANT_SPARSE=true` for durable lexical; in-process BM25 alone is not cold-start durable |
| PgVector HNSW/IVFFlat, bulk ingest, lexical | Platform | Blocks PgVector-as-reference at full scale |
| Qdrant no live-qual gate at million scale | Platform / benchmark | Correctness proven at unit level; scale is architectural fit, not yet benchmarked |
| MySQL BETA status | Platform | Secondary catalog alternative only |
| `DATASET_DISTRIBUTION` unresolved | Scenario | Blocks public reproduction regardless of providers |

---

## Decision for next task

**VPI-IMPLEMENTATION-5: Reusable Storage Bootstrap & Ingest**

Deliver:

- scenario-owned PostgreSQL schema + bootstrap (source truth, exact/structured),
- scenario-owned Qdrant collection bootstrap (lexical + dense derived representations),
- ingest pipeline wiring from `DerivedOfferSearchRepresentation`,
- embedding dimension validation against `VPI_EMBEDDING_*` configuration,
- validation gate (counts, checksum, READY),
- reference adapters behind existing ports.

Embedding reference configuration: [`EMBEDDING_PROVIDER_DECISION.md`](EMBEDDING_PROVIDER_DECISION.md).

Do **not** start: fusion, verification, proof evaluator, or full-corpus embedding generation.
