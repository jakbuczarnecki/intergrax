# VPI lexical retrieval (5C6C)

Canonical production lexical retrieval uses the indexed BM25 adapter:

- `storage_bootstrap/adapters/postgresql/lexical_search_adapter.py`
- composition: `retrieval/composition.py` → `build_lexical_candidate_search`

## Backend

PostgreSQL derived-state inverted index:

- `vpi_lexical_document` — one row per source offer lexical document
- `vpi_lexical_posting` — `(term → offer posting)` with B-tree index on `term`

Queries resolve postings by indexed `term = ANY(%s)` lookup, then score with Okapi BM25 in the adapter.

## Lexical document source

Derived from Data Pack `record_json` via:

- `derive_search_representation()` → `LexicalSearchRepresentation`
- `flatten_lexical_text()` (title, brand, description, structured fragments)

Identity preserved as `SourceRecordRef` (`catalog_id`, `offer_id`, `source_revision`).

## Score semantics

`LexicalChannelScore.bm25_score` carries **real Okapi BM25** (unnormalized).

Lexical retrieval is **recall only** — high score is not product verification.

## Legacy

`integrations/catalog_store/postgresql/catalog_search_adapter.py` → `PostgreSQLLexicalSearchAdapter`

**LEGACY / REFERENCE ONLY** (`record_json ILIKE`, fake `query.limit - rank` score). Not wired by `build_lexical_candidate_search`.

## Token policy

- case-insensitive via Unicode casefold
- hyphen/alphanumeric model tokens preserved (`MZ-V9P2T0BW`, `ABC-123-XY`)
- no NL stop-word list in scenario BM25 engine
