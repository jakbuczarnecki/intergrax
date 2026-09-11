# VPI Query Understanding

Raw user product-identification requests are converted into one immutable
`ProductIdentificationQuery` before the existing 5C12 production pipeline runs.

## Boundary

| Allowed | Forbidden |
|--------|-----------|
| Parse and normalize user text | Retrieval execution |
| Populate `verification_context` and `search_text` | PostgreSQL / Qdrant / Data Pack |
| Deterministic identifier and attribute extraction | Identity hypotheses / verification |
| Optional typed interpreter port (no vendor SDK in core) | Clarification question selection |
| Provenance spans and typed issues | Benchmark truth / `cluster_id` |

## Raw input

`RawProductIdentificationRequest` (`frozen`, `slots`) carries `raw_text` only
(optional `correlation_id` for tracing). Max length: `MAX_RAW_QUERY_CHARS` (4096).

## Deterministic-first flow

1. Labelled and structural GTIN identifier extraction (reuse `normalize_exact_lookup_value`).
2. Explicit capacity / interface / memory_type / ECC and negative phrases.
3. Optional `ProductIdentificationQueryInterpreter` (protocol) for future NL — default `None`.
4. `DeterministicQueryUnderstandingMergePolicy` — deterministic evidence wins.
5. `ProductIdentificationQuery` with whitespace-normalized `search_text` (no paraphrase).

## Hard vs soft vs negative

- **Hard required:** explicit tokens (e.g. `2TB`, `NVMe`, `DDR5`, `ECC` without negation).
- **Negative:** `not` / `without` / `no` / `except` + supported attribute value.
- **Soft:** `prefer` / `ideally` / `preferably` → `soft_preferences` only.
- Unlabeled codes (e.g. `ABC-123`) are **not** guessed as MPN/SKU.

## Provenance

`QuerySourceSpan` stores offsets plus optional bounded fragment. Extracted records
retain raw and normalized values and a normalization rule label.

## Conflicts and failure

Query understanding failures are typed `QueryUnderstandingIssue` codes (not verification
outcomes). Conflicting constraints or GTINs block query emission.

## Composition

`build_product_identification_query_understanding_service()` wires deterministic
extractors and merge policy. Thin orchestration: `ProductIdentificationApplicationService`.

## Scope limits (current)

- English-oriented explicit patterns; language-neutral tokens (GTIN, MPN, SKU, units).
- Narrow attribute vocabulary aligned with 5C12 tests (`capacity`, `interface`, `memory_type`, `ecc`).
- No brand/model structured fields — remain in `search_text`.
- `pipeline_input_origin=RAW_QUERY` on understanding result; 5C12 pipeline contract unchanged.

## Future

Optional LLM adapter behind `ProductIdentificationQueryInterpreter`, multilingual NL,
and host-level raw→pipeline wiring with observability `RAW_QUERY` stage handoff.
