# ADR-LLM-002: Central ModelCatalog and context window resolution

**Status:** Accepted (2026-06-14)  
**Phase:** M-LLM-X  
**Context:** Deep LLM adapter audit (2026-06-14) - per-adapter context dicts stale; new vendor models (Opus, Fable, OpenRouter ids) break Nexus budgeting.

## Decision

Introduce a Tier-0 **`ModelCatalog`** with deterministic **`resolve_context_window_tokens(provider, model, options)`** as the **single** context-window source for all `LLMAdapter` instances. Bundled YAML + optional operator overlay; profile override remains authoritative.

## Resolution order (deterministic)

1. `LLMProfile.options["context_window_tokens"]` - operator override (wins always).
2. `ModelCatalog` exact match on `model_id` - **no** catalog-miss diagnostic.
3. `ModelCatalog` prefix rules (`claude-*`, `gpt-*`, `gemini-*`, `anthropic.*`, `meta-llama/*`, …) - emit **`ModelCatalogMissDiagV1`** with `resolution_tier=prefix_rule` once per model/run.
4. Optional gateway metadata session merge when `fetch_gateway_metadata=True` - **no** miss when API returns a value; does not bypass steps 1–2.
5. **Deprecated:** inline per-adapter `_CONTEXT_WINDOWS` dict lookups (legacy fallback only).
6. Provider-family default from catalog (e.g. `openrouter: 128_000`) - emit miss with `resolution_tier=provider_default`.
7. Global `fallback_default` from catalog YAML - emit miss with `resolution_tier=fallback_default`.

**Amended (2026-06-19 · M-LLM-X.15):** catalog-miss diagnostics fire on **any non-exact** resolution tier (steps 3, 6, 7), not only on global fallback. Plane A trace step `llm_catalog_miss`, Prometheus `intergrax_llm_catalog_miss_total`, wired from `RuntimeState.configure_llm_tracker()` regardless of core adapter presence.

## ModelRecord (minimum)

| Field | Required |
|-------|----------|
| `model_id` | yes |
| `context_window_tokens` | yes |
| `supports_vision`, `supports_tools`, `supports_structured_output` | no (default false / true) |
| `provider_hints` | no |

## Rationale

- **New models are free strings** - API calls work without platform releases; **budgeting** must not depend on hardcoded adapter dicts updated by hand.
- Nexus `context_preflight`, `engine_history_layer`, and `resolve_input_budget_tokens` read `adapter.context_window_tokens` - wrong values cause silent history trim or API overflow.
- Bedrock prefix heuristics proved the pattern; generalize to catalog prefix rules for all families.
- Ollama-only `context_window_tokens=` constructor override is insufficient - override must flow from **`LLMProfile`** for every provider.

## Token accounting (paired decision)

When an `LLMAdapter` is in scope, preflight and history compression **must** use `adapter.count_messages_tokens(messages)` - not `chars / 4`. See M-LLM-X.3.

## Non-goals

- Central LLM gateway microservice (separate platform ADR).
- Vendor-native tokenizer plugins for every provider (deferred; tiktoken + SDK usage counts remain acceptable for budgeting).
- Model **pricing** in catalog v1 (cost routing uses existing metrics + AHI; pricing table optional later).

## Consequences

- All 19 adapters call resolver at construction; inline dicts shrink then delete.
- `INTERGRAX_LLM_MODEL_CATALOG_PATH` for operator YAML overlay.
- CI: `check_model_catalog_coverage.py` warns when default models missing from bundled YAML.
- Cross-domain: `CONTEXT_ENGINEERING` preflight (CE-LLM-X) depends on this ADR.

## References

- [architecture/LLM_ADAPTERS.md](../../../../architecture/LLM_ADAPTERS.md) § Model catalog
- [plan/LLM_ADAPTERS.md](../../../../maintainers/plans/LLM_ADAPTERS.md) Phase M-LLM-X
- [ADR-LLM-001](../2026-06-06/ADR-LLM-001.md) - envelope + two-layer usage (unchanged)
