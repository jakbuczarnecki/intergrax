# ADR-LLM-002: Central ModelCatalog and context window resolution

**Status:** Accepted (2026-06-14)  
**Phase:** M-LLM-X  
**Context:** Deep LLM adapter audit (2026-06-14) — per-adapter context dicts stale; new vendor models (Opus, Fable, OpenRouter ids) break Nexus budgeting.

## Decision

Introduce a Tier-0 **`ModelCatalog`** with deterministic **`resolve_context_window_tokens(provider, model, options)`** as the **single** context-window source for all `LLMAdapter` instances. Bundled YAML + optional operator overlay; profile override remains authoritative.

## Resolution order (deterministic)

1. `LLMProfile.options["context_window_tokens"]` — operator override (wins always).
2. `ModelCatalog` exact match on `model_id`.
3. `ModelCatalog` prefix rules (`claude-*`, `gpt-*`, `gemini-*`, `anthropic.*`, `meta-llama/*`, …).
4. Provider-family default from catalog (e.g. `claude_default: 200_000`).
5. Safe conservative default per provider slug — emit **`ModelCatalogMissDiagV1`** once per model/run.
6. **Deprecated (remove over M-LLM-X):** inline per-adapter `_CONTEXT_WINDOWS` dict lookups.

Optional: gateway metadata fetch (OpenRouter `/models`) merges into session cache when `fetch_gateway_metadata=True` — does not bypass steps 1–2.

## ModelRecord (minimum)

| Field | Required |
|-------|----------|
| `model_id` | yes |
| `context_window_tokens` | yes |
| `supports_vision`, `supports_tools`, `supports_structured_output` | no (default false / true) |
| `provider_hints` | no |

## Rationale

- **New models are free strings** — API calls work without platform releases; **budgeting** must not depend on hardcoded adapter dicts updated by hand.
- Nexus `context_preflight`, `engine_history_layer`, and `resolve_input_budget_tokens` read `adapter.context_window_tokens` — wrong values cause silent history trim or API overflow.
- Bedrock prefix heuristics proved the pattern; generalize to catalog prefix rules for all families.
- Ollama-only `context_window_tokens=` constructor override is insufficient — override must flow from **`LLMProfile`** for every provider.

## Token accounting (paired decision)

When an `LLMAdapter` is in scope, preflight and history compression **must** use `adapter.count_messages_tokens(messages)` — not `chars // 4`. See M-LLM-X.3.

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

- [architecture/LLM_ADAPTERS.md](../../../architecture/LLM_ADAPTERS.md) § Model catalog
- [plan/LLM_ADAPTERS.md](../../../plan/LLM_ADAPTERS.md) Phase M-LLM-X
- [ADR-LLM-001](../2026-06-06/ADR-LLM-001.md) — envelope + two-layer usage (unchanged)
