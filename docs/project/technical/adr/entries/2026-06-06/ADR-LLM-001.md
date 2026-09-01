# ADR-LLM-001: Typed LLM adapter response envelope

**Status:** Accepted (2026-06-06)  
**Phase:** M-LLM-R  
**Context:** Tier-0 LLM adapter audit (2026-06-06)

## Decision

All `LLMAdapter` completion methods return a strongly typed **`LLMAdapterResponse`** envelope (or streaming/structured variants derived from it) instead of bare `str` or untyped `Dict[str, Any]`.

## Rationale

- Production observability requires `finish_reason`, per-call token usage, provider correlation ids, and refusal signals at the call site - not only via side-channel usage logs.
- Replay and cost attribution need synchronous access to per-call metadata when building traces and `LLMCallInfo`.
- Tool calls must be typed (`LLMToolCall`) to avoid dict parsing scattered across Nexus and agents.
- Extensibility uses `LLMProviderExtensions` (tagged optional slices), not open dict bags.

## Contract summary

| Method | Return type |
|--------|-------------|
| `generate_messages` | `LLMAdapterResponse` |
| `generate_with_tools` | `LLMAdapterResponse` |
| `stream_messages` / `stream_with_tools` | `Iterable[LLMStreamEvent]` |
| `generate_structured` | `LLMStructuredResult[T]` |

Primary text field: **`content: str`** (alias **`text`**).

## Two-layer usage model (unchanged)

1. **Per call:** `LLMAdapterResponse.usage` (`LLMTokenUsage`) - source of truth for the caller.
2. **Per run:** `LLMAdapter.usage` (`LLMAdapterUsageLog`) + runtime `LLMUsageTracker` - aggregation only; integers must match per-call usage recorded in `end_call`.

## Consequences

- All provider adapters and consumers migrate to `.content` / `.tool_calls`.
- `make_tool_result` removed; use `build_adapter_response` from `_shared/adapter_response_builders.py`.
- CI guards: `scripts/maintenance/check_llm_adapter_typed_returns.py`, `scripts/maintenance/check_agents_llm_adapter_response.py`.
- Replay: `CoreLLMCallRecordedDiagV1` trace payload + `trace_replay_bridge.py` → `LLM_CALL` DTOs.

## References

- [architecture/LLM_ADAPTERS.md](../../architecture/LLM_ADAPTERS.md) § Response envelope
- [intergrax_runtime_architecture.md](../../intergrax_runtime_architecture.md) Phase M-LLM-R
