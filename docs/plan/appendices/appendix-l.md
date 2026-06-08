**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Appendix L — LLM completion response envelope traceability (Phase M-LLM-R)

**Source:** Tier-0 LLM adapter audit (2026-06-06) — plain `str` / `Dict[str, Any]` returns insufficient for production observability, replay, cost attribution, and L4 adaptive signals.

**Phase register:** [Phase M-LLM-R](#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06) · **Band 2z** · queue [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

### L.1 Audit finding → remediation map

| # | Audit finding | Remediation | Task IDs |
|---|---------------|-------------|----------|
| 1 | `generate_messages` returns bare `str` | `LLMAdapterResponse` with `content: str` | M-LLM-R.1.1, M-LLM-R.2.1, M-LLM-R.3.*, M-LLM-R.4–6.* |
| 2 | `generate_with_tools` returns `Dict[str, Any]` | Same envelope; `tool_calls: tuple[LLMToolCall, ...]` | M-LLM-R.1.3, M-LLM-R.1.7, M-LLM-R.2.2, M-LLM-R.4.2 |
| 3 | Streaming yields `str` / dict chunks | `LLMStreamEvent` partial/final | M-LLM-R.1.5, M-LLM-R.2.3–2.4, M-LLM-R.3.6 |
| 4 | `generate_structured` return untyped | `LLMStructuredResult[T]` | M-LLM-R.1.6, M-LLM-R.2.5, M-LLM-R.3.7 |
| 5 | SDK `finish_reason` / stop metadata lost | `LLMFinishReason` on response | M-LLM-R.1.1, M-LLM-R.3.1–3.4 |
| 6 | Provider `response_id` / request correlation lost | `response_id: str \| None` on response | M-LLM-R.1.1, M-LLM-R.3.1 |
| 7 | Cached / reasoning tokens discarded | `LLMTokenUsage.cached_input_tokens`, `reasoning_tokens` | M-LLM-R.1.2, M-LLM-R.3.1 |
| 8 | Refusal / content-filter signals lost | `refusal: str \| None` + finish_reason enum | M-LLM-R.1.1, M-LLM-R.3.1–3.2 |
| 9 | Usage only via side-channel (`LLMAdapterUsageLog`) | Per-call `usage` on response + aligned `end_call` | M-LLM-R.1.2, M-LLM-R.2.6, M-LLM-R.7.1 |
| 10 | Inconsistent token counting (estimate vs SDK) | Prefer SDK counts; flag estimate in `LLMProviderExtensions` | M-LLM-R.3.5, M-LLM-R.1.4 |
| 11 | No extensibility without dict bags | `LLMProviderExtensions` tagged union | M-LLM-R.1.4 |
| 12 | Replay `LLMCallInfo` not populated from adapter | Trace bridge from `LLMAdapterResponse` | M-LLM-R.7.2, M-LLM-R.7.3 |
| 13 | `CoreLLMAdapterReturnedDiagV1` tracks `adapter_return_type="str"` | Diagnostics carry finish_reason + tokens | M-LLM-R.7.4 |
| 14 | Conformance enforces `isinstance(text, str)` | Typed conformance helpers | M-LLM-R.8.2 |
| 15 | ~50 call sites assume `str` | Full consumer refactor (Nexus, RAG, agents, websearch) | M-LLM-R.4.*, M-LLM-R.5.*, M-LLM-R.6.* |
| 16 | `make_tool_result` dict factory | Delete; typed `build_adapter_response` | M-LLM-R.1.7 |
| 17 | Public API missing response types | Re-export from `llm_adapters/__init__.py` | M-LLM-R.1.8 |
| 18 | Docs describe two-layer usage but not response envelope | `architecture/LLM_ADAPTERS.md` envelope section | M-LLM-R.8.1 |
| 19 | No CI guard against regression to `str` returns | `check_llm_adapter_typed_returns.py` | M-LLM-R.8.3 |

### L.2 Consumer inventory (must migrate)

| Area | Modules | Task |
|------|---------|------|
| Nexus core LLM | `core_llm_step.py` | M-LLM-R.4.1 |
| Tool planning | `tool_planning_service.py` | M-LLM-R.4.2 |
| Planning / history | `plan_sources.py`, `engine_history_layer.py` | M-LLM-R.4.3 |
| Profile services | `user_profile/*`, `organization/*`, `session_memory_consolidation_service.py` | M-LLM-R.4.4 |
| Supervisor | `supervisor.py` | M-LLM-R.4.5 |
| RAG | `query_refiner.py`, `query_expander.py`, `chunk_enricher.py`, `llm_graph_indexer.py` | M-LLM-R.5.1 |
| Websearch | `websearch_context_generator.py`, `websearch_answerer.py` | M-LLM-R.5.2 |
| Legacy RAG | `legacy/rag_answers/pipeline/answer_pipeline.py` | M-LLM-R.5.3 |
| Agents (Tier-2) | `agents/*/steps/pipeline.py`, `mock_agents.py` | M-LLM-R.6.1 |
| Scaffold / tests | `scaffold/new_agent.py`, `testing_support/builder.py` | M-LLM-R.6.2–6.3 |
| All providers | `llm_adapters/providers/*` | M-LLM-R.3.* |

### L.3 Paydown log

| Date | M-LLM-R ID | Summary |
|------|------------|---------|
| 2026-06-06 | M-LLM-R.0.1 | Phase M-LLM-R register + §6.1v + §6.2ad + Appendix L + Band 2z |
| 2026-06-06 | M-LLM-R.* | Typed `LLMAdapterResponse` envelope; providers + consumers migrated; gate **755** passed |
| — | — | *(append row per merged PR)* |

---
