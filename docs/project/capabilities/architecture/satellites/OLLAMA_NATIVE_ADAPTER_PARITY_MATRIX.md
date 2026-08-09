<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LCI-6A — Native Ollama Adapter Architecture and Parity Matrix

**Status:** LCI-6B READY_FOR_REVIEW
**Task:** `LCI-6A-NATIVE-OLLAMA-ADAPTER-ARCHITECTURE-AND-PARITY-MATRIX`
**Baseline:** `LangChainOllamaAdapter` in `intergrax/llm_adapters/providers/ollama_adapter.py`
**Implementation status:** LCI-6B implemented behind a non-default path; live proof remains pending.

This satellite freezes the target contract and the observable behavior that LCI-6B must implement and compare. It is an architecture and behavior specification, not a transport implementation, resolver change, LKW cutover, Token Optimization change, packaging change, or LCI-6B kickoff.

## Decision

```text
NativeOllamaAdapter implements the existing Intergrax LLMAdapter ABI.

It does not introduce a second public LLM contract.

Existing message, response, stream, tool, structured-output,
usage, capability and context-window contracts remain authoritative.

LangChainOllamaAdapter remains the parity baseline until LCI-6D cutover.

LCI-6B implements behind a non-default path.

LCI-6C supplies mandatory real-Ollama evidence.

LCI-6D may change the default only when mandatory parity rows are proven.

LCI-6E then moves LangChainOllamaAdapter to compatibility-only packaging.
```

Roadmap state:

```text
LCI-5A — APPROVED
LCI-5B — APPROVED
LCI-5C — APPROVED
RAG-REGRESSION-GATE-1 — PASS / CLOSED

LCI-6A — APPROVED
LCI-6B — READY_FOR_REVIEW
LCI-6C — NEXT AFTER ACCEPTANCE
LCI-6D — PLANNED
LCI-6E — PLANNED
LCI-7 — PLANNED
LCI-8 — PLANNED
```

No claim of production readiness, full parity, or LangChain removal is made here. Live proof remains pending.

## Evidence boundary and baseline

The baseline is code behavior, with tests used to confirm the behavior. Where a test, architecture statement, and implementation differ, **CODE_BEHAVIOR_IS_BASELINE**.

| Evidence | Baseline fact |
|---|---|
| `intergrax/llm_adapters/contracts/llm_adapter.py` | `LLMAdapter` is the public execution ABI; messages are `ChatMessage`; optional surfaces have defined defaults. |
| `intergrax/llm_adapters/contracts/adapter_response.py` and `_shared/adapter_response_builders.py` | Responses and stream events are typed envelopes. |
| `intergrax/llm_adapters/providers/ollama_adapter.py` | Ollama currently crosses the provider boundary through `ChatOllama`. |
| `intergrax/llm_adapters/providers/ollama_capabilities.py` | Capability resolution is lazy per adapter, cached per instance, manually refreshable, and fail-closed. |
| `intergrax/llm_adapters/providers/_ollama_schema.py` | Ollama receives a projected provider-compatible JSON Schema; the original model remains the validation contract. |
| `intergrax/llm_adapters/registry/context_window.py` | Context windows use the shared deterministic resolver and constructor override. |
| `tests/unit/llm_adapters/test_ollama_tool_calling.py` | Tool shape, choice validation, message mapping, capability behavior, and usage behavior. |
| `tests/unit/llm_adapters/test_ollama_structured_output.py` | JSON Schema path, raw/parsed result handling, revalidation, errors, and options. |
| `intergrax/runtime/token_optimization/llm_router.py` and its unit tests | Token Optimization consumes capability flags and the typed tool/structured ABI; it does not use a vendor client. |
| `applications/local_workspace_application/model_runtime_proof/stages.py` and `conversation/interaction_planner.py` | LKW proof paths use plain generation, structured output, and non-streaming tools; the planner validates the typed structured result. |

## Public Intergrax ABI

The native adapter must subclass `LLMAdapter` and expose the same methods and return types:

| Surface | Contract |
|---|---|
| Identity | `provider == LLMProvider.OLLAMA`, `model` is the configured model string. |
| Required generation | `generate_messages(messages, temperature=None, max_tokens=None, run_id=None) -> LLMAdapterResponse`. |
| Streaming | `stream_messages(...) -> Iterable[LLMStreamEvent]`. Events are `PARTIAL` or `FINAL`. |
| Streaming capability flag | Current adapter inherits `supports_streaming() == False` despite implementing `stream_messages`; this is code behavior, not a second stream contract. |
| Tools | `supports_tools()` and `generate_with_tools(...) -> LLMAdapterResponse`. |
| Tool streaming | `stream_with_tools(...)` is `NOT_SUPPORTED_CURRENTLY`. |
| Structured output | `supports_structured_output()` and `generate_structured(...) -> LLMStructuredResult[T]`. |
| Budgeting | `context_window_tokens`, `count_messages_tokens`, and the inherited estimator path. |
| Usage | `self.usage.begin_call()` / `end_call()` and `LLMAdapterResponse.usage`. |
| Resilience boundary | Reuse inherited `LLMAdapter._execute()` where the baseline behavior uses it; do not invent a second retry/quota hierarchy. |

`LLMAdapterResponse`, `LLMStreamEvent`, `LLMToolCall`, `LLMStructuredResult`, `LLMTokenUsage`, `LLMProviderExtensions`, `ChatMessage`, response builders, the capability resolver, schema preparation, and context-window resolver remain authoritative existing contracts.

## Message parity

The native adapter must serialize the same model-facing meaning. It must not expose native client message classes outside the provider module.

| `ChatMessage` | Current `ChatOllama` representation | Native target |
|---|---|---|
| `system` | `SystemMessage(content=m.content)` | `{"role": "system", "content": m.content}` |
| `user` | `HumanMessage(content=m.content)` | `{"role": "user", "content": m.content}` |
| `assistant` without calls | `AIMessage(content=m.content)` | `{"role": "assistant", "content": m.content}` |
| `assistant` with calls | `AIMessage(content=m.content, tool_calls=[...])` | `{"role": "assistant", "content": m.content, "tool_calls": canonical native calls}` |
| `tool` | `ToolMessage(content=m.content, tool_call_id=str(m.tool_call_id))` | `{"role": "tool", "content": m.content, "tool_call_id": str(m.tool_call_id)}` |
| unknown/custom role | `SystemMessage(content=f"[ROLE]\n{content}")` | Same explicit system projection; do not send an arbitrary native role. |

Message rules frozen for LCI-6B:

* Empty content is preserved for system, user, assistant, and tool messages. An assistant with calls may have empty text.
* `ChatMessage.content` is a string at this boundary. The current AI response coercion additionally joins string and `{type: "text", text: ...}` blocks and ignores other response blocks; the native response normalizer must preserve this observable text result.
* A tool message without a non-blank `tool_call_id` raises `ValueError` before provider invocation: `tool message requires tool_call_id`.
* Assistant calls accept the current internal shape `{"name", "args", "id"}` and the OpenAI-style shape `{"function": {"name", "arguments"}, "id"}`. Arguments must become a JSON object; malformed JSON raises `ValueError`.
* A call name is required. Tool-call argument JSON is canonicalized into `LLMToolCall.arguments_json` with Unicode preserved.
* Unknown roles are a compatibility projection, not a new public role. The native adapter must not broaden `MessageRole` in LCI-6A.

## Generation options

The native request object does not need to copy LangChain's `options` dictionary. The resulting Ollama behavior must be equivalent.

| Intergrax parameter | Current LangChain mapping | Native Ollama target | Requirement | Test method |
|---|---|---|---|---|
| `temperature` | `options["temperature"]` when non-`None` | `options["temperature"]` when non-`None` | Preserve explicit value; absence remains client/provider default. | Mock request capture; deterministic live request. |
| `max_tokens` | `options["num_predict"]` when non-`None` | `options["num_predict"]` when non-`None` | No rename at the Intergrax boundary; no arbitrary default. | Mock request capture; bounded live generation. |
| constructor `defaults["options"]` | Existing options are copied then per-call values overwrite matching keys | Merge into native `options`, with explicit per-call values taking precedence | Preserve existing option passthrough behavior. | Unit request normalization. |
| `temperature` / `max_tokens` defaults | Parsed into `LLMCallConfig`, and also remain in `defaults`; current Ollama methods do not synthesize provider defaults from `LLMCallConfig` | Preserve effective baseline behavior in 6B; do not broaden option semantics | Any resilience/default change is a separate improvement. | Unit plus side-by-side harness. |
| `timeout_sec` and other call config | `LLMCallConfig` parses them, but plain/stream/structured paths do not apply `_execute`; tools call `_execute` | No arbitrary timeout value. Preserve effective library-owned/pass-through behavior until live evidence and a separate policy decision. | Do not silently claim uniform timeout/retry behavior. | Matrix rows 034–040. |

## Non-streaming generation

`generate_messages()` currently:

1. begins usage with the supplied `run_id` and adapter identity;
2. estimates input tokens using `estimate_tokens_for_messages`;
3. maps `ChatMessage` values at the provider boundary;
4. merges `defaults` with `options.temperature` and `options.num_predict`;
5. invokes `ChatOllama.invoke`;
6. extracts `res.content`, falling back to `str(res)` when content is falsey;
7. estimates output tokens from the returned text;
8. returns `build_adapter_response(...)` with model, provider, usage, and `LLMProviderExtensions(usage_source="estimate")`;
9. ends usage in `finally`, marking success or the concrete exception type.

The native target uses the official Ollama client `chat` operation, with `stream=False`, normalized messages, equivalent `options`, and no prompt rewriting. It extracts assistant text, builds the same Intergrax response envelope, and preserves the baseline error/lifecycle observable contract. Native provider counters are handled by the usage policy below, not by a second response type.

LCI-6B side-by-side acceptance compares request mapping, message sequence, options, response envelope fields, usage shape/source, run lifecycle, and failure outcome. Generated natural-language text is not compared byte-for-byte.

## Streaming

### Event ABI and ordering

The only current stream event contract is:

```text
zero or more:
  LLMStreamEvent(kind=PARTIAL, delta_content=<non-empty text>, response=None)
exactly one on success:
  LLMStreamEvent(kind=FINAL, delta_content="", response=<full envelope>)
```

Empty content chunks are not emitted as partial events. A successful empty response therefore emits only the final event. The final response contains the concatenation of emitted text and estimated usage. No final event is emitted when the call terminates with an exception.

### Current fallback and frozen target

The current adapter calls `chat.stream(...)`. If that operation raises, it calls `chat.invoke(...)` and emits the fallback text as one partial event followed by one final event. This fallback is currently unconditional, including after partial events have already escaped the generator.

That creates a known defect: if `a` was emitted and the stream then fails, an invoke fallback returning `ab` emits `a` followed by `ab`, duplicating content. This is recorded as `CURRENT_BEHAVIOR_DEFECT / PARITY_EXCEPTION`.

The native target decision is:

* retain fallback-to-invoke only when the stream fails before any partial event;
* after any partial event, propagate the stream failure and do not invoke a second completion;
* never emit a final event for the failed partial stream;
* never duplicate already emitted content;
* record the failed call in usage with the input estimate already known;
* do not add a replay or event-retraction mechanism to the public ABI.

This is an intentional, bounded improvement and is not a reason to broaden streaming semantics. It must be covered by a deterministic unit test and a side-by-side failure harness.

Tool streaming remains unsupported in the parity baseline. Any incremental tool-call event support is `POST_PARITY_ENHANCEMENT` and cannot be required for LCI-6B or LCI-6C.

## Tool calling

The baseline supports non-streaming tool calling only when the resolver reports `"tools"` for the installed model. `generate_with_tools()`:

* fails closed with `ValueError` before provider invocation when capability resolution is not resolved or does not include tools;
* accepts `tool_choice=None`, `"auto"`, and `"required"` only;
* raises `ValueError` before provider invocation for `"none"`, named tools, function dictionaries, or any other value;
* passes the supplied tool schema through `bind_tools()` without rewriting;
* maps returned calls to `LLMToolCall(id, name, arguments_json)` while preserving call order;
* returns `LLMFinishReason.TOOL_CALLS` when calls exist and `LLMFinishReason.COMPLETED` otherwise;
* rejects the provider's `invalid_tool_calls` result with `ValueError`;
* returns empty assistant content with valid calls unchanged.

The canonical native representation is:

```text
LLMToolCall(
    id=<provider call id or "">,
    name=<non-blank function name>,
    arguments_json=<JSON object string>,
)
```

The native Ollama request uses the same function schema and native `tools`/`tool_choice` fields supported by the official client. LCI-6B must not add new `tool_choice` values or name-specific semantics. Tool-result continuation sends the preceding assistant call and then a `tool` message with the required matching `tool_call_id`; the adapter does not execute tools or invent a continuation loop.

## Structured output

The baseline structured path:

1. calls `prepare_ollama_generation_schema(output_model)`;
2. passes that projection to `with_structured_output(..., method="json_schema", include_raw=True)`;
3. receives `{raw, parsed, parsing_error}`;
4. propagates `parsing_error`, rejects missing `parsed`, and rejects an unexpected result type;
5. revalidates dictionaries with the original `output_model`;
6. uses raw assistant content when non-empty, otherwise serializes the validated value;
7. returns `LLMStructuredResult(parsed=validated, response=<typed envelope>)`.

The provider generation schema is not the final validation contract. The native target passes the same projected schema as Ollama's `format`/JSON Schema request field, parses raw assistant content, and validates with the original model using the existing Intergrax validation path. A parse failure, schema validation failure, missing result, or malformed provider response is an error; parseable JSON alone is insufficient.

`supports_structured_output()` remains `True` because the current adapter declares the JSON Schema path available. `OllamaModelCapabilities` has no structured-output flag; LCI-6B must not create a second capability system.

## Capabilities and context window

### Capability resolution

Reuse `OllamaModelCapabilityResolver` and `OllamaModelCapabilities`:

* resolution is lazy on first `model_capabilities` access;
* the result is cached on the adapter instance;
* `refresh_model_capabilities()` performs a new resolution;
* empty, missing, malformed, unknown, or unavailable capability data is `resolved=False`, has no claimed capabilities, and fails closed;
* `supports_tools()` is true only for a resolved capability set containing `"tools"`;
* resolver errors expose only an error type, not provider payload details;
* no static model-name allowlist is added.

The resolver already uses the official Ollama client's `show` operation. The native adapter must use the same resolver instance contract and not add TTL, global cache, or a duplicate `/api/show` implementation.

### Context window

The native adapter must call the same `init_adapter_context_window_tokens` contract. Its resolution order is:

1. explicit positive `context_window_tokens`;
2. `ModelCatalog` exact match;
3. `ModelCatalog` prefix rule;
4. optional gateway metadata session;
5. legacy per-adapter mapping;
6. provider-family default;
7. catalog fallback default.

The current `_estimate_ollama_context_window_from_model()` helper is not used by the constructor's canonical path. It is legacy/dead relative to the shared resolver and its name/tag lookup is not a second context contract. It is not removed in LCI-6A.

## Usage accounting

The current adapter uses estimates in all four implemented methods and marks the response extension `usage_source="estimate"`. `LLMTokenUsage.total_tokens` is derived from input plus output. Usage lifecycle is:

```text
begin_call(run_id=run_id, adapter=self)
  estimate input
  provider call
  estimate output
end_call(call, input_tokens, output_tokens, success, error_type)
```

The default run id is `"general"`. `begin_call` attaches provider/model; the usage log aggregates per run and records metrics at `end_call`.

The native target policy is:

```text
use provider-reported prompt_eval_count/eval_count when present,
non-negative, and internally trustworthy;
otherwise use the existing Intergrax estimator;
always expose the source through LLMProviderExtensions.usage_source.
```

This is a `PARITY-SENSITIVE CHANGE`: native provider counts may differ from LangChain-baseline estimates and can affect Token Optimization observations. LCI-6A does not change the current adapter or Token Optimization policy. LCI-6B must prove shape/lifecycle parity; LCI-6C must record provider counters and fallback behavior; LCI-6D must explicitly accept any accounting delta before cutover.

On provider failure, no fabricated output usage may be reported. The baseline has method-specific input accounting on failure: non-streaming, tools, and structured response envelopes have no successful response usage, while the stream `finally` retains the already computed input estimate. The native implementation must preserve this observable lifecycle until a separately approved usage policy changes it.

## Errors and timeout semantics

Intergrax currently has no Ollama-specific exception hierarchy. Concrete LangChain and native-client exception classes may differ; the required observable contract is failure rather than fabricated success, safe error classification, and usage error accounting.

The current timeout value is **CURRENTLY_LIBRARY_OWNED**. `timeout_sec` is parsed by `LLMCallConfig`, but current Ollama methods do not uniformly route through `_execute`; no arbitrary timeout value is specified here. LCI-6B must make an explicit implementation choice without silently claiming that a new timeout or retry policy is baseline parity.

The failure rows in the matrix distinguish deterministic unit injection from behavior that needs a real Ollama proof. No live Ollama call is part of LCI-6A.

## TOKEN OPTIMIZATION PARITY REQUIREMENTS

The direct Token Optimization consumer requires:

* `context_window_tokens` and `count_messages_tokens` from `LLMAdapter`;
* `supports_tools()` to distinguish native-tools transport;
* resolved capability failure to fail closed, without structured fallback;
* resolved no-tools models to use structured fallback only when `supports_structured_output()` is true;
* exactly one expected `LLMToolCall`, valid JSON arguments, and stable `tool_call_id`/name fields;
* the exact tool schema and message sequence supplied by the router;
* no second tokenizer, private client, or Token Optimization usage tracker;
* response usage and adapter usage lifecycle to remain available to existing aggregation.

The native adapter must not change prompt-cache envelope semantics, stable prefix behavior, or router transport selection in LCI-6A.

The LCI-6D acceptance test must run the existing Token Optimization router proof with the native adapter under:

```text
resolved tools        -> NATIVE_TOOLS
resolved no-tools     -> STRUCTURED_OUTPUT when allowed
unresolved capability -> CAPABILITY_RESOLUTION_FAILED / fail closed
native provider error -> LLM_ERROR, no structured fallback
```

## LKW CUTOVER REQUIREMENTS

The inspected direct LKW surfaces use:

* plain `generate_messages()` in the model-runtime proof;
* `supports_structured_output()` and `generate_structured()` for planning, followed by Intergrax request validation and one repair attempt;
* `supports_tools()` and non-streaming `generate_with_tools()` for workspace search, followed by exactly-one-call, tool-name, JSON-argument, and workspace-boundary validation.

No LKW stream path is a required current consumer surface. LCI-6D must prove these paths without changing the resolver in LCI-6A:

1. plain generation returns a typed response and preserves provider/model identity;
2. structured planning returns the requested model type or produces existing failure/repair behavior;
3. workspace search receives one valid `LLMToolCall` with stable arguments and preserves fail-closed validation;
4. capability failures and provider failures remain distinguishable by existing proof diagnostics.

## Native transport target

The recommended transport is the existing official `ollama` Python client, already used by `OllamaModelCapabilityResolver`. No dependency is added in LCI-6A. The native adapter target requires only these operations:

| Operation | Native client responsibility |
|---|---|
| Plain chat | `chat(model, messages, options=..., stream=False)` |
| Streaming chat | `chat(model, messages, options=..., stream=True)` normalized to `LLMStreamEvent` |
| Tools | `tools=...` and frozen `None`/`auto`/`required` choice surface |
| Structured output | `format=<projected JSON Schema>` plus raw content parsing and Intergrax validation |
| Capability metadata | existing resolver's `show(model)` path |

The implementation must keep the client object and provider response objects inside `intergrax/llm_adapters/providers`. It must reuse the existing response builders, usage log, schema projector, capability resolver, and context resolver. No new public ABI, context table, capability cache, or exception hierarchy is required by this architecture.

## LCI-6B side-by-side harness

The bounded harness is:

```text
same ChatMessage/tool/schema request
        ↓
LangChainOllamaAdapter       NativeOllamaAdapter
        ↓                           ↓
normalized observable result
        ↓
deterministic parity assertions
```

The harness must normalize only provider representation details. It must compare:

* serialized message roles, content, assistant calls, and tool-result IDs;
* option mapping and presence/absence of defaults;
* response type, content field shape, finish reason, model/provider;
* tool-call count/order/id/name/JSON-object arguments;
* structured raw/parsed/validation outcomes;
* partial/final event kind and ordering;
* usage fields, source, run id, success/error lifecycle;
* capability resolved state, flags, source, refresh behavior;
* context-window value and resolution override behavior;
* error category and fail-closed behavior.

It must not assert natural-language output byte-for-byte. Live quality/content parity is evaluated semantically only in the bounded LCI-6C proof where it has meaning. The harness must explicitly test stream failure-before-partial and failure-after-partial cases so that the known duplication defect is not copied.

## Acceptance matrix

Classification is exactly one of: `MUST_MATCH`, `INTENTIONAL_IMPROVEMENT`, `CURRENTLY_UNSUPPORTED`, `POST_PARITY_ENHANCEMENT`, `LIVE_PROOF_REQUIRED`.

`unit proof` and `integration proof` are LCI-6B proof locations. `live proof required` names the LCI-6C requirement rather than asserting it has already passed.

| ID | Surface | Current LangChain behavior | Intergrax contract | Native target | Classification | Unit proof | Integration proof | Live proof required | Consumer impact | Failure severity |
|---|---|---|---|---|---|---|---|---|---|---|
| OLLAMA-PARITY-001 | Constructor/configuration | Model from argument, env, or `llama3.1:latest`; optional `base_url`, resolver, context override; remaining defaults retained. | Construct an `LLMAdapter` with stable config and no provider object leakage. | Official client plus same injectable resolver/config boundary. | MUST_MATCH | Constructor fixtures. | Profile-to-adapter construction. | No. | All callers. | P2 |
| OLLAMA-PARITY-002 | Provider identity | `LLMProvider.OLLAMA`; response provider slug `ollama`. | Provider identity is stable in adapter and response. | Same enum/value. | MUST_MATCH | Identity assertions. | Registry/profile path. | No. | Usage and LKW diagnostics. | P2 |
| OLLAMA-PARITY-003 | Model identity | `adapter.model == chat.model`; response model is that value. | Model is the configured provider model. | Same string, including tag. | MUST_MATCH | Mock client identity. | Profile construction. | No. | Routing, metrics, proof logs. | P2 |
| OLLAMA-PARITY-004 | Explicit context window | Positive `context_window_tokens` is consumed by shared initialization. | Positive override is authoritative. | Same shared initializer. | MUST_MATCH | Context override test. | Profile/context preflight. | No. | Token budgeting. | P1 |
| OLLAMA-PARITY-005 | Model-derived context window | Shared resolver uses catalog, prefix, gateway, legacy, provider default, fallback; legacy helper is not constructor path. | One canonical context path; no second Ollama table. | Reuse `init_adapter_context_window_tokens`. | MUST_MATCH | Catalog/fallback tests. | Adapter/preflight integration. | No. | Token Optimization and LKW budgets. | P1 |
| OLLAMA-PARITY-006 | System messages | `SystemMessage` with exact content. | System content and order preserved. | Native role `system`. | MUST_MATCH | Message capture. | Harness normalized request. | No. | All prompts. | P2 |
| OLLAMA-PARITY-007 | User/assistant content | Human/AI messages preserve empty strings; response text blocks are joined by current coercion. | Empty and multiblock observable text behavior preserved. | Native text normalization with same result. | MUST_MATCH | Empty/multiblock fixtures. | Plain response harness. | No. | Plain and structured prompts. | P2 |
| OLLAMA-PARITY-008 | Tool-result messages | Requires non-blank `tool_call_id`; otherwise `ValueError`. | Tool results are correlated, never sent without an ID. | Native role `tool` plus required ID. | MUST_MATCH | Missing-ID test. | Assistant/tool continuation harness. | No. | Tools and LKW. | P1 |
| OLLAMA-PARITY-009 | Unknown/custom role | Projected to system content `[ROLE]\ncontent`. | No arbitrary vendor role becomes public ABI. | Same compatibility projection. | MUST_MATCH | Role mapping fixture. | Request normalization. | No. | Legacy/custom callers. | P3 |
| OLLAMA-PARITY-010 | Plain generation request | `chat.invoke(lc_msgs, **kwargs)` with merged options. | Typed `generate_messages` request and response. | Native `chat(..., stream=False)` with equivalent request. | MUST_MATCH | Mock call capture. | Side-by-side request normalization. | No. | LKW plain path. | P1 |
| OLLAMA-PARITY-011 | Plain response envelope | Text, estimated usage, model/provider, `usage_source=estimate`; default completed reason. | Return `LLMAdapterResponse`, never bare text. | Existing builder and envelope fields. | MUST_MATCH | Response contract tests. | Normalized response comparison. | No. | All ABI consumers. | P1 |
| OLLAMA-PARITY-012 | Plain run lifecycle | `begin_call`/`end_call`; default run id `general`; errors count on exception. | Exactly one lifecycle per invocation. | Same usage log and error classification. | MUST_MATCH | Usage lifecycle fixtures. | Runtime usage tracker. | No. | Token accounting. | P1 |
| OLLAMA-PARITY-013 | Partial stream event | Non-empty chunks become `PARTIAL`, no response envelope. | `LLMStreamEvent` ABI and delta semantics. | Normalize native chunks to same event. | MUST_MATCH | Synthetic chunk stream. | Harness event trace. | No. | Streaming callers. | P1 |
| OLLAMA-PARITY-014 | Final stream event | One final event after success; full text in response, empty delta. | Final event terminates successful stream. | Same ordering and fields. | MUST_MATCH | Event ordering fixture. | Stream integration harness. | No. | Streaming callers. | P1 |
| OLLAMA-PARITY-015 | Stream fallback before partial | Stream exception triggers one `invoke` fallback, then partial fallback text and final event. | Preserve fallback where no token escaped. | Native retry only before partial. | MUST_MATCH | Injected pre-token failure. | Failure normalization. | No. | Future streaming callers. | P1 |
| OLLAMA-PARITY-016 | Stream fallback after partial | Unconditional fallback can duplicate emitted content. | Do not duplicate or retract emitted content. | Propagate failure after partial; no fallback/final. | INTENTIONAL_IMPROVEMENT | Partial-then-fail test. | Side-by-side failure harness. | No. | Streaming correctness. | P1 |
| OLLAMA-PARITY-017 | Stream failure lifecycle | Failed stream raises; `finally` records input estimate/error; no final event. | Failure is observable, not fabricated success. | Same shape with row 016 fix. | MUST_MATCH | Disconnect fixture. | Usage plus event harness. | No. | Observability. | P1 |
| OLLAMA-PARITY-018 | Tool definition serialization | Schema passed unchanged to `bind_tools`. | Tool schema is caller-owned Intergrax data. | Same function schema to native `tools`. | MUST_MATCH | Captured schema. | Token Optimization envelope hash. | No. | LKW and Token Optimization. | P1 |
| OLLAMA-PARITY-019 | `tool_choice` | Only `None`, `"auto"`, `"required"`; others fail before provider. | No expanded choice semantics. | Same values and fail-fast behavior. | MUST_MATCH | Parametrized choice test. | Tool request harness. | No. | LKW forced/automatic mode. | P2 |
| OLLAMA-PARITY-020 | Assistant tool-call response | Maps calls to typed tuple, preserves order, sets `TOOL_CALLS`. | Canonical `LLMToolCall` response. | Parse native calls into same tuple. | MUST_MATCH | Typed call fixtures. | Native response normalization. | No. | Token Optimization/LKW. | P1 |
| OLLAMA-PARITY-021 | Tool arguments | Dict/JSON-object arguments become `arguments_json`; invalid calls fail. | Arguments remain valid JSON objects. | Parse and validate before response. | MUST_MATCH | Malformed/Unicode fixtures. | Native tool response normalization. | `OLLAMA-PARITY-044`. | Tool execution safety. | P1 |
| OLLAMA-PARITY-022 | Assistant/tool continuation | Assistant calls and tool results are separate messages; no execution loop. | Preserve call ID across continuation. | Native canonical assistant calls and tool ID. | MUST_MATCH | Message round-trip fixture. | LKW continuation harness. | No. | LKW tools. | P1 |
| OLLAMA-PARITY-023 | Tool streaming ABI | `stream_with_tools` inherited `NotImplemented`; adapter documents unsupported. | Tool streaming is `NOT_SUPPORTED_CURRENTLY`. | Keep unsupported in 6B. | CURRENTLY_UNSUPPORTED | Unsupported-method assertion. | None. | No. | No current direct consumer. | P2 |
| OLLAMA-PARITY-024 | Incremental tool-call streaming | No current behavior or contract. | Not mandatory for parity. | Future enhancement only. | POST_PARITY_ENHANCEMENT | None in 6B. | Separate future proof. | No. | Future streaming consumers. | P3 |
| OLLAMA-PARITY-025 | Structured schema preparation | Projects Pydantic schema with `prepare_ollama_generation_schema`. | Provider schema may be projected; original model validates. | Same projector and `format` schema. | MUST_MATCH | Schema projection tests. | Captured native request. | No. | LKW planning. | P1 |
| OLLAMA-PARITY-026 | Structured raw/parsed contract | `include_raw=True`; typed parsed result plus raw response envelope. | `LLMStructuredResult(parsed, response)` authoritative. | Parse raw message and build same result. | MUST_MATCH | Structured mock tests. | Native structured harness. | No. | LKW and Token Optimization fallback. | P1 |
| OLLAMA-PARITY-027 | Structured validation | Dicts revalidated with original model; invalid data raises. | Parseable JSON is insufficient. | Same original-model boundary. | MUST_MATCH | Invalid Pydantic fixtures. | Native malformed payload harness. | No. | Correctness and safety. | P1 |
| OLLAMA-PARITY-028 | Structured errors | Parsing error/missing parsed/unexpected type raises; usage records error. | Fail closed with typed-result contract. | Same outcome; no silent JSON fallback. | MUST_MATCH | Error lifecycle tests. | LKW planner failure path. | No. | LKW repair behavior. | P1 |
| OLLAMA-PARITY-029 | Capability discovery | Lazy `/api/show`, per-instance cache, explicit refresh. | Reuse existing objects and resolver. | Same resolver, no duplicate cache. | MUST_MATCH | Resolver/cache tests. | Adapter capability integration. | No. | Tools and Token Optimization. | P1 |
| OLLAMA-PARITY-030 | Capability failure/unknown model | Missing/malformed/unavailable is unresolved and `supports_tools=False`; router fails closed. | Unknown is not resolved no-tools. | Same unresolved state and selection. | MUST_MATCH | Resolver error fixtures. | Token Optimization router proof. | No. | Security/correctness. | P1 |
| OLLAMA-PARITY-031 | `supports_structured_output` | Always `True`; resolver has no structured flag. | No new capability system. | Return `True` while JSON Schema path exists. | MUST_MATCH | Capability assertion. | Router fallback path. | No. | LKW/Token Optimization. | P2 |
| OLLAMA-PARITY-032 | Usage source/provider counters | Always estimates and reports `usage_source=estimate`. | Same shape; counters only when trustworthy. | Prefer `prompt_eval_count`/`eval_count`, fallback estimator, expose source. | INTENTIONAL_IMPROVEMENT | Fake counter/fallback fixtures. | Usage aggregation comparison. | `OLLAMA-PARITY-050`. | Token Optimization accounting. | P1 |
| OLLAMA-PARITY-033 | Run ID/success/error accounting | `run_id` reaches lifecycle; totals/errors/metrics aggregate at end. | Preserve lifecycle and no duplicate calls. | Same usage log calls and fallback decision. | MUST_MATCH | Usage log assertions. | Runtime tracker proof. | No. | Token Optimization/observability. | P1 |
| OLLAMA-PARITY-034 | Timeout semantics | Timeout is library-owned/pass-through; no uniform adapter timeout proven. | No arbitrary timeout value. | Preserve effective default or adopt only proven policy. | LIVE_PROOF_REQUIRED | Injected timeout config. | Call-config integration. | LCI-6C effective-timeout proof. | LKW reliability. | P2 |
| OLLAMA-PARITY-035 | Connection refused | Provider exception propagates; no response; call lifecycle records error where begun. | Transport failure remains an error. | Native failure normalized only at harness boundary. | LIVE_PROOF_REQUIRED | Injected connection error. | Error/lifecycle harness. | Real unavailable endpoint. | All consumers. | P1 |
| OLLAMA-PARITY-036 | Timeout failure | Client timeout propagates according to effective provider behavior. | No silent retry or success. | Same outcome under timeout decision. | LIVE_PROOF_REQUIRED | Injected timeout. | Retry/config harness. | Real bounded timeout proof. | LKW/Token Optimization. | P1 |
| OLLAMA-PARITY-037 | Model missing/invalid | Generation fails; capability lookup catches and returns unresolved. | Generation fails; capability checks fail closed. | Same split behavior. | LIVE_PROOF_REQUIRED | Mock 404/show failure. | Capability/generation harness. | Missing model on real Ollama. | Resolver and LKW. | P1 |
| OLLAMA-PARITY-038 | Invalid request | Provider/client exception propagates; no new hierarchy. | Request failure visible; usage not success. | Same observable category. | LIVE_PROOF_REQUIRED | Invalid schema/message fixture. | Native request/error harness. | Real invalid-request proof. | Tool/structured callers. | P1 |
| OLLAMA-PARITY-039 | HTTP 4xx | No adapter remapping; client/provider error propagates. | Preserve failure, no fabricated response. | Same category/lifecycle. | LIVE_PROOF_REQUIRED | Injected 4xx. | Error normalizer. | Real provider 4xx where reproducible. | All consumers. | P2 |
| OLLAMA-PARITY-040 | HTTP 5xx | No remapping; tools may pass through `_execute` resilience. | Do not broaden retry semantics accidentally. | Preserve method-specific effective behavior. | LIVE_PROOF_REQUIRED | Injected 5xx. | Resilience boundary harness. | Bounded 5xx/transport proof. | Tool reliability. | P1 |
| OLLAMA-PARITY-041 | Malformed response | Plain extraction can fall back to `str(res)`; structured/tool paths fail on invalid shape. | Typed response rules and fail-closed structured/tool paths. | Explicit native normalization per method. | LIVE_PROOF_REQUIRED | Malformed payload fixtures. | Side-by-side normalized response. | Real malformed-payload proof if possible. | All consumers. | P1 |
| OLLAMA-PARITY-042 | Stream disconnect | Partial events may precede exception; no final on failed stream; fallback defect in row 016. | No duplicate output/final after failure. | Pre-partial fallback only; post-partial failure. | LIVE_PROOF_REQUIRED | Synthetic disconnect both points. | Event/usage harness. | Bounded real disconnect proof. | Streaming callers. | P1 |
| OLLAMA-PARITY-043 | Structured parse failure | `parsing_error`/missing parsed raises; original validation required. | No parseable-JSON-only acceptance. | Native raw JSON parse plus model validation. | LIVE_PROOF_REQUIRED | Invalid JSON/schema fixtures. | Native format/validation harness. | Real constrained-output failure proof. | LKW planner. | P1 |
| OLLAMA-PARITY-044 | Tool-call parse failure | Invalid call/argument shape raises; no execution. | Invalid calls fail closed. | Native parser rejects invalid name/JSON/object. | LIVE_PROOF_REQUIRED | Invalid call fixtures. | LKW/Token Optimization tool harness. | Real invalid tool-call proof. | LKW/Token Optimization. | P1 |
| OLLAMA-PARITY-045 | LKW plain generation | Model-runtime proof calls `generate_messages` and reads typed `content`. | Typed response and identity remain usable. | Same ABI behind non-default native path. | MUST_MATCH | Mock proof adapter. | Model-runtime proof. | LCI-6C live generation. | LKW. | P1 |
| OLLAMA-PARITY-046 | LKW structured planning | Calls capability and `generate_structured`; validates model and may repair once. | Preserve typed result and failure/repair semantics. | Same result/error behavior. | MUST_MATCH | Planner unit path. | LKW structured proof. | LCI-6C live structured proof. | LKW. | P1 |
| OLLAMA-PARITY-047 | LKW non-streaming tools | Calls capability then tools; requires one expected tool and valid workspace args. | Preserve fail-closed validation. | Same canonical tool call. | MUST_MATCH | Tool proof fixtures. | LKW tool proof. | LCI-6C live tools proof. | LKW. | P1 |
| OLLAMA-PARITY-048 | Token Optimization transport | Resolved tools use native tools; resolved no-tools may fallback structured; unresolved fails closed. | Flags and typed calls drive existing router. | Same selection; no router change. | MUST_MATCH | Existing router unit suite. | Native router integration. | LCI-6C capability/tool proof. | Token Optimization. | P1 |
| OLLAMA-PARITY-049 | Token Optimization usage/ABI | Router uses adapter methods, tool calls, schema envelope, usage path. | No private tokenizer or usage tracker. | Reuse contracts; report counter delta. | MUST_MATCH | Existing Token Optimization tests. | Router usage/cache integration. | LCI-6C evidence feeds LCI-6D. | Token Optimization. | P1 |
| OLLAMA-PARITY-050 | Native transport/counter evidence | LangChain exposes estimate-only behavior; native client may expose prompt/eval counters. | Differences explicit and attributable. | Official client operations plus trustworthy-counter policy. | LIVE_PROOF_REQUIRED | Fake native client contract. | Side-by-side harness. | Real chat/stream/tools/format/show evidence. | LCI-6B/6C gate. | P1 |
| OLLAMA-PARITY-051 | Streaming capability flag | Current adapter inherits `supports_streaming() == False` even though `stream_messages()` is implemented. | Capability flags must truthfully advertise implemented public surfaces. | Return `True` when native streaming is implemented; no change to event ABI. | INTENTIONAL_IMPROVEMENT | Capability flag assertion. | Consumer capability integration. | No. | Future stream consumers. | P2 |

## LCI-6B UNIT/INTEGRATION EVIDENCE

The following deterministic evidence is implemented in
`tests/unit/llm_adapters/test_native_ollama_adapter.py`:

```text
001-012  constructor, identity, context, message mapping, plain generation,
         typed response and usage lifecycle
013-017  successful streaming, pre-partial fallback, post-partial failure,
         final-event ordering and failed-call accounting
018-024  caller-owned tool schemas, tool_choice validation, canonical tool
         calls, tool-result correlation, unsupported tool streaming
025-028  projected structured schema, raw JSON parsing, original-model
         validation and structured failure lifecycle
029-033  capability cache/refresh, fail-closed flags, provider counters,
         estimator fallback and run accounting
045-049  typed ABI identity, structured/tool surfaces, side-by-side request
         normalization and unchanged Token Optimization-facing contracts
051      truthful native streaming capability flag
```

Rows `034-044` and `050` remain `LIVE-PROOF-PENDING` for LCI-6C. The harness
uses fake native and LangChain transports and does not claim real-Ollama
evidence.

## Matrix summary and gate ownership

The 51 rows above have these classifications:

```text
MUST_MATCH                 34
INTENTIONAL_IMPROVEMENT     3
CURRENTLY_UNSUPPORTED       1
POST_PARITY_ENHANCEMENT     1
LIVE_PROOF_REQUIRED         12

P0                          0
P1                         36
P2                         14
P3                          1
```

LCI-6B may implement only the target behavior described here and must attach deterministic unit/integration evidence to all implemented rows. LCI-6C must execute and record the live rows, especially:

```text
OLLAMA-PARITY-034
OLLAMA-PARITY-035
OLLAMA-PARITY-036
OLLAMA-PARITY-037
OLLAMA-PARITY-038
OLLAMA-PARITY-039
OLLAMA-PARITY-040
OLLAMA-PARITY-041
OLLAMA-PARITY-042
OLLAMA-PARITY-043
OLLAMA-PARITY-044
OLLAMA-PARITY-050
```

LCI-6D is the first task allowed to change the default resolver or LKW/Token Optimization execution path, and only after mandatory rows are proven. LCI-6E then handles compatibility packaging. LCI-6A itself changes no runtime code, resolver, LKW path, Token Optimization policy, dependency, or test.
