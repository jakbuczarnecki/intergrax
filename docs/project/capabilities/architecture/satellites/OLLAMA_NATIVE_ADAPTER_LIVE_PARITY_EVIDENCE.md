# LCI-6C Native Ollama Live Parity Evidence

## Run identity

- Task: `LCI-6C-NATIVE-OLLAMA-MANDATORY-LIVE-PARITY-PROOF`
- Validated commit: `0eb7aec60d8b5a4023da46dc43f4d53a67cc08ab`
- Date: 2026-08-09
- Ollama: `0.32.5`
- Endpoint class: local loopback HTTP, `http://127.0.0.1:11434`
- Tools model: `qwen2.5:7b`
- Resolved no-tools model: `nomic-embed-text:latest` (`embedding` only; no local completion model without `tools`)
- Missing model: `intergrax-lci6c-definitely-missing-model`
- Secrets, credentials, and full private prompts are not recorded.

## Live matrix

| Row | Status | Exact observation |
|---|---|---|
| `OLLAMA-PARITY-034` | `PASS` | Injected client-owned 1 ms timeout raised `ReadTimeout`; native usage lifecycle recorded one call and one error. |
| `OLLAMA-PARITY-035` | `PASS` | Unused loopback endpoint raised `ConnectionError`; no response; usage recorded one error. |
| `OLLAMA-PARITY-036` | `PASS` | Same bounded client timeout raised `ReadTimeout`; no fabricated response or success usage. |
| `OLLAMA-PARITY-037` | `PASS` | Missing model generation raised `ResponseError` HTTP 404; `/api/show` resolved false, `supports_tools=false`; usage recorded one error. |
| `OLLAMA-PARITY-038` | `PASS` | Real endpoint rejected malformed JSON Schema with `ResponseError` HTTP 400. |
| `OLLAMA-PARITY-039` | `PASS` | Real endpoint rejected invalid format schema with `ResponseError` HTTP 400. |
| `OLLAMA-PARITY-040` | `LIVE_NOT_REPRODUCIBLE` | No local 5xx fault injection or proxy was used. |
| `OLLAMA-PARITY-041` | `LIVE_NOT_REPRODUCIBLE` | Ollama runtime exposes no controlled malformed-response switch. |
| `OLLAMA-PARITY-042` | `LIVE_NOT_REPRODUCIBLE` | Disconnect injection would disrupt shared Ollama sessions; no isolated repo mechanism exists. |
| `OLLAMA-PARITY-043` | `PROVIDER_PREVENTS_REPRODUCTION` | Constrained real-provider structured request returned valid output; malformed structured output was not emitted. |
| `OLLAMA-PARITY-044` | `PROVIDER_PREVENTS_REPRODUCTION` | Real provider emitted no malformed tool call; no synthetic failure was promoted to live evidence. |
| `OLLAMA-PARITY-050` | `PASS` | Native raw Ollama counters were present for plain, tools, structured, and stream responses; adapter used `usage_source=sdk`. |

## Plain generation and side-by-side baseline

`NativeOllamaAdapter.generate_messages()` returned `LLMAdapterResponse` with provider `ollama`, model `qwen2.5:7b`, non-empty content, usage `input=41`, `output=3`, `total=44`, and `usage_source=sdk`.

The same canonical request through `LangChainOllamaAdapter` returned the same response ABI, provider, model, and non-empty-content shape. Its estimate usage was `input=12`, `output=2`, `total=14`, with `usage_source=estimate`. Generated words were not compared byte-for-byte.

## Capabilities

`/api/show` resolved `qwen2.5:7b` as `resolved=True`, `supports_tools=True`, `capabilities={completion, tools}`, `source=api_show`. The adapter cache returned the same value after refresh. `resolved no-tools chat model: BLOCKED_MODEL_AVAILABILITY`.
`nomic-embed-text:latest` resolved as a local embedding-only model without `tools`; no local completion model with resolved no-tools capability was installed. The embedding-only model is not evidence of no-tools chat parity. The missing model resolved false and did not claim capabilities.

## Tools

Both adapters used the same `get_weather(city: string)` schema and produced one canonical tool call: `name=get_weather`, `arguments={"city":"Warsaw"}`, `finish_reason=TOOL_CALLS`. Native response usage was `input=159`, `output=21`, `total=180`, `usage_source=sdk`. The tool was not executed.

## Structured output

Both adapters projected and accepted the small Pydantic model `{city: str, temperature_c: int}`. Native output revalidated as `{"city":"Warsaw","temperature_c":20}`. Native response usage was `input=41`, `output=20`, `total=61`, `usage_source=sdk`. Original Pydantic validation passed.

## Stream

Native `stream_messages()` emitted 30 `PARTIAL` events followed by exactly one `FINAL`. The final content equaled the concatenation of all partial deltas. The final response exposed `usage_source=sdk`; raw counters were `prompt_eval_count=35`, `eval_count=31`. LangChain produced the same partial/final ordering shape with 30 partial events and one final event.

## Provider counters

| Surface | `prompt_eval_count` | `eval_count` | Native usage source |
|---|---:|---:|---|
| plain | 41 | 3 | `sdk` |
| tools | 159 | 21 | `sdk` |
| structured | 41 | 20 | `sdk` |
| stream final | 35 | 31 | `sdk` |

## Error evidence

- Connection refused: native `ConnectionError`, no response, usage `calls=1/errors=1`.
- Timeout: injected official-client timeout raised `ReadTimeout`, no response, usage `calls=1/errors=1`; timeout remains client/library-owned.
- Missing model: native `ResponseError` HTTP 404 and unresolved capability state.
- Invalid request: native official client sent a malformed JSON Schema and Ollama returned HTTP 400.
- HTTP 4xx: invalid format schema returned `ResponseError` HTTP 400.
- HTTP 5xx: `LIVE_NOT_REPRODUCIBLE`; no fault injection was introduced.
- Malformed response: `LIVE_NOT_REPRODUCIBLE`; no provider-controlled malformed payload was available.

Supporting deterministic LCI-6B evidence for rows `040-044` remains in
`tests/unit/llm_adapters/test_native_ollama_adapter.py`, including injected
transport/disconnect and malformed-output failure cases. These cases validate
fail-closed error semantics only; they are synthetic support and are not
represented as live PASS.

## Proof implementation

- `tests/integration/llm_adapters/test_native_ollama_live_parity.py`
- Explicit gate: `INTERGRAX_LCI6C_LIVE=1`
- No automatic model pull
- No production runtime, resolver, LKW, or Token Optimization changes

## Verdict

`LCI-6C: READY_FOR_REVIEW`

No `LIVE_PARITY_REGRESSION_FOUND` was observed. Critical real-user surfaces passed: plain generation, provider counters/fallback policy, capability resolution, missing model, connection failure, stream success, tools, and structured output.

Roadmap:

```text
LCI-6A - APPROVED
LCI-6B - APPROVED
LCI-6C - READY_FOR_REVIEW
LCI-6D - NEXT AFTER ACCEPTANCE
```
