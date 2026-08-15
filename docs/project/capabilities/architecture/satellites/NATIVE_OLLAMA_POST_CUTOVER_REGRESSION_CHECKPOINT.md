# Native Ollama post-cutover regression checkpoint

- validated SHA: `17a4a641c761aca87a5f242ab82b62c0a8f03a84`
- LCI-6D SHA: `0c0476a6a980fe9e9c9bacc3b0999c2d7f6e6070`
- date: `2026-08-09`
- verdict: `PASS`

## Gate matrix

| Gate | Result | Evidence |
|---|---|---|
| A registry | PASS | Enum and canonical string `ollama` resolve through the registry to `NativeOllamaAdapter`; catalog wrapper unwraps to native. |
| B compatibility | PASS | Explicit LangChain unit/parity coverage and 2 live compatibility tests pass; it is not the registry default. |
| C blocked LangChain | PASS | Subprocess blocks `langchain`, `langchain_core`, and `langchain_ollama`; registry construction still returns `NativeOllamaAdapter`. |
| D plain ABI | PASS | Deterministic and live responses are `LLMAdapterResponse` with provider, model, content, finish reason, usage, and extensions. |
| E streaming | PASS | Deterministic success/failure semantics pass; live smoke produced 3 partials, exactly 1 final, and matching concatenation. |
| F tools | PASS | Canonical schema, required choice, `LLMToolCall`, JSON arguments, `TOOL_CALLS`, and fail-closed invalid data pass. |
| G structured | PASS | Native schema preparation, JSON decode, original Pydantic validation, result contract, and fail-closed invalid output pass. |
| H capabilities | PASS | Resolved tools/no-tools/unresolved states are distinct; unresolved is not treated as no-tools. |
| I Token tools | PASS | Existing router tests select native tools and accept canonical tool results. |
| J Token no-tools | PASS | Existing deterministic router policy selects structured fallback only for resolved no-tools. |
| K Token unresolved | PASS | Existing router policy returns `CAPABILITY_RESOLUTION_FAILED` and does not fall back. |
| L Token usage | PASS | SDK counters and estimate fallback are both covered. |
| M context | PASS | Known qwen context, explicit override, catalog resolution, and fallback behavior pass. |
| N LKW plain | PASS | Production-like LKW runtime session uses registry default and basic generation passes. |
| O LKW structured | PASS | Production-like planner returns validated typed plan. |
| P LKW tools | PASS | Production-like workspace tool schema returns one canonical call with required choice and validated args. |
| Q LKW health | PASS | Live health probe reports `NativeOllamaAdapter`, provider `ollama`, model `qwen2.5:7b`, and capabilities. |
| R live default | PASS | Registry-created live default passes plain, tools, and structured smoke. |
| S live capability | PASS | `qwen2.5:7b` resolves `completion/tools` via API `/api/show`. |
| T live streaming | PASS | Registry-created live stream passes partial/final and concatenation checks. |
| U usage lifecycle | PASS | Plain/tools/structured/stream and deterministic failure paths have one begin/end lifecycle and correct success/error accounting. |
| V legacy references | PASS | Compatibility/test references remain classified; no active default generation path requires the LangChain class. |
| W structural audits | PASS | Inventory and LangChain boundary audits report zero new forbidden imports and zero stale grandfather entries. |
| X regression suites | PASS | Native, LangChain baseline, registry, LKW targeted, Token Optimization/router suites: 256 passed. |
| Y construction/concurrency | PASS | Five repeated registry constructions return distinct native adapters and clients. |
| Z packaging | PASS | `langchain-ollama` and `langchain-core` remain required at this stage. |

## Runtime and suite evidence

- Ollama: `0.32.5`
- live model: `qwen2.5:7b`
- resolver type: `NativeOllamaAdapter` (catalog wrapper removed for inspection)
- live usage source: `sdk`
- default smoke: plain/tools/structured/stream PASS
- existing native/LangChain live parity: `1 passed`
- explicit LangChain live tool/structured tests: `2 passed`
- deterministic baseline: `256 passed`, `0 failed`
- audits: inventory and boundary both PASS

## Known exclusions

- Full two-provider LKW portability proof was not rerun because this checkpoint only required the available real Ollama default path; no VLLM environment was required.
- `intergrax/multimedia/image_smart_loader.py` retains a legacy, vision-specific `LangChainOllamaAdapter` type check. It is not on the active default LLM generation/registry path and remains a `STALE DEFAULT DEPENDENCY` for a separate compatibility review.
- Rows `0029/0030` and packaging removal remain owned by LCI-6E/7A.
- No production code was changed.

## Status

- LCI-6A: `APPROVED`
- LCI-6B: `APPROVED`
- LCI-6C: `APPROVED`
- LCI-6D: `APPROVED`
- Native Ollama regression gate: `READY_FOR_REVIEW`
- next: `LCI-6E`
