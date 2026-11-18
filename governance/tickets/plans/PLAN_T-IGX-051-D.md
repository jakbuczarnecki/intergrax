# PLAN T-IGX-051-D — RW train, node T-AF2-RW-RUNTIME: LLM runtime resilience, typed error contract v1, served model, refusal matrix, SDK boundary

Draft, 2026-11-17. Nothing was run and nothing was changed outside `.tmp`. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan; labels such as "VERIFIED" or "STATIC_VERIFIED" are quoted with attribution. Evidence notation `path@H:line`; `H` = `2727a2c2364a97492202b931375772a400279a9b` (`origin/development`, fetched 2026-11-17). The product tree of `H` equals AUDIT_BASELINE_2 `7960ae7f6` (`git diff 7960ae7f6 origin/development -- intergrax agents applications tests/unit/llm_adapters tests/qualification` is empty) and the train branch `arch-freeze/rw-train` (`05699b633`, PT-WA merged) has no difference from `H` on any path of this write set (`git diff origin/development origin/arch-freeze/rw-train -- <scope paths>` is empty). Every code claim is listed with its line in `FACT_CHECK.md` (ids `F-nn`).

## 1. Purpose and findings

Node `T-AF2-RW-RUNTIME` (RW-PLAN v13, ticket `T-IGX-051-D`, lane 3, product-path ticket of the single RW train; plan of record section E row "REMEDIATION — runtime / LLM" and answer WP12-Q02 for A4-018). Binding scope: the block `T-AF2-RW-RUNTIME (T-IGX-051-D)` of `RW_PLAN.md` appendix A1 (17 firm paths: 9 product, 8 test; 0 conditional), its negative-test rows NT-ROW-005, -036, -037, -039, -103..-108, the regression scope `tests/unit/llm_adapters`, `tests/qualification/ebh_5`, `tests/qualification/ebh_6`, `tests/unit/multimedia`, and lane 3 with no file overlap with any other train ticket.

Owner per RW-PLAN (exactly-once ownership, section g): A4-018, EX-D3-01, EX-D3-03, EX-L-07, EX-L-08, FRZ-CTR-01, FRZ-CTR-02, FRZ-CTR-06, PDC-D3-R005, PDC-D3-R019. Contributes to: FRZ-CFG-02, FRZ-CFG-06, FRZ-SEC-04, PDC-D3-R018. The dispatch of this draft also lists PDC-D3-R018 as owned; RW-PLAN v13 reassigns it to `T-AF2-RW-SUPPLY` (dedupe D-07: root cause F-D3-12 is in `intergrax/llm_adapters/governance/quota.py`, which is not in this write set). This plan follows RW-PLAN; see Q-RUN-01.

| Item | Statement (quoted from the DS-EXC package, `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`, a session artefact of the ARCH-FREEZE session; "STATIC_VERIFIED" is its label) | What this plan does | What it cannot close alone |
|---|---|---|---|
| EX-L-08 (E7 / G-11) | `_execute_streaming` hands the resilience wrapper a lazy generator, so the breaker records success before any SDK call and retry never sees stream-open errors | Part A: the stream is opened and its first event pulled inside the physical attempt, so the breaker and the retry loop see open errors | closure is the exact-SHA wave audit (`PT-Q02`) and the W7 independent session; a live provider is not exercised (C10/C15) |
| EX-D3-01 (F-D3-03) | `timeout_sec` is computed and then dropped; a generic wall-clock wrapper cannot cancel a hung SDK call | Part B: per-provider forwarding of the effective timeout to the SDK call where the SDK offers a per-call or client-level option; adapters that cannot forward fail closed (Q-RUN-02) | live-provider qualification named in the reopen condition is not possible under the C10/C15 freeze; fake-SDK tests only |
| EX-D3-03 (F-D3-07) | no typed provider-error contract | Part C: typed error contract v1 and its mapping at the adapter boundary | consumer migration outside this write set (see section 12) |
| EX-L-07 (F-D3-02 pins) | refusal and finish normalization pinned on two methods only; `generate_with_tools`, `stream_with_tools` and the Bedrock invoke paths unpinned | Part D: a parametrized matrix over every shipped adapter method and the Bedrock invoke paths | whether the matrix is red at base is unknown (Q-RUN-07) |
| PDC-D3-R005 (F-D3-10) | served-model attribution (PARTIAL, documented debt) | Part E: the served model travels on the response and the call accounting uses it | the accounting sink files are outside the write set (design in 3.5) |
| PDC-D3-R019 (F-D3-09) | no vendor selection hidden in core; VBN classification of provider `DEFAULT_MODEL` constants | Part F: register rows for the provider defaults; capability-flag parity | `LLMProfile.lab` and the rag default embedding provider are outside the write set; final VBN text is `T-AF2-VBN-DOSSIER` |
| FRZ-CTR-01 | declared streaming capability must match the implemented method (PDC D3-R022, D3-R006) | Part F: parity test over every shipped adapter | per-adapter result is measured at step 0 (basis C) |
| A4-018, FRZ-CTR-02, FRZ-CTR-06 | direct `ollama` SDK use outside the provider boundary (Tier-0 multimedia; Tier-2 qualifier) | Part G: SDK use moves into the provider package; the qualifier classifies typed errors | the banned-list extension of the two vendor-SDK gate scripts stays with `T-AF2-RW-GATES` (dedupe D-05) |

Scope hold: the ticket allows no edit of `intergrax/`, `agents/` or `applications/` before its wave; this plan only describes the change. Branch rule: product commits of this ticket go only to branch `arch-freeze/rw-train`. Prerequisites that this plan does not assume: PT-WA and PT-WB merged into the train (RW-PLAN lane rule), the approval block of the ticket, the product-path authorization of the node (id not stated in the inputs of this draft).

## 2. Design rules applied to every part

- Fake SDKs only. No network, no credential, no Docker, no vendor integration (C10/C15 freeze, `intergrax-env-vendor-qualification`). Live parity is not claimed anywhere.
- No SDK import at module level outside `intergrax/llm_adapters/providers/**` (Part G rule); provider-specific classification of SDK errors uses attribute access on the raised object, not an SDK import in shared code.
- Ratchet (OD-AF-11): contract surfaces get no new `Any`, `object` or reflection dispatch. This ticket edits no path under `intergrax/contracts/**` (A1 lists none). New code in `intergrax/llm_adapters/contracts/*` is written with concrete types only; the orchestrator's wave-packet contract check (`rw-plan/work/typ05/measure_typ05.py` plus `typ_measure.py`) must report nothing new for those files. If a path under `intergrax/contracts/` turns out to be necessary, that is an amendment, not a silent edit.
- The write set is the A1 block. Concrete files under an A1 glob are an implementation breakdown, not a widening. A needed file outside the block stops the work (section 12, R-04).
- Behaviour that is not changed stays pinned: breaker, retry budget, rate limit, quota, deadline admission, external-operation lifecycle and permit release keep their order and semantics (Part A tests include controls).

## 3. Parts

### 3.1 Part A — stream-open errors through breaker and retry (EX-L-08)

Current state (`H`): `_execute_streaming` builds `physical_attempt()` which returns the lifecycle wrapper (`base_llm_adapter.py@H:367`, `:374`). The wrapper is a generator whose first line of work is `factory()` (`provider_stream_admission.py@H:75`), so nothing runs until the consumer iterates. `execute_with_resilience` calls `fn()` at `resilience.py@H:263` and records success at `:268`; for a stream, `fn()` returns immediately, so success is recorded before the SDK is touched and the retry loop never sees the open error. A permit-failure path exists (`stream_factory_with_admission`, `:48-59`) and the wrapper marks the external operation failed on a factory error (`:77`).

Design:
- `provider_stream_admission.py`: a helper `open_primed_stream(stream)` that obtains the iterator, pulls the first event, and returns an iterable that yields that event followed by the rest, with `close()` and `finally` semantics preserved (permit release, transport close, external-operation terminal state). An empty stream returns an empty iterable. An error at the first pull closes the underlying iterator (so the wrapper's `finally` runs) and re-raises.
- `base_llm_adapter.py` `_execute_streaming`: each `physical_attempt()` returns `open_primed_stream(<lifecycle wrapper>)` in both branches (with and without a dependency boundary). The attempt therefore fails inside `execute_with_resilience` when the open or the first event fails; the unchanged loop records the breaker failure and applies the existing retry classification and budget; a new attempt re-runs admission, external-operation attempt and permit acquisition exactly as today.
- Errors after the first event are not retried (the stream is already visible to the consumer). They pass through the lifecycle wrapper, are mapped to typed errors (Part C), mark the external operation failed, and are not recorded by the breaker (Q-RUN-04).
- Latency: the wait for the first event now happens inside the attempt; a hung first event is bounded by the forwarded timeout (Part B).

Negative tests: NT-RW-RUN-01 (section "Negative tests"). Controls inside the same file: a failure after the first event is not retried; a successful stream records breaker success only after the open succeeded; permits are released once on every path (reuse of the existing dependency-admission fakes).

### 3.2 Part B — timeout forwarding (EX-D3-01)

Current state: `LLMCallConfig.timeout_sec` is documented as an admission budget that "is not forwarded to the provider SDK" (`contracts/call_config.py@H:20`); both execute paths compute the effective timeout and only clamp a config copy for the resilience loop (`base_llm_adapter.py@H:287-295`, `:332-340`); no provider call site passes a timeout (for example `openai_chat_completions_adapter.py@H:107`, `:236`); the Responses adapter lists `timeout_sec` as an adapter-only key and `timeout` as a constructor-only key (`openai_responses_adapter.py@H:88`, `:72`).

Design:
- `base_llm_adapter.py`: `_execute` and `_execute_streaming` publish the effective timeout for the duration of one physical attempt in a request-scoped variable (set before the attempt, reset in `finally`); a protected accessor `_effective_call_timeout_sec()` returns it. The factory and `fn` closures of the providers run inside the attempt, so they read it at call time. The attempt priming of Part A runs the SDK open inside the same scope.
- Providers pass the value through the SDK mechanism that exists for that SDK. The expected mechanisms are a hypothesis to be confirmed at step 0 against the SDK versions in `uv.lock` and against the repository's fake SDKs (nothing was checked in this draft): per-request `timeout` for the OpenAI-compatible and Anthropic clients, a millisecond option for Mistral, a request option for Cohere, an HTTP option in the Gemini request config, a client-level timeout for the native Ollama client and for the LangChain Ollama client, and for Bedrock a client configuration (no per-call option is known). Step 0 produces a table `adapter -> mechanism -> PER_CALL | CLIENT_LEVEL | NONE` that the implementer records in the ticket.
- Adapters whose result is `NONE` declare `supports_call_timeout() -> False`; a call with an effective `timeout_sec` on such an adapter raises the existing `LLMCapabilityMismatchError` (new `LLMFeature` member in `contracts/capabilities.py` if needed) instead of silently dropping the budget (Q-RUN-02). Adapters with a client-level mechanism that cannot be changed per call are handled by constructing or deriving the client with the configured timeout when the adapter is built, and the plan records that the value is then fixed for the adapter's lifetime.
- The `LLMCallConfig` docstring is corrected to state what is forwarded and what is not. The field list of the dataclass is unchanged.
- A generic wall-clock wrapper is deliberately not added (the DS-EXC row states it cannot cancel a hung SDK call).

Negative tests: NT-RW-RUN-02 (section "Negative tests").

### 3.3 Part C — typed error contract v1 (EX-D3-03)

Current state: no typed provider-error class exists; the only typed adapter errors are `LLMCapabilityMismatchError` (`contracts/capabilities.py@H:65`), `LLMRateLimitError` and `LLMCircuitOpenError` (`_shared/resilience.py@H:52`, `:56`, outside the write set) and registry/config errors. Retry and failover classify by an HTTP status attribute or by CamelCase class-name words (`_shared/retry.py@H:83`), consumed by the failover policy (`registry/failover_policy.py@H:29`). Usage accounting labels failures with the raw class name (`openai_chat_completions_adapter.py@H:116` and the same pattern in the other providers).

Design:
- New `intergrax/llm_adapters/contracts/provider_error.py` (under the A1 glob): `LLM_PROVIDER_ERROR_CONTRACT_VERSION = 1`; a closed `LLMProviderErrorCategory` string enumeration; a base `LLMProviderError(RuntimeError)` with typed fields `category`, `provider`, `model`, `status_code: int | None`, `retry_after_seconds: float | None`, `retriable: bool`; and one subclass per category (draft set: timeout, rate limit, authentication, permission, invalid request, model not found, quota exhausted, overloaded or unavailable, server error, connection, unknown). Concrete types only; no `Any`, `object` or reflection dispatch.
- The message of a typed error carries category, provider, model and status, not the SDK message text; the SDK exception is the `__cause__` and is not serialized by the contract (Q-RUN-03 records the choice).
- Mapping at the adapter boundary in `base_llm_adapter.py`: `_execute` and `_execute_streaming` wrap the resilience call and convert any non-typed `Exception` into the matching typed error `from` the original. The generic classifier reads the status attribute and error-class words with the same helpers as the retry module and adds a provider hook `_classify_provider_error(exc)` (default none) that provider adapters override for their SDK (for example the Ollama response error with status 404 maps to model-not-found; the Bedrock client error code maps by code). Classification uses attribute access on the raised object; no SDK import is added to shared code.
- Pass-through (identity preserved, not wrapped): the pre-existing typed errors (`LLMRateLimitError`, `LLMCircuitOpenError`, provider rate-limit and retry-budget errors, `LLMQuotaExceeded`, `ExecutionProtectedWorkDeniedError`, `CooperativeCancellationAbort`, `LLMCapabilityMismatchError`, dependency-admission failures) and every `BaseException` that is not an `Exception`. The contract module documents them as members of v1 by reference (Q-RUN-05) because `resilience.py` is not in the write set.
- Retry and failover parity: typed errors carry `status_code` and `retry_after_seconds` and their class names keep the transient words, so `is_retriable_provider_error(typed)` equals `is_retriable_provider_error(raw)` for every fixture; the mapping happens outside the resilience loop, so the breaker and the retry loop keep seeing the raw SDK exception.
- Mid-stream errors are mapped inside the lifecycle wrapper: `stream_with_external_operation_lifecycle` gets an optional `map_error` keyword (default none, so the existing direct callers and tests keep working).
- Consumer: `agents/model_routing_qualifier/qualification_types.py` classifies the typed errors (Part G).

Negative tests: NT-RW-RUN-03 (section "Negative tests").

### 3.4 Part D — refusal and finish normalization matrix (EX-L-07)

Current state: normalization is one function (`contracts/finish_reason.py@H:65`) applied by `build_adapter_response` (`_shared/adapter_response_builders.py@H:27`). The existing PDC matrix registers only `generate_messages` and `stream_messages` paths, 15 paths in total (`tests/qualification/pdc_x/d3/test_d3_f02_refusal_paths.py@H:341-356`); `generate_with_tools` and `stream_with_tools` of every adapter and the Bedrock invoke paths (`aws_bedrock_adapter.py@H:551`, `:655`, `:785`) are not in it. Both Ollama adapters implement `generate_with_tools` and have no `stream_with_tools` (`native_ollama_adapter.py@H:560`, `ollama_adapter.py@H:484`).

Design:
- One new parametrized test file: for every shipped adapter, every method it implements among `generate_messages`, `stream_messages`, `generate_with_tools`, `stream_with_tools`, plus the Bedrock invoke-family paths for each codec, drive the real adapter method with a fake SDK payload for each signal the provider has (normal, length, content filter, refusal) and assert the finish reason and refusal text of the response (and of the final stream event). The cell list is derived from the adapter classes, with a completeness self-check that fails when an implemented method has no registered driver.
- Product edits only where a cell exposes a defect, inside `providers/*` or `_shared/openai_completion_mapping.py`. A defect whose fix needs `_shared/stream_finish_signal.py`, `contracts/finish_reason.py` (inside the glob) or other `_shared` files outside the block: `contracts/finish_reason.py` may be edited (glob); other `_shared` files stop the work and are an amendment.
- The existing PDC test file stays unchanged and is part of the regression scope.

Negative tests: NT-RW-RUN-04 (section "Negative tests"). Whether any cell is red at base is unknown (Q-RUN-07).

### 3.5 Part E — served-model attribution (PDC-D3-R005)

Current state: `LLMAdapterResponse` has `model` and `provider` and no served-model field (`contracts/adapter_response.py@H:28`); the providers fill `model` with the requested model (for example `openai_chat_completions_adapter.py@H:110`, `:173`, `:239`, `:330`, `:376`), and the shared OpenAI mapping takes `model` as a parameter (`_shared/openai_completion_mapping.py@H:160`). Call accounting records the adapter's requested model at `begin_call` (`base/usage_log.py@H:62`) and the LLM_CALL evidence and metrics read `call.model` at `end_call` (`:126`, `:140`); the context-attribution evidence reads the model from that event payload (`runtime/llm/model_context_attribution.py@H:141`). There are 37 `end_call` call sites in nine provider files (5 in the Responses adapter, 4 in each of the others). A central helper `_shared/call_lifecycle.py@H:47` exists and, like `usage_log.py`, is outside the write set.

Design (inside the write set):
- `contracts/adapter_response.py`: an additive field `served_model: str | None = None` (the existing `model` keeps its meaning, the requested model).
- Providers and `_shared/openai_completion_mapping.py` fill it from the SDK response where the SDK reports the model that served the call (expected: the response model attribute of the OpenAI-compatible, Anthropic, Mistral and Ollama responses; a model-version attribute for Gemini; Cohere and Bedrock to be confirmed at step 0 and left `None` when the SDK does not echo a model). The value is never fabricated: absent means `None`. Because `build_adapter_response` is outside the block, the field is set with `dataclasses.replace` on the built response (Q-RUN-06).
- `base_llm_adapter.py`: a helper `_attribute_served_model(call, response)` sets `call.model` to the served model when present, called by each provider before `end_call` (the 37 sites; stream paths use the final event's response). `usage_log.py` is not edited, so `begin_call` still records the requested model first and the helper overwrites it.
- `intergrax/runtime/llm/*` is an A1 glob; the expected edit is none, because the evidence adapters read the model from the payload that the helper now feeds. If the implementer finds that an edit there is needed, it is recorded and stays inside the glob.

Negative tests: NT-RW-RUN-05 (section "Negative tests").

### 3.5b Part F — capability-flag parity and VBN classification of provider defaults (FRZ-CTR-01, PDC-D3-R019)

- Parity (NT-ROW-036, basis C): for every shipped adapter class, `supports_streaming` is true exactly when `stream_messages` is implemented and `supports_tool_streaming` is true exactly when `stream_with_tools` is implemented, per the base defaults (`base_llm_adapter.py@H:426`, `:430`) and the declared flags. At step 0 the implementer records the per-adapter result; the two Ollama adapters are the known candidates (no `stream_with_tools`). A mismatch is fixed in the provider by correcting the flag or implementing the method, whichever the step-0 record supports. If nothing is mismatched at base, the row is recorded NOT_REPRODUCED per the RW-PLAN re-disposition rule and closes nothing.
- VBN classification (NT-ROW-108): `tests/qualification/ebh_6/_findings_register.py` gets one `HardcodingFindingRecord` per provider adapter that defines a `DEFAULT_MODEL` constant (8 at `H`: claude, cohere, gemini, mistral, native Ollama, LangChain Ollama, OpenAI Responses, Vertex Gemini; `F-49..F-56`), appended to `_TRACKED_DEBT` (`@H:210`). Draft classification: `ALLOWED — PROVIDER ADAPTER BOUNDARY`, remediation class `RATIFY_FIXED` or `MOVE_ADAPTER` as the VBN section 2.3 reading supports, status `CANDIDATE-RESOLVED` with a note that the final text is `T-AF2-VBN-DOSSIER`'s (Q-RUN-08). The register's derived counters (`count_unclassified`, `count_prohibited_unresolved`) stay 0; the id-uniqueness and tracked-debt tests keep passing; the roadmap-replay ledger classifies every register id automatically (`F-62`).

### 3.6 Part G — SDK boundary: no `ollama` import outside the provider package (A4-018, FRZ-CTR-02, FRZ-CTR-06)

Current state: `intergrax/multimedia/images_loader.py` imports `ollama` (and `IPython`) at module level (`@H:6`, `:7`); the Tier-2 qualifier imports `ollama.ResponseError` in a try block (`agents/model_routing_qualifier/qualification_types.py@H:149`) and classifies by that class and by message substrings (`:157`), and the job catches the boundary tuple (`steps/model_routing_job.py@H:242`, outside the write set); the smart loader reaches the vision bridge through a lazy import of `transcribe_image` (`image_smart_loader.py@H:162`) after trying `describe_image` (`:176`); the provider package already imports the SDK lazily (`native_ollama_adapter.py@H:94`, `ollama_capabilities.py@H:63`). An existing test pins the `images_loader.transcribe_image` seam through `sys.modules` and the keyword names `prompt`, `model`, `image_path` (`tests/unit/multimedia/test_image_smart_loader.py@H:43`); that test file is not in the write set.

Design (default option A, Q-RUN-09):
- New provider-package module `intergrax/llm_adapters/providers/ollama_vision.py` (under the A1 glob) holding the vision chat call with a function-level SDK import.
- `images_loader.py`: no module-level SDK or notebook import; `transcribe_image(prompt, image_path, model)` keeps its signature and delegates to the provider module. The seam used by the existing test stays, so that test needs no edit.
- `image_smart_loader.py`: the Ollama bridge path keeps its shape; the file is edited only if the delegation changes a call or message there (it may end up unedited; it is in the block as a ceiling).
- `agents/model_routing_qualifier/qualification_types.py`: the SDK import and `_OLLAMA_RESPONSE_ERROR` go; `classify_invocation_failure` maps the typed errors of Part C (model-not-found to `PROVIDER_MODEL_NOT_FOUND`; timeout and connection to `PROVIDER_TRANSPORT_FAILURE`) and keeps its existing message fallbacks; the boundary tuple gains the typed base class, which is already covered by `RuntimeError` in the tuple, so the job file needs no edit.
- Observed and left out of scope: a Tier-3 benchmark module imports the SDK (`applications/local_workspace_application/benchmarks/local_model_qualification/environment.py@H:108`, `F-64`); it is outside A1 and outside the row's AST scope (`intergrax/**` minus provider packages, and `agents/**`).

Negative tests: NT-RW-RUN-07 (section "Negative tests").

## Write set

Branch rule: product commits of this ticket go only to branch `arch-freeze/rw-train`; no other branch receives a product path of this ticket. The list below equals `RW_PLAN.md` appendix A1 block `T-AF2-RW-RUNTIME (T-IGX-051-D)` (RW-PLAN v13): 17 firm paths, 9 of them product paths, 0 conditional. Verified at `H`: 7 files exist, the 3 globs match 26 (`contracts/*`), 40 (`providers/*`, recursive) and 3 (`runtime/llm/*`) tracked files, and every new test path has an existing parent directory.

### Product paths (ACT-N11), edited
```
intergrax/llm_adapters/base/base_llm_adapter.py
intergrax/llm_adapters/_shared/provider_stream_admission.py
intergrax/llm_adapters/_shared/openai_completion_mapping.py
intergrax/multimedia/images_loader.py
intergrax/multimedia/image_smart_loader.py
agents/model_routing_qualifier/qualification_types.py
```

### Product paths (ACT-N11), globs (existing files edited and new files created under them)
```
intergrax/llm_adapters/contracts/*
intergrax/llm_adapters/providers/*
intergrax/runtime/llm/*
```

### Tests and support (not ACT-N11 paths), edited
```
tests/qualification/ebh_6/_findings_register.py
```

### Tests and support, new
```
tests/unit/llm_adapters/test_rw_runtime_stream_open_breaker.py
tests/unit/llm_adapters/test_rw_runtime_timeout_enforced.py
tests/unit/llm_adapters/test_rw_runtime_typed_error_contract_v1.py
tests/unit/llm_adapters/test_rw_runtime_served_model_attribution.py
tests/unit/llm_adapters/test_rw_runtime_refusal_normalization_matrix.py
tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py
tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py
```

### Docs
```
(none)
```

**Forbidden (a need for any of these stops the work):** `intergrax/llm_adapters/_shared/resilience.py`, `_shared/retry.py`, `_shared/call_lifecycle.py`, `_shared/adapter_response_builders.py`, `_shared/stream_finish_signal.py`, `base/usage_log.py`, `governance/quota.py`, `registry/*` (including `profile.py`), `agents/model_routing_qualifier/steps/model_routing_job.py`, the gate scripts `scripts/maintenance/check_agents_no_vendor_sdk_imports.py` and `check_integration_vendor_imports.py` (banned-list extension is `T-AF2-RW-GATES`), `intergrax/contracts/**`, existing tests other than the register above, `scripts/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, PDC-X and IIAC record directories, Docker or vendor work. A need for any of them is an amendment through the orchestrator, not a silent edit.

## Negative tests

Basis codes are RW-PLAN's (B behavioural, S structural, C conditional on a step-0 measurement). Red-first rule (RW-PLAN section d, binding for B and S): the test is written first and run once, alone, under the shared pytest lock, on a detached checkout of the pre-fix product tree; the test must be collected (an absent new API makes it fail on an assertion, so each test resolves new modules inside the test body with an explicit existence assertion, never at import); the failing assertion is the one in the row; the failing node is not a known-red row and not an already-red gate; the same test passes at the ticket's head. Logs go under `.tmp/session/arch-freeze/rw-runtime/red-first/`.

| Row | Item | Basis | Test file | Planned tests (names to be confirmed by the implementer) | Red-first reason at base |
|---|---|---|---|---|---|
| NT-ROW-103 / NT-RW-RUN-01 | EX-L-08 | B | `tests/unit/llm_adapters/test_rw_runtime_stream_open_breaker.py` | first-iteration failure trips the breaker (threshold 1, next call raises the circuit-open error); a retriable first-iteration failure is retried within `max_retries` and the factory is called `max_retries + 1` times; controls: a failure after the first event is not retried and is not recorded; permits released exactly once | at `H` the wrapper returns before any SDK call (`F-02`, `F-03`, `F-05`): no failure recorded, no retry |
| NT-ROW-104 / NT-RW-RUN-02 | EX-D3-01 | B | `tests/unit/llm_adapters/test_rw_runtime_timeout_enforced.py` | parametrized over adapters with a fake client: the effective timeout (and its clamp by an active deadline) reaches the fake call, buffered and streamed; a fake SDK that honours the forwarded value and hangs raises its timeout error within a bound and the adapter surfaces a typed timeout error; adapters whose mechanism is `NONE` raise the capability error when a timeout is set | no provider call site passes a timeout at `H` (`F-14`, `F-18`) |
| NT-ROW-105 / NT-RW-RUN-03 | EX-D3-03 | B | `tests/unit/llm_adapters/test_rw_runtime_typed_error_contract_v1.py` | the module exists and exposes version 1 (assertion, not import); one fixture per category and provider maps to the typed class with provider, model, status and retry-after; pass-through of the pre-existing typed errors keeps identity; retry/failover parity `is_retriable_provider_error(typed) == is_retriable_provider_error(raw)`; mid-stream error mapped; the qualifier classifies a typed model-not-found error without importing the SDK | no typed contract at `H` (`F-19`) |
| NT-ROW-106 / NT-RW-RUN-04 | EX-L-07 | B | `tests/unit/llm_adapters/test_rw_runtime_refusal_normalization_matrix.py` | cells for `generate_with_tools`, `stream_with_tools` of every adapter that implements them and the Bedrock invoke-family paths, for every signal the provider has; completeness self-check over the shipped adapters | only two methods are pinned at `H` (`F-31`, `F-32`); whether a cell fails is UNVERIFIED (Q-RUN-07) |
| NT-ROW-107 / NT-RW-RUN-05 | PDC-D3-R005 | B | `tests/unit/llm_adapters/test_rw_runtime_served_model_attribution.py` | per adapter with a fake SDK response whose reported model differs from the requested one: `response.served_model` equals the reported model; `LLM_CALL` accounting (`call.model` at `end_call`) records the served model; SDKs that report none give `served_model is None` and the requested model stays in accounting | requested model is recorded at `H` (`F-23`, `F-25`) |
| NT-ROW-108 / NT-RW-RUN-06 | PDC-D3-R019 | S | `tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py` | AST over `intergrax/llm_adapters/providers/*.py` finds every `DEFAULT_MODEL` constant; each has a VBN register row whose location names the file; counters stay 0 | no row exists at `H` (`F-49..F-56`, `F-58`) |
| NT-ROW-036 / NT-CTR-01-1 | FRZ-CTR-01 | C | `tests/unit/llm_adapters/test_rw_runtime_capability_flag_parity.py` | parametrized over every shipped adapter: a declared streaming or tool-streaming capability without the method fails, and an implemented method with the flag false fails | characterization: per-adapter result recorded at step 0, not claimed red (Q-RW-08 rule) |
| NT-ROW-005 / NT-RW-RUN-07 (NT-CTR-02-1; design of NT-WF-F01..F03) | A4-018 | S | `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py` | AST over `intergrax/**` minus `intergrax/llm_adapters/providers/**` and over `agents/**`: no import of `ollama` at any nesting level; image captioning with an injected fake adapter exposing `describe_image` succeeds under a meta-path finder that raises on `ollama` | fails on `images_loader.py` and `qualification_types.py` at `H` (`F-39`, `F-41`) |
| NT-ROW-037 | FRZ-CTR-02 | S | same file as NT-ROW-005 | as A4-018 | as A4-018 |
| NT-ROW-039 | FRZ-CTR-06 | S | same file as NT-ROW-005 | as A4-018 | as A4-018 |

Counts: 10 RW-PLAN rows, 7 new test files, 7 NT ids (NT-RW-RUN-01..07) plus the NT-CTR aliases. `NT-CFG-06-1` of `PT-FC3` (NT-ROW-048) is defined as `NT-RW-RUN-07`; this ticket owns the test. The red-first set is 9 rows (B: rows 103-107; S: rows 005, 037, 039, 108); row 036 is basis C and is not counted as red-first. The exact number of test functions (parametrized cells) is fixed at step 0.

## 11. Regression scope and test commands

Nothing was run in this draft. One pytest process at a time under the shared lock `D:/Projekty/_wt/_locks/pytest.lock` (created and removed by the same session), multiple paths in one command, no broad suite locally, no live or network-marked test (C10/C15).

RW-PLAN regression scope for this ticket (appendix, section e): `tests/unit/llm_adapters`, `tests/qualification/ebh_5`, `tests/qualification/ebh_6`, `tests/unit/multimedia`; L1 common gates (`check_intergrax_no_applications_imports.py`, `check_agent_registry_bypass.py`, `check_agents_no_tier3_imports.py`, `check_agents_no_vendor_sdk_imports.py`, `check_integration_vendor_imports.py`, `check_dependency_ownership.py`, `igx_gov ticket`, `verify_arch_freeze_plan_traceability.py`, `arch_freeze_b1_closed_world_inventory.py --strict-certify`) and L2 common (`tests/unit` gate-marked in CI only; `tests/unit/governance` one process). Expected-red entries of the train (table 1 of RW-PLAN) stay expected; this ticket adds none.

Proposed additions beyond the RW-PLAN paths, listed because the change touches what they pin (Q-RUN-10, orchestrator ruling): `tests/qualification/pdc_x/d3/test_d3_f01_stream_guard.py`, `test_d3_f02_refusal_paths.py`, `test_d3_r009_refusal.py`; `tests/unit/runtime/architecture/test_enterprise_scale_resilience_w4_d_provider_cancellation.py` and `..._w2_dependency_isolation_inventory.py`; `tests/qualification/harness_02/test_harness_02_gates.py`; `tests/unit/runtime/execution/deadline_authority/test_harness_02_r1a_qualification.py` and `..._r1b_qualification.py`; `tests/qualification/roadmap_replay_x` ledger and counter tests (they consume the findings register, `F-62`); the existing `tests/unit/multimedia/test_image_smart_loader.py` (inside the multimedia directory, `F-46`).

Ratchet check: this ticket edits no path under `intergrax/contracts/**`; the new and changed code under `intergrax/llm_adapters/contracts/*` uses no `Any`, `object` or reflection dispatch, and the wave-packet contract check (`measure_typ05.py` plus `typ_measure.py`) must report nothing new. If an edit under `intergrax/contracts/` becomes necessary it is an amendment and the check is mandatory.

Commands (to be run by the implementer, not by the drafter):
- Red-first (S0): the seven new test files in one `uv run pytest <seven paths> -p no:cacheprovider` on a detached worktree at the base commit (location `<parent>/_wt/intergrax/rw-runtime`, slug within the CLAUDE.md limit), log under `.tmp/session/arch-freeze/rw-runtime/red-first/`.
- After the change (S8): the same seven files; then one command over `tests/unit/llm_adapters tests/unit/multimedia`; then one over the two `ebh` directories (these are large; the full runs belong to GitHub Actions at M3 and the local command may be restricted to the files that import the register, the inventory and the adapter modules, with the restriction recorded); then the proposed additions; then the L1 gates; `uv run ruff check` and `uv run pyright` on the changed Python files.

## 12. Order of work inside the ticket

S0 measure and write the seven tests, red-first run, step-0 tables (adapter timeout mechanisms, parity result, matrix cells red or green, served-model availability per SDK, inventories that enumerate `llm_adapters/contracts`). S1 Part C contract and mapping (other parts depend on the typed classes). S2 Part A priming. S3 Part B timeout forwarding. S4 Part E served model. S5 Part D matrix fixes, if any cell is red. S6 Part G SDK boundary. S7 Part F flag fixes and register rows. S8 regression and gates. S9 closeout of the ticket record (red-first logs kept for the auditor). A stop condition at any step: a needed file outside the write set.

## 13. Overlaps and merge order

RW-PLAN lists no write-path overlap for this ticket (lane 3, after PT-WA and PT-WB; the lane table says only "lane 2 full"). `tests/qualification/ebh_6/_findings_register.py` is read by the roadmap-replay ledger but edited by no other train ticket per A1. Rebase onto the train after PT-WB; merge into `arch-freeze/rw-train` with a merge commit after the head SHA equals the pushed SHA and every red CI job is a registered expected-red entry.

## 14. ACT-N11 / IIAC consequence

Every product path above breaks the IIAC activation until the single RB-2 re-pin (memory of the program, `ACT-N11`); this ticket does not perform that re-pin and must not merge outside the wave.

## 15. ARCH-FREEZE impact and what closes each item

Nothing is closed by this draft or by the implementation alone. Per item the closing condition is: the red-first log and the green run at the wave head, the independent audit of the exact wave head (`PT-Q02`), and for DS-EXC items the W7 independent session (`T-B7-ECAP-2C`). EX-D3-01 and EX-D3-03 carry an Operator-decision reopen condition in the DS-EXC package; the plan of record section E has already fixed the disposition as REMEDIATION, so this plan builds, and Q-RUN-02 and Q-RUN-03 only fix the details. PDC-D3-R018 is contributed to (the typed errors and the unchanged quota admission order), not owned (Q-RUN-01).

## 16. Independent audit design

Audit line (RWA-05, audit flag A4-018 bound to `PT-Q02`): the independent audit of this ticket is a separate session at the exact wave head; the implementer does not PASS. Mutation checks for the auditor: revert the first-event priming and NT-RW-RUN-01 must fail; remove the timeout argument from one provider and NT-RW-RUN-02 must fail; remove one category mapping and NT-RW-RUN-03 must fail; restore the requested model in `call.model` and NT-RW-RUN-05 must fail; reintroduce an SDK import in `images_loader.py` and NT-RW-RUN-07 must fail; delete one register row and NT-RW-RUN-06 must fail.

## 17. Risks

| ID | Risk | Handling |
|---|---|---|
| R-01 | Priming moves the first-token wait into the attempt: longer permit hold; a lifecycle wrapper `finally` must still run on a failed first pull | the helper closes the iterator on error; controls in NT-RW-RUN-01; the existing dependency-admission and cancellation tests are in the proposed regression list |
| R-02 | Typed mapping changes the failure label in usage metrics (the raw SDK class name becomes the typed class name) and could change failover classification | parity test in NT-RW-RUN-03; step 0 lists tests that pin the labels (`test_llm_metrics.py`, `test_llm_adapter_usage_log.py` are the candidates) and, if one pins a raw name, the change is reported as a write-set gap because only listed files may be edited |
| R-03 | About 37 `end_call` sites and nine provider files are edited: merge friction and review size | one base helper; edits are mechanical; split into commit groups by part |
| R-04 | A needed file lies outside A1 (`resilience.py`, `retry.py`, `usage_log.py`, `call_lifecycle.py`, `adapter_response_builders.py`, `stream_finish_signal.py`) | the design works around each (sections 3.1-3.5); if a design assumption fails the work stops and the orchestrator amends A1 |
| R-05 | SDK timeout mechanisms are a hypothesis (nothing was checked in this draft) | step 0 table; fake SDK fixtures follow the repository's existing fakes |
| R-06 | NT-ROW-106 may be green at base | Q-RUN-07: recorded as not red, re-disposed by the orchestrator, never reported as red |
| R-07 | The register change is read by roadmap-replay and VBN counters outside the listed regression paths | proposed additions in section 11; classification is a draft default pending the VBN dossier |
| R-08 | The existing `test_image_smart_loader.py` pins the `images_loader` seam | option A keeps the seam; option B would need a test edit that is not in A1 (Q-RUN-09) |
| R-09 | No live provider run is possible under the C10/C15 freeze, so the SDK behaviour is only fake-modelled | stated as a limitation in the ticket; no live claim anywhere |

## 18. Rollback

All changes land only on `arch-freeze/rw-train` as a PR merged with a merge commit; rollback is a revert of that merge. The new contract and provider modules are additive; the changed behaviours (stream attempt priming, typed errors, timeout forwarding with fail-closed adapters, served-model accounting) have no data migration.

## 19. Open questions with draft defaults

| ID | Question | Draft default |
|---|---|---|
| Q-RUN-01 | PDC-D3-R018 ownership: the dispatch and plan of record section E assign it to this node; RW-PLAN v13 dedupe D-07 reassigns it to `T-AF2-RW-SUPPLY` and lists it as contributed here | follow RW-PLAN: contributed, no `quota.py` edit |
| Q-RUN-02 | Adapters whose SDK offers no per-call or client-level timeout: fail closed with the capability error when `timeout_sec` is set, or record "not enforced" and continue | fail closed |
| Q-RUN-03 | Typed error message and `__cause__`: SDK message text excluded from the typed message (kept on the cause) | excluded |
| Q-RUN-04 | Mid-stream errors after the first event: typed, not retried, not recorded by the breaker | as stated |
| Q-RUN-05 | `LLMRateLimitError` and `LLMCircuitOpenError` are documented as v1 members by reference because `resilience.py` is outside A1 (the alternative is an A1 amendment to make them subclasses) | by reference |
| Q-RUN-06 | `served_model` is set with `dataclasses.replace` and attributed through a base helper because `adapter_response_builders.py`, `usage_log.py` and `call_lifecycle.py` are outside A1 (the cleaner end-call parameter needs an A1 amendment of up to three paths) | no amendment |
| Q-RUN-07 | If every NT-ROW-106 cell is green at base: record "not red" and ask for a re-disposition (the row is basis B) | do not self-reclassify |
| Q-RUN-08 | Classification and status of the 8 provider default rows (draft `ALLOWED — PROVIDER ADAPTER BOUNDARY`, `CANDIDATE-RESOLVED`); `LLMProfile.lab` and the rag default embedding provider named by the DS-EXC row are outside A1 | add the 8 rows; the other two are reported to `T-AF2-VBN-DOSSIER` |
| Q-RUN-09 | Image captioning: option A keeps the `images_loader.transcribe_image` seam and moves the SDK call into a provider module; option B routes through the adapter and edits the existing test (not in A1) | option A |
| Q-RUN-10 | Regression additions beyond the RW-PLAN paths (section 11) | add them |
| Q-RUN-11 | The A1 glob `intergrax/runtime/llm/*` is firm but the design expects no edit there | keep as ceiling, report unused |

## 20. Authorization requests (to the orchestrator, none is assumed)

- The approval block of this ticket (delegate-written; this draft contains none).
- The product-path authorization of the node (the id is not stated in the inputs of this draft).
- A worktree outside the repository at `<parent>/_wt/intergrax/rw-runtime` for the implementer, and a detached worktree at the base commit for the red-first run; pytest lock use as in section 11.
- A ruling on Q-RUN-01 before the wave packet (ownership of PDC-D3-R018) and on Q-RUN-10.
