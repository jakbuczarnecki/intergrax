# PDC-P7 - independent audit of domain D3 (MODEL-RUNTIME)

| Field | Value |
| --- | --- |
| **Stage** | PDC-X / P7, domain D3 (`PDC-D3-R001..R024`) |
| **Role** | `IA-D3`: independent auditor (OD-PDC-IIAC-SCOPE, Option A). Clean session; never implemented, signed or certified anything of D3; no product, test, script or governance file was edited |
| **Auditor session** | `PDC-P7-IA-D3-INDEPENDENT-AUDITOR-e45cd4f2-8d05-4285-9958-55e3d9c5448b` |
| **Audited SHA** | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (origin/development at audit start; the auditor branch `claude/pdc-x-ia-d3-d10590` descends from it) |
| **Policy** | `IIAC-2026-11-05.1`; manifest under `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/` |
| **Dossier verdict** | **PARTIAL** (IIAC manifest value `CONDITIONAL`, see limitation LIM-verdict-mapping) |
| **Rows** | 12 PASS, 3 PASS_SCOPED, 9 PARTIAL, 0 FAIL (24 rows) |
| **Findings** | 1 HIGH, 6 MEDIUM, 6 LOW, 3 INFO |
| **Certification** | none: this report certifies nothing and promotes no FRZ criterion |

## 1. Summary

The declared-capability work of B-D3 (ADR-PDC-03) is real and mostly sound: the flag family, the typed `LLMCapabilityMismatchError`, the conformance gate, catalog-owned context windows and bounded failover hold under adversarial probing, and 12 of 14 product mutants are killed by the D3 suite. The domain does not reach PASS because of one HIGH and six MEDIUM findings:

- **F-D3-01 (HIGH):** the stream paths of the default OpenAI adapter, Gemini/Vertex and the LangChain Ollama adapter skip `_execute_streaming`, so they bypass the protected-work guard, the tenant quota and the cancel-transport registration while the registry declares `terminated` / `transport_closed` for them. The cancel test registers a stream by hand and cannot see it (mutant M12 survives). No in-tree streaming consumer exists, so the defect is latent.
- **F-D3-02, -03, -04, -07 (MEDIUM):** refusal normalization is not applied on streaming and several buffered paths; `timeout_sec` is a dead setting; the pre-call capability check has no production feature source (F-37 re-verified); there is no typed provider-error taxonomy.
- **F-D3-05, -06 (MEDIUM):** the `llm_adapters` unit suites and the TENANT-X replay that several rows name as closure evidence are red at this SHA (the 12 unit failures are stale tests that were already red before B-D3; the TENANT-X count change is outside D3).

Routed leads F-49 and F-50 are **resolved**, and the shared-slug defect class is closed (F-D3-14).

## 2. Method and independence

1. Read `OD-PDC-IIAC-SCOPE`, the 24 D3 rows of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, `PDC_OWNERSHIP_MATRIX.md`, ADR-PDC-03, the B-D3 / B-REG-T / B-REG-P / B-P6-J entries of `PDC_IMPLEMENTATION_WBS.md`, tickets T-IGX-036-K and PLAN_T-IGX-036-K, and the P7-00 dossier plus addenda 1..4 (routed leads only; none of their conclusions was reused without re-verification).
2. For every row: read the code at the cited owner path, ran the cited suites (one pytest process at a time, narrow paths, outputs in `.tmp/` and copied to `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/logs/`), and tried to break the claim with probes (`adversarial_scripts/`), a 14-mutant sample in a scratch worktree at the audited SHA, and control runs at `35b6772af~1` (before B-D3) to separate regressions from pre-existing reds.
3. Re-verified the routed leads F-05, F-37, F-42, F-49, F-50 at this SHA (section 5).
4. Independence: the auditor made no product, test, script or governance edit; the only repository changes are this report and the evidence directory. The mutation worktree and the control worktree live outside the repository (`D:/Projekty/_wt/intergrax/`).

## 3. Per-row verdicts

Verdicts use PASS / PASS_SCOPED (only Operator-authorized exclusions) / PARTIAL / FAIL.

| Row | Requirement | Verdict | Basis at the audited SHA | Findings |
| --- | --- | --- | --- | --- |
| R001 | Typed provider-neutral adapter ABI | **PARTIAL** | Protocol and base class present and used by all adapters; ABI-adjacent unit tests red (registry raises TypeError before the documented registration error) | F-D3-05 |
| R002 | ModelCatalog capability metadata | **PASS** | ebh_5 243 passed; catalog loads (64 records); one duplicate record | F-D3-11 (LOW) |
| R003 | Configuration-driven provider selection | **PASS** | ebh_5 243 passed; XD-01 provider swap by config; 199-slug catalog resolves | F-D3-14 |
| R004 | Routing and failover bounded | **PASS** | d3 R004 tests; mutant M04 killed; cancellation abort and KeyboardInterrupt stop the chain; streaming has no failover | F-D3-07 (classification) |
| R005 | Model version pinning and runtime attribution | **PARTIAL** | pin by model string and profile digest (p6_model green); served model not recorded; floating alias defaults | F-D3-10 |
| R006 | Streaming / tool-call streaming parity or explicit capability fail-clo | **PASS** | undeclared tool streaming raises a typed error before any call; both Ollama adapters declare False; mutants M02, M14 killed | F-D3-04, F-D3-08 |
| R007 | Structured outputs | **PASS** | structured-output unit suites green; not probed adversarially | F-D3-05 (suite status) |
| R008 | Tool-call normalization | **PASS** | strict_tool_call_validation and tool_call_identity suites green; not probed adversarially | F-D3-05 (suite status) |
| R009 | Refusal semantics: typed finish reason exists, refusal payload is a st | **PARTIAL** | normalizer correct and tested; stream and several buffered provider paths never feed it | F-D3-02 |
| R010 | Usage, token and cost accounting | **PASS** | usage tracker suites green; stream usage is estimated and labeled; three adapters' stream paths skip the quota check | F-D3-01 (cross-effect) |
| R011 | Cancellation explicit per provider (seam beside the ABI; Ollama has no | **PARTIAL** | typed outcome and per-provider declaration exist and the seam closes registered transports; default OpenAI, Gemini, Vertex and LangChain Ollama streams never register; mutant M12 survives | F-D3-01, F-D3-13 |
| R012 | Timeouts | **PARTIAL** | `timeout_sec` exists in the contract but is not enforced (probe) | F-D3-03 |
| R013 | Provider error taxonomy | **PARTIAL** | no typed taxonomy; heuristic classification | F-D3-07 |
| R014 | Context-window enforcement | **PASS** | ce_02 green (combined run); resolver order verified; mutant M07 killed | - |
| R015 | Capability mismatch fails closed | **PARTIAL** | adapter-level fail-closed holds; the runtime pre-check has no production feature source; mutant M03 survives | F-D3-04, F-D3-13 |
| R016 | Provider replaceability | **PASS** | ebh_5 243 passed; F-49 class closed | F-D3-14 |
| R017 | Local/offline models and native Ollama | **PASS_SCOPED** | native adapter suites green except 3 stale regex tests; stream path correct; live parity is Path B (OD-PDC-PATHB-01) | F-D3-05 |
| R018 | Tenant-aware model configuration | **PARTIAL** | tenant_x replay red (831 vs 819); quota fails open on a malformed value | F-D3-06, F-D3-12 |
| R019 | No vendor selection hidden in core (provider DEFAULT_MODEL constants,  | **PARTIAL** | classification evidence named by the matrix not found; defaults are floating aliases | F-D3-09 |
| R020 | Provider plugin registration layer (LLM-PROVIDER-PLUGIN-1 Planned) | **PASS_SCOPED** | OD-PDC-SCOPE-03 A authorizes the exclusion; no `LLMProviderRegistration` implementation exists | - |
| R021 | Live provider qualification | **PASS_SCOPED** | live provider qualification is Path B; live tests skip without credentials | - |
| R022 | Streaming capability flags misreport: six adapters stream but supports | **PASS** | gate and 7 adapters verified; mutants M01, M13 killed; wrappers follow the chain | F-D3-13 (LOW) |
| R023 | Context-window tables duplicated inside providers and ModelCatalog | **PASS** | 7 provider tables, all read only as `legacy_windows`; catalog wins; mutant M07 killed | F-D3-11 (LOW) |
| R024 | EBH-5 replaceability gates red at HEAD: proof matrix/registry drift (4 | **PASS** | ebh_5 243 passed with the `integrations-duckdb` extra (4 fail without it) | F-D3-15 |

Rows closed by `ALREADY_CERTIFIED_REUSABLE` replay (R007, R018) were replayed, not trusted: R007 passes its suites; R018 does not (F-D3-06). R020 and R021 are scoped by OD-PDC-SCOPE-03 A and OD-PDC-PATHB-01.

## 4. Findings

| ID | Severity | Rows | Title |
| --- | --- | --- | --- |
| F-D3-01 | HIGH | R011, R010, R018 | Streaming on the default OpenAI adapter (and Gemini/Vertex, LangChain Ollama) bypasses the provider guard; the declared cancel capability is not backed by those stream paths |
| F-D3-02 | MEDIUM | R009 | Refusal / content-filter normalization holds only for a subset of provider paths; every streaming path and several buffered paths report COMPLETED with empty content |
| F-D3-03 | MEDIUM | R012 | `LLMCallConfig.timeout_sec` is never enforced or forwarded to the SDK call |
| F-D3-04 | MEDIUM | R015, R006 | The pre-call capability check has no production feature source; the XD-01 gate is satisfiable by a dormant caller (carries F-37) |
| F-D3-05 | MEDIUM | R001, R007, R008, R010, R012, R013, R017 | The `llm_adapters` unit suites cited as closure evidence are not green at the audited SHA |
| F-D3-06 | MEDIUM | R018 | The TENANT-X replay gate cited for R018 is red: carrier count 831 vs 819 |
| F-D3-07 | MEDIUM | R013, R004 | No typed provider-error taxonomy: retry and failover classify by `status_code` and class-name substrings |
| F-D3-08 | LOW | R006 | LangChain Ollama `stream_messages` silently re-issues the model call after a mid-stream failure |
| F-D3-09 | LOW | R019 | The VBN classification of provider defaults named as R019's test case is not present |
| F-D3-10 | LOW | R005 | Runtime attribution records the requested model id, never the model the provider reports serving |
| F-D3-11 | LOW | R002, R023 | `model_catalog.yaml` carries a duplicate record and the loader accepts duplicates and non-positive windows |
| F-D3-12 | LOW | R018 | Tenant token quota is one process-wide cap that silently disables itself on a malformed value |
| F-D3-13 | LOW | R022, R015, R011 | Test strength gaps in the D3 gates (2 of 14 mutants survive; the AST gate accepts any call) |
| F-D3-14 | INFO | R003, R016, R024 | Positive: F-49 and F-50 are resolved; the shared-slug defect class is closed; EBH-5 is green |
| F-D3-15 | INFO | - | Environment notes and known reds |
| F-D3-16 | INFO | R006, R022 | Documentation drift: the architecture and plan docs do not describe the declared-capability contract; the B-D3 commits carry an AI co-author trailer that predates the repo rule |

### F-D3-01 - HIGH - Streaming on the default OpenAI adapter (and Gemini/Vertex, LangChain Ollama) bypasses the provider guard; the declared cancel capability is not backed by those stream paths

- **Requirements:** PDC-D3-R011, PDC-D3-R010, PDC-D3-R018
- **Verification:** PROBE_CONFIRMED + MUTANT_SURVIVED
- **Evidence:** openai_responses_adapter.py:676 and :778, gemini_adapter.py:218/:248/:412 and ollama_adapter.py:391 wrap the SDK stream only in `_provider_dependency_attempt()`. The other streaming adapters (openai chat/compat/azure, claude, mistral, cohere, bedrock, native ollama) go through `BaseLLMAdapter._execute_streaming`, which runs `assert_protected_provider_call_allowed()`, the deadline check, `check_llm_tenant_quota`, rate limit / circuit breaker / retry, the external-operation attempt and `stream_with_external_operation_lifecycle` (the stream-transport registration used by the cancel seam). Probe `probe_stream_guard.py` (guard patched to deny): `OpenAIChatResponsesAdapter.generate_messages` denied, `stream_messages` ALLOWED, `OpenAIChatCompletionsAdapter.stream_messages` denied. The registry declares OPENAI/GEMINI/VERTEX_GEMINI as `terminated` and OLLAMA as `transport_closed` (registrations/external_operation_seams.py), but those stream paths never register a transport. The R011 test `test_declared_cancel_capability_equals_observed_outcome` registers a stream into the seam by hand, so it cannot see this: mutant M12 (remove the `stream_registry.register(...)` call in provider_stream_admission.py) SURVIVES all 150 D3 tests. The docstring of `contracts/llm_adapter.py` and HARNESS-02 `H02-llm-provider-boundary` state that every physical provider access runs the guard.
- **Impact:** For the default `openai` provider (registered adapter = OpenAIChatResponsesAdapter) a streamed call is not subject to the protected-work/deadline guard, the tenant token quota, rate limiting, circuit breaker or retry, and cannot be cancelled through the registered seam although the registry declares `terminated`. Mitigation observed: no in-tree runtime caller of `stream_messages`/`stream_with_tools` exists (grep over intergrax/, agents/, applications/ finds only wrappers), so the defect is latent until a streaming consumer is wired.
- **Reproduction:** `uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/adversarial_scripts/probe_stream_guard.py ; mutant M12 via adversarial_scripts/run_mutants.py M12`
- **Suggested repair scope:** Route the stream paths of openai_responses_adapter, gemini_adapter and ollama_adapter through `_execute_streaming` (same admission, quota, resilience and transport registration as the other adapters); add a test that drives each shipped adapter's real `stream_messages` with a fake SDK and asserts (a) the guard is consulted and (b) the stream transport is registered with the provider seam, and a source gate that forbids an adapter stream method that calls the SDK outside `_execute_streaming`.
- **Owner path:** intergrax/llm_adapters/providers/ (openai_responses_adapter.py, gemini_adapter.py, ollama_adapter.py); tests/qualification/pdc_x/d3/test_d3_r011_cancel_outcome.py

### F-D3-02 - MEDIUM - Refusal / content-filter normalization holds only for a subset of provider paths; every streaming path and several buffered paths report COMPLETED with empty content

- **Requirements:** PDC-D3-R009
- **Verification:** PROBE_CONFIRMED + CODE
- **Evidence:** `build_adapter_response` normalizes whatever it is given, but most provider paths never pass a finish reason or refusal to it: OpenAI chat `stream_messages` (openai_chat_completions_adapter.py:128-175) ignores `delta.refusal` and `finish_reason`; the plain `generate_messages`/`stream_messages` of gemini (:158, :230, :262), cohere (:133, :189), bedrock converse/invoke (:515, :551, :614, :662), LangChain Ollama (:346, :403, :421) and the streams of mistral, openai-responses, claude and native ollama build the final response with the default finish reason. Probe `probe_refusal_stream.py`: a stream carrying `delta.refusal` and a stream ending `finish_reason=content_filter` both end as `finish=COMPLETED refusal=None content=''`. The R009 tests exercise `parse_finish_reason`, `build_adapter_response` and the OpenAI/Claude non-stream mappers directly, not the provider stream paths; the `SAFETY`/`guardrail_intervened`/`ERROR_TOXIC` vocabulary of `parse_finish_reason` has no producer on the Gemini, Bedrock and Cohere plain paths.
- **Impact:** A refused or filtered streamed answer is indistinguishable from an empty successful answer; downstream governance keyed on REFUSAL/CONTENT_FILTER never fires on streams.
- **Reproduction:** `uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/adversarial_scripts/probe_refusal_stream.py`
- **Suggested repair scope:** Read the provider stop/finish/refusal signal in every stream and plain-generate path and feed it to `build_adapter_response`; extend the R009 tests to drive each shipped adapter's stream and plain paths with a fake SDK refusal and content-filter payload.
- **Owner path:** intergrax/llm_adapters/providers/*_adapter.py; tests/qualification/pdc_x/d3/test_d3_r009_refusal.py

### F-D3-03 - MEDIUM - `LLMCallConfig.timeout_sec` is never enforced or forwarded to the SDK call

- **Requirements:** PDC-D3-R012
- **Verification:** PROBE_CONFIRMED + CODE
- **Evidence:** `BaseLLMAdapter._execute` / `_execute_streaming` (base_llm_adapter.py:275-345) compute `effective_timeout` and only reject a budget that is already <= 0; `timeout_sec` is then carried in `effective_config`, which `call_with_retry` and `execute_with_resilience` do not read (retry.py uses max_retries/retry_on_status/backoff only; resilience.py has no timeout). No provider adapter passes a timeout to its SDK call. Probe `probe_timeout.py`: `OpenAIChatCompletionsAdapter(timeout_sec=0.1, max_retries=0)` with an SDK call that sleeps 1.0 s returned 'late' after 1.06 s and the SDK kwargs contain no timeout. No unit test in tests/unit/llm_adapters or the D3 suite exercises `timeout_sec`.
- **Impact:** A configured per-call timeout is a dead setting: a hung provider call is bounded only by the SDK default or an outer execution deadline that is consulted at admission.
- **Reproduction:** `uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/adversarial_scripts/probe_timeout.py`
- **Suggested repair scope:** Apply the effective timeout to the SDK request (per-call `timeout=` kwarg or client option) in `_execute`/`_execute_streaming` and add a test with a slow fake SDK; or remove `timeout_sec` from the contract and record the exclusion.
- **Owner path:** intergrax/llm_adapters/base/base_llm_adapter.py, intergrax/llm_adapters/_shared/resilience.py; new test under tests/qualification/pdc_x/d3/

### F-D3-04 - MEDIUM - The pre-call capability check has no production feature source; the XD-01 gate is satisfiable by a dormant caller (carries F-37)

- **Requirements:** PDC-D3-R015, PDC-D3-R006
- **Verification:** CODE + TEST_RUN + MUTANT_SURVIVED
- **Evidence:** `LLMAdapterRegistry.require_capabilities` is called in production only by `_require_llm_features` in intergrax/applications/_shared/llm_resolver.py:300, and `required_llm_features` has no caller outside tests (grep `required_llm_features=` over intergrax/, agents/, applications/: none; `llm_profile_source=`: none). The runtime resolver `intergrax/runtime/wiring/llm_resolver.py:166` (used by XD-01 itself) has no such parameter. `test_xd01_required_model_capabilities_are_checked_at_composition_time` only asserts that some production file calls `require_capabilities`, which the dormant `_require_llm_features` satisfies. The adapter-level raise in `BaseLLMAdapter.stream_with_tools` (before provider I/O) is the real fail-closed backstop and holds. Mutant M03 (failover falls back to `adapters[0]` when no chain member declares the feature) SURVIVES because the base class raises anyway; an under-reporting third-party member would stream undeclared.
- **Impact:** `AgentContract.required_llm_features` is accepted and normalized but never reaches adapter resolution at runtime, so the ADR-PDC-03 statement that the runtime checks requested features before the call is true only for the adapter methods themselves.
- **Reproduction:** `grep -rn "required_llm_features=" intergrax agents applications ; mutant M03 via adversarial_scripts/run_mutants.py M03`
- **Suggested repair scope:** Pass the agent contract's `required_llm_features` into the resolver on the runtime composition path (both resolvers), make XD-01 drive that path, and pin the failover pre-check with a test using `UnderReportingStreamAdapter` (declared False, streams anyway).
- **Owner path:** intergrax/runtime/wiring/llm_resolver.py, intergrax/applications/_shared/llm_resolver.py, intergrax/agent_distribution/ (composition); tests/qualification/pdc_x/xd/test_xd_01_agent_model.py

### F-D3-05 - MEDIUM - The `llm_adapters` unit suites cited as closure evidence are not green at the audited SHA

- **Requirements:** PDC-D3-R001, PDC-D3-R007, PDC-D3-R008, PDC-D3-R010, PDC-D3-R012, PDC-D3-R013, PDC-D3-R017
- **Verification:** TEST_RUN (control at 35b6772af~1)
- **Evidence:** `uv run --extra <all llm extras> pytest tests/unit/llm_adapters tests/qualification/ce_02 tests/qualification/pdc_x/d3`: 12 failed / 692 passed / 8 skipped. Eight failures are deterministic and also red at the control checkout 35b6772af~1 (13 failed there, one of them fixed since): 4 x `test_fixture_strict_adapter_parity` (platform_proofs `IncidentInvestigatorAgent` cannot be instantiated: abstract `run`), 3 x `test_ollama_tool_calling::test_unsupported_tool_choice_rejects_before_provider` (regex stale since d5f5cde99; the message now ends 'or a forced function choice'), 1 x `test_registry_model_contract::test_invalid_adapter_missing_model_fails_clearly` (the registry raises TypeError from the Protocol isinstance check before the documented `LLMAdapterRegistrationError`; the AttributeError branch of `_resolve_adapter_model_id` is unreachable through `create`). 4 further tests (`test_missing_optional_llm_dependency_is_controlled`, 3 x `test_provider_creation_surfaces_dependency_error_without_sdk`) fail only in the combined run with the SDK extras installed (state dependent) and pass in isolation (28 passed). In the default environment (no extras) `test_adapter_conformance.py` and `test_structured_output_flags.py` fail at collection (`import anthropic`).
- **Impact:** R001, R007, R008, R010, R012, R013 and R017 name 'llm_adapters unit suites' as requalification evidence; a green replay cannot be claimed, and two stale tests encode a contract (error type, message) that differs from the product.
- **Reproduction:** `uv run --extra llm-anthropic --extra llm-mistral --extra llm-ollama --extra llm-langchain-ollama --extra llm-gemini --extra llm-bedrock --extra llm-cohere-native --extra llm-openai pytest tests/unit/llm_adapters tests/qualification/ce_02 tests/qualification/pdc_x/d3 -q -p no:randomly`
- **Suggested repair scope:** Repair or retire the stale tests (regex, protocol order, fixture agent `run`), guard the two extra-dependent modules with `importorskip`, and isolate the registry-state tests; then record a green replay.
- **Owner path:** tests/unit/llm_adapters/ (test_ollama_tool_calling.py, test_registry_model_contract.py, test_fixture_strict_adapter_parity.py, test_adapter_conformance.py, test_structured_output_flags.py, test_llm_provider_registry.py, test_p2_002_c_provider_registration.py); platform_proofs/scenarios/ai_incident_investigation/

### F-D3-06 - MEDIUM - The TENANT-X replay gate cited for R018 is red: carrier count 831 vs 819

- **Requirements:** PDC-D3-R018
- **Verification:** TEST_RUN (control at 35b6772af~1 green)
- **Evidence:** `uv run --extra integrations-duckdb pytest tests/qualification/tenant_x`: 1 failed / 193 passed. `test_tenant_x_p0_current_head.py::test_tx_p0_r3_head_q02_class_field_carrier_count_stable` asserts 831 == 819 (`TENANT_X_REPLAY_HEAD_CARRIER_TOTAL`); the same test is green at the control checkout 35b6772af~1. The D3 diff adds no tenant-bearing class fields (the 4 added lines mentioning tenant in intergrax/llm_adapters are re-ordered imports), so the +12 comes from other PDC-X domains and is not isolated here.
- **Impact:** R018 (tenant-aware model configuration) is closed 'by tenant_x replay'; that replay is red and unclassified, so R018 cannot be requalified as claimed.
- **Reproduction:** `uv run --extra integrations-duckdb pytest tests/qualification/tenant_x/test_tenant_x_p0_current_head.py -q -p no:randomly -k carrier_count_stable`
- **Suggested repair scope:** The owner of the TENANT-X replay classifies the +12 carriers in `_replay_x_carrier_reconciliation.py` (or fixes the fields) and updates the constant; no D3 product change.
- **Owner path:** tests/qualification/tenant_x/_replay_x_carrier_reconciliation.py

### F-D3-07 - MEDIUM - No typed provider-error taxonomy: retry and failover classify by `status_code` and class-name substrings

- **Requirements:** PDC-D3-R013, PDC-D3-R004
- **Verification:** PROBE_CONFIRMED + CODE
- **Evidence:** The row claims 'typed provider errors'. `intergrax/llm_adapters` defines no normalized provider error classes (auth, rate limit, timeout, invalid request, content filter, server); SDK exceptions propagate raw. `_is_retryable` (_shared/retry.py:81-99) decides by the `status_code` attribute and `any(token in type(exc).__name__.lower() for token in ('timeout','connection','rate','overloaded'))`. Probe `probe_failover.py` with the default failover policy: an exception class named `GenerateContentError` (a non-transient 400-style error) and one named `SeparateTenantError` both fail over and the same request reaches the secondary provider; status 401 stops; `CooperativeCancellationAbort` and `KeyboardInterrupt` stop (positive). Streaming calls have no failover at all (`FailoverLLMAdapter.stream_messages`/`stream_with_tools` pick the first declaring member and delegate).
- **Impact:** Whether a request and its content are re-sent to another provider depends on a substring of a vendor class name; the taxonomy the matrix claims does not exist as a contract.
- **Reproduction:** `uv run python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/adversarial_scripts/probe_failover.py`
- **Suggested repair scope:** Introduce a small typed provider-error contract mapped at the adapter boundary (or classify by an explicit allowlist of SDK types and status codes) and test it per provider with fake SDK errors.
- **Owner path:** intergrax/llm_adapters/_shared/retry.py, intergrax/llm_adapters/contracts/ (new error contract), intergrax/llm_adapters/registry/failover_policy.py

### F-D3-08 - LOW - LangChain Ollama `stream_messages` silently re-issues the model call after a mid-stream failure

- **Requirements:** PDC-D3-R006
- **Verification:** PROBE_CONFIRMED
- **Evidence:** ollama_adapter.py:413 `except Exception:` after partial output falls back to `self.chat.invoke(...)` and emits the whole answer again. Probe `probe_ollama_stream.py`: partials 'par','tial' then a ConnectionError yield a partial 'FULL REISSUED ANSWER' and a FINAL containing only the second answer; `invoke` is called once and the original error is swallowed. `NativeOllamaAdapter` (the registered `ollama` provider) is correct (`if emitted_partial: raise`).
- **Impact:** Duplicated output and a second provider call after a transport close or failure; reachable only by constructing `LangChainOllamaAdapter` directly (it is not the registered default).
- **Reproduction:** `uv run --extra llm-langchain-ollama python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/adversarial_scripts/probe_ollama_stream.py`
- **Suggested repair scope:** Re-raise when any partial was emitted (as the native adapter does) and add a test.
- **Owner path:** intergrax/llm_adapters/providers/ollama_adapter.py

### F-D3-09 - LOW - The VBN classification of provider defaults named as R019's test case is not present

- **Requirements:** PDC-D3-R019
- **Verification:** CODE + TEST_RUN
- **Evidence:** The matrix test case is 'VBN register classification of each default' (provider `DEFAULT_MODEL` constants, `LLMProfile.lab`, rag default embedding provider). `LLMProfile.lab()` (llm_profile.py:108, Ollama `llama3.1:latest`) is consumed by intergrax/applications/contracts/environment_profile/bundles.py:243 and presets.py:138; provider `DEFAULT_MODEL` constants are floating aliases (`claude-3-5-sonnet-latest`, `mistral-large-latest`, `llama3.1:latest`, `gemini-2.5-flash`, `gpt-5-mini`). No EBH-6 register row, test or VBN document names them (grep over tests/qualification and docs/project/maintainers/qualification finds none), and the EBH-6 class gates skip `/providers/` trees (`_aj_class_gate_scan.py:221`). `tests/qualification/ebh_6` is green, which does not cover these defaults.
- **Impact:** R019 closure cannot be shown from the named register; the defaults may well be allowed (vendor-owned tree, lab preset) but are unclassified.
- **Reproduction:** `grep -rn "DEFAULT_MODEL" tests/qualification docs/project/maintainers/qualification`
- **Suggested repair scope:** Add a classification row (allowed vendor-owned default / lab-only) per default to the register and a test that fails when a new default is added unclassified.
- **Owner path:** tests/qualification/ebh_6/_findings_register.py, docs/project/maintainers/qualification/EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md

### F-D3-10 - LOW - Runtime attribution records the requested model id, never the model the provider reports serving

- **Requirements:** PDC-D3-R005
- **Verification:** CODE + PROBE
- **Evidence:** `adapter_response_from_openai_chat_completion` takes `model=` from the adapter and ignores `res.model` (the probe SDK reports 'm', the adapter model is 'gpt-4o-mini', the response model is 'gpt-4o-mini'); only `system_fingerprint` is kept (openai_completion_mapping.py:150-160). `LLMAdapterResponse` has no served-model field. The pin is a model string, and the shipped defaults are floating aliases (F-D3-09).
- **Impact:** Attribution (FRZ-TRC-04) cannot show which snapshot answered when an alias is pinned.
- **Reproduction:** `read intergrax/llm_adapters/_shared/openai_completion_mapping.py:140-186; adversarial_scripts/probe_timeout.py builds a ChatCompletion whose model differs from the adapter model`
- **Suggested repair scope:** Add an optional served-model field to the response / provider extensions and fill it from each provider's response where one is reported.
- **Owner path:** intergrax/llm_adapters/contracts/adapter_response.py, intergrax/llm_adapters/_shared/openai_completion_mapping.py

### F-D3-11 - LOW - `model_catalog.yaml` carries a duplicate record and the loader accepts duplicates and non-positive windows

- **Requirements:** PDC-D3-R002, PDC-D3-R023
- **Verification:** PROBE_CONFIRMED
- **Evidence:** `meta-llama/Llama-3.2-3B-Instruct` appears twice (model_catalog.yaml:256 and :284, different `provider_hints`); `ModelCatalog.from_mapping` appends every row and `lookup_exact` returns the first, `_merge` de-duplicates only overlays; `int(row['context_window_tokens'])` is not validated as positive. The R023 'single source' tests do not check catalog uniqueness.
- **Impact:** Latent: a later edit of one duplicate with a different window would silently be ignored (first wins).
- **Reproduction:** `python -c "import yaml,collections;d=yaml.safe_load(open('intergrax/llm_adapters/registry/model_catalog.yaml'));print([k for k,v in collections.Counter(m['model_id'] for m in d['models']).items() if v>1])"`
- **Suggested repair scope:** Reject duplicate ids and non-positive windows in `from_mapping`, add a catalog integrity test and merge the two rows.
- **Owner path:** intergrax/llm_adapters/registry/model_catalog.py, intergrax/llm_adapters/registry/model_catalog.yaml

### F-D3-12 - LOW - Tenant token quota is one process-wide cap that silently disables itself on a malformed value

- **Requirements:** PDC-D3-R018
- **Verification:** CODE
- **Evidence:** quota.py `_max_tokens_per_tenant` returns 0 (disabled) on `ValueError`, uses a single `INTERGRAX_LLM_TENANT_MAX_TOKENS` for every tenant and counts in-process metrics only (`get_llm_metrics_collector().tenant_total_tokens`).
- **Impact:** A typo in the cap turns the control off without a signal; the cap is not per-tenant configuration; counters reset per process.
- **Reproduction:** `read intergrax/llm_adapters/governance/quota.py:23-27`
- **Suggested repair scope:** Fail closed (raise) on a malformed non-empty value; document the per-process scope or move the counter behind the tenant store.
- **Owner path:** intergrax/llm_adapters/governance/quota.py

### F-D3-13 - LOW - Test strength gaps in the D3 gates (2 of 14 mutants survive; the AST gate accepts any call)

- **Requirements:** PDC-D3-R022, PDC-D3-R015, PDC-D3-R011
- **Verification:** MUTATION
- **Evidence:** Mutation sample (run_mutants.py, scratch worktree at the audited SHA, control green 176 passed / 5 skipped): 12 killed, 2 survived: M03 (failover fallback to `adapters[0]`, see F-D3-04) and M12 (stream registration removed, see F-D3-01). `test_every_provider_source_class_declares_what_it_implements` treats any `return <call>` as a declaration (`_declares_true`), so `return bool(self._x)` passes; it is the only gate for gemini, mistral and cohere because their SDKs are not installed in the default environment.
- **Impact:** Regressions of the two surviving mutants would not be caught by the D3 suite.
- **Reproduction:** `python docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/adversarial_scripts/run_mutants.py (adapt WT/OUT paths); results in logs/mutation/`
- **Suggested repair scope:** Add the two missing tests (F-D3-01, F-D3-04) and tighten `_declares_true` to constant True or delegation to a named flag.
- **Owner path:** tests/qualification/pdc_x/d3/

### F-D3-14 - INFO - Positive: F-49 and F-50 are resolved; the shared-slug defect class is closed; EBH-5 is green

- **Requirements:** PDC-D3-R003, PDC-D3-R016, PDC-D3-R024
- **Verification:** PROBE + MUTATION + TEST_RUN
- **Evidence:** `probe_multicat.py` bootstraps `register_default_integrations(preset='full')`: 199 slugs, 0 duplicates, 2 multi-category rows (`ollama`, `openai`), all 4 category resolves return the right integration (the only error is the expected missing OPENAI_API_KEY for managed retrieval). Mutants M10 (dispatch inverted) and M11 (`enabled` hard-wired) of openai/register.py are killed by p6_ecapfix. `tests/qualification/pdc_x/{p6_ecapfix,reg_p,p6_model,p6_impact}`: 110 passed. `tests/qualification/ebh_5`: 243 passed with the `integrations-duckdb` extra.
- **Impact:** None; supports PASS of R003, R016, R024.
- **Reproduction:** `see logs/`
- **Suggested repair scope:** none
- **Owner path:** intergrax/integrations/providers/embedding_provider/openai/register.py

### F-D3-15 - INFO - Environment notes and known reds

- **Requirements:** none
- **Verification:** TEST_RUN
- **Evidence:** The default environment lacks the optional extras: 5 D3 tests skip (`importorskip`), 4 ebh_5 tests fail without `duckdb` (243 pass with the extra), 2 unit modules error at collection without `anthropic`. The pdc_x/xd (80 passed), d3 and p6 runs exited normally: debt R-05 (pytest not exiting after the summary) was not reproduced. EBH-7, roadmap_replay_x, dist_scale_x and Docker were not re-reported (already classified).
- **Impact:** Interpretation of green/red depends on the installed extras.
- **Reproduction:** `see logs/`
- **Suggested repair scope:** none
- **Owner path:** n/a

### F-D3-16 - INFO - Documentation drift: the architecture and plan docs do not describe the declared-capability contract; the B-D3 commits carry an AI co-author trailer that predates the repo rule

- **Requirements:** PDC-D3-R006, PDC-D3-R022
- **Verification:** CODE
- **Evidence:** `LLMFeature`, `CancelCapability`, `supports_tool_streaming` and `LLMCapabilityMismatchError` appear only in ADR_PDC_03 (still headed 'Decision (proposed)'), not in docs/project/architecture/LLM_ADAPTERS.md or its plan. Commits 35b6772af and ff469148d carry a `Co-Authored-By: Claude` trailer; the no-attribution rule was added 2026-11-13 (68632a06f), after those commits (2026-11-12), so no violation is recorded.
- **Impact:** Maintainers learn the contract only from the ADR.
- **Reproduction:** `grep -rn "CancelCapability" docs`
- **Suggested repair scope:** Add a short section to LLM_ADAPTERS.md and its plan in the D3 doc lane.
- **Owner path:** docs/project/architecture/LLM_ADAPTERS.md, docs/project/maintainers/plans/LLM_ADAPTERS.md


## 5. Routed findings re-verified

| Lead | Result at the audited SHA |
| --- | --- |
| F-05 (six strict xfails, IA-D2/D3) | Resolved: `grep xfail tests/qualification/pdc_x` finds only a comment; the XD suite is 80 passed |
| F-37 (`required_llm_features` / `llm_profile_source` without a production caller) | **Confirmed and widened**: F-D3-04 (runtime resolver has no parameter; the XD-01 gate is satisfiable by the dormant caller) |
| F-42 (two `[integration.embedding_provider]` tests red since PDC-X) | Resolved: ebh_5 243 passed with the duckdb extra |
| F-49 (`openai` row resolves embedding through the managed-retrieval factory) | **Resolved**: dispatching factory in `openai/register.py`; probe over the full bootstrapped catalog shows 199 slugs, 0 duplicates, 2 multi-category rows, all resolving the requested category (F-D3-14) |
| F-50 (wrapper `enabled` / kwargs not pinned) | **Resolved**: mutants M10 and M11 are killed by `p6_ecapfix`; 110 tests passed |

## 6. Known reds not re-reported

EBH-7 identity gate, the five `roadmap_replay_x` tests, `dist_scale_x` and Docker scope (C10/C15 freeze). The `pdc_x` process-exit debt R-05 was **not reproduced** in the runs of this audit (xd, d3, p6 exit normally).

## 7. Overall verdict

**PARTIAL.** 12 rows PASS, 3 PASS_SCOPED, 9 PARTIAL, 0 FAIL. D3 should not be certified until F-D3-01 is repaired and the MEDIUM findings are repaired or dispositioned by the Operator. Suggested repair order: F-D3-01 (single `_execute_streaming` path plus test), then F-D3-02/-03 (same adapter files), F-D3-04 (resolver wiring), F-D3-05/-06 (test owners), F-D3-07 (error contract). A fresh auditor re-audits after repair (loop 1 of 2 for D3).

## 8. Limitations (what this audit did not cover)

- Only the 24 D3 rows and the routed leads; not IA-D1/D2/D4..D8, the global P8 audit, FRZ counters, CI contexts or `G-DOC` gates.
- No network and no live provider call (Path B); provider SDK behaviour was modeled with fakes, so SDK-specific streaming and error shapes were not observed. Gemini, Mistral, Cohere and Claude paths were read statically or exercised through fakes where the SDK is importable.
- R007 (structured outputs) and R008 (tool-call normalization) were verified through their existing suites and a code read, not by dedicated adversarial probes. R010 (usage and cost) likewise.
- Mutation is a 14-mutant sample, not exhaustive. The probes are reproducible scripts but not part of any CI gate.
- Environment: the default `.venv` has no optional extras; runs name the extras they used. `INTERGRAX_*` environment, Docker and a real Ollama were absent.
- The F-D3-06 attribution (the +12 tenant carriers) was not isolated to a domain; only the absence of D3 contribution was shown.
- Historical evidence of EBH-2E/EBH-5 was not trusted and not re-audited beyond the replays above.
- This report does not certify, promote or close anything; the certifier is a separate role.

## 9. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/D3/`: `iiac_audit_session_manifest.PDC-X.d3.json`, `audit_scope.json`, `iiac_audit_d3_findings.json`, `test_matrix.txt`, `EVIDENCE_MANIFEST.txt`, `adversarial_scripts/`, `logs/` (pytest logs, `probes/`, `mutation/`).
