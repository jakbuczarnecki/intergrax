# PROVIDER_BACKEND_ABSTRACTION — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** PROVIDER_BACKEND_ABSTRACTION
- **Tier(s):** cross-domain Tier-0 / Tier-1 / Tier-2 / Tier-3 provider/backend boundary audit
- **audited_sha:** `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Status:** COMPLETE
- **Auditor:** OpenAI ChatGPT / GPT-5.6 Sol — independent auditor
- **Verdict:** FAIL
- **Architecture doc(s):**
  - `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`
  - `docs/project/architecture/INTEGRATIONS.md`
  - `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md`
  - `docs/project/maintainers/plans/INTEGRATIONS.md`
  - `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`
- **Scope in:**
  - provider/backend abstraction discipline
  - actual dependency/call-path verification
  - concrete provider type/import/config leakage
  - paper abstractions
  - composition-root exceptions
  - canonical Integration reuse
  - provider governance/test proof
  - selected material persistence boundaries
- **Scope out:**
  - remediation implementation
  - full security audit except security impact of bypassed provider boundaries
  - universal production qualification of every provider
  - properly provider-local SDK implementation internals
  - re-audit of previously published layers
- **Prior audit reference(s):** legacy provider/integrations material is historical only; no prior canonical Protocol v2.2 PROVIDER_BACKEND_ABSTRACTION snapshot
- **Exact audit-start time:** not captured; preserve date-level precision rather than fabricate one
- **post_sync_sha:** `—`

## Executive summary

**Verdict: FAIL.** Intergrax has multiple genuine provider-neutral persistence and behavior ports, but five accepted findings (2 HIGH, 3 MEDIUM) show paper abstractions, canonical observability bypass, provider-specific configuration leakage into generic contracts, governance that blesses a known vendor bypass, and a missing experiment persistence port on a reusable workflow. Production-critical long-running checkpoint semantics depend on `SQLiteTaskCheckpointStore` in generic Nexus/coordinator paths despite an existing `TaskCheckpointPersistence` port. RAG parser telemetry reaches Sentry/Langfuse without the canonical sanitized observability export boundary. Generic guardrail profiles carry Bedrock-specific fields. Vendor-import governance explicitly allows the parser-trace exporter file containing direct `sentry_sdk` usage. `ExperimentSession` and debug HTTP consumers type against `SQLiteExperimentStore` with no stable experiment persistence port. FAIL means provider/backend discipline is not structurally universal — not that every SQLite implementation or provider-owned SDK use is invalid.

## Verdict

**FAIL**

## Findings

### AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-01

**Long-running checkpoint port exists but core runtime depends on SQLite concrete store**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Provider classification:** PAPER_ABSTRACTION
- **Related classification note:** VENDOR_LEAK
- **Confidence:** CONFIRMED
- **Claim falsified:** Long-running Nexus checkpoint persistence is consumed through the provider-neutral `TaskCheckpointPersistence`/`TaskCheckpointReader` boundary, so replacing SQLite does not require editing generic runtime orchestration.
- **Observation:** `TaskCheckpointPersistence` exists. `NexusLoop` imports and publicly types `checkpoint_store` as `SQLiteTaskCheckpointStore`. `LongRunningCoordinator` also consumes `SQLiteTaskCheckpointStore` directly. Provider-neutral checkpoint construction is exposed as `SQLiteTaskCheckpointStore.build_checkpoint()`. Therefore the real runtime call path depends on the concrete provider despite the existence of the port.
- **Location:**
  - `intergrax/runtime/long_running/persistence_contract.py:L18-L48` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/runtime/nexus/nexus_loop.py:L58,L134-L135,L189` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/runtime/long_running/coordinator.py:L23,L54-L55,L102,L120` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/runtime/long_running/store.py:L59,L338-L357` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Reproduction:**
  1. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/long_running/persistence_contract.py` — `TaskCheckpointPersistence` / `TaskCheckpointReader` port definitions (L18-L48).
  2. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/nexus/nexus_loop.py` — import of `SQLiteTaskCheckpointStore` (L58); constructor parameter typed as concrete store (L134-L135).
  3. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/long_running/coordinator.py` — `store: SQLiteTaskCheckpointStore` parameters (L54-L55, L102); `SQLiteTaskCheckpointStore.build_checkpoint(...)` call (L120).
  4. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/long_running/store.py` — `SQLiteTaskCheckpointStore` implementation (L59); `build_checkpoint` classmethod on SQLite store (L338-L357).
- **Impact:** Production-critical pause/resume/checkpoint semantics are coupled to SQLite; backend substitution requires changes in generic runtime and provider-neutral logic is partly owned by the SQLite implementation.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-02

**RAG parser observability bypasses the canonical Observability provider boundary**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Provider classification:** VENDOR_LEAK
- **Related platform-reuse classification:** BYPASSED PLATFORM MECHANISM
- **Confidence:** CONFIRMED
- **Claim falsified:** External observability vendor delivery flows through the canonical, policy-sanitized `ObservabilityVendorIntegrationContract`/provider boundary; RAG/runtime consumers do not call Sentry/Langfuse directly.
- **Observation:** Canonical `ObservabilityVendorIntegrationContract` validates/maps a policy-sanitized `ObservabilityExportEnvelope` before vendor delivery. `parser_trace_exporter` branches on `sentry` / `langfuse`. It imports `sentry_sdk` directly and calls Sentry. It performs a direct `httpx` POST to Langfuse ingestion. `ParserPipeline` calls `export_parser_trace()`. Persisted parser traces are also replayed into the same exporter from the Nexus/journal observability path. Parser trace contains `source` and raw parser `error: str(exc)`. This path therefore bypasses the canonical sanitization/provider boundary.
- **Location:**
  - `intergrax/runtime/integrations/observability.py:L198-L267` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/rag/document_loaders/observability/parser_trace_exporter.py:L24-L66,L69-L105` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/rag/document_loaders/pipeline/parser_pipeline.py:L67-L69,L96-L98` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/runtime/nexus/tracing/parser_trace_flush.py:L18-L29` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/runtime/observability/export_bridge.py:L20,L70` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Reproduction:**
  1. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/integrations/observability.py` — canonical `ObservabilityVendorIntegrationContract` and envelope mapping (L198-L267).
  2. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/rag/document_loaders/observability/parser_trace_exporter.py` — `export_parser_trace` branches to `_export_sentry` / `_export_langfuse` (L63-L66); direct `sentry_sdk` (L69-L77); direct `httpx.post` to Langfuse (L82-L105).
  3. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/rag/document_loaders/pipeline/parser_pipeline.py` — trace includes `"error": str(exc)` (L67-L69); calls `export_parser_trace` (L96-L98).
  4. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/nexus/tracing/parser_trace_flush.py` — replay path calls same exporter (L18-L29).
  5. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/runtime/observability/export_bridge.py` — journal replay invokes `export_parser_traces_from_events` (L20, L70).
- **Impact:** Canonical provider abstraction and observability safety/export guarantees are not structurally universal on the parser-trace path. The required sanitization/provider boundary is bypassed and `source`/`error` metadata can reach an external vendor without that canonical boundary. No secret exfiltration was proven.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-03

**Generic guardrail configuration contains AWS Bedrock-specific configuration**

- **Severity:** MEDIUM
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Provider classification:** VENDOR_LEAK
- **Confidence:** CONFIRMED
- **Claim falsified:** Generic platform/host guardrail contracts remain provider-neutral and provider-specific configuration is owned by provider modules/composition.
- **Observation:** Generic `GuardrailProfile` contains `bedrock_guardrail_policy_id`. `GuardrailBackendOptions` is explicitly vendor-specific and also contains it. Tier-3 guardrail wiring copies it from the generic environment profile. Bedrock provider consumes it. `LlmGuardrailBackend` behavior abstraction itself DOES exist and is valid; the defect is configuration ownership leakage, not absence of the behavior port.
- **Location:**
  - `intergrax/contracts/host_profile_slices.py:L43-L54` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/applications/_shared/guardrail_runtime_bridge.py:L24-L29` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/integrations/contracts/llm_guardrail.py:L20-L27,L66` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/integrations/providers/llm_guardrail/bundles/bedrock_guardrails.py:L18-L20,L77-L78` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Reproduction:**
  1. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/contracts/host_profile_slices.py` — `GuardrailProfile.bedrock_guardrail_policy_id` on generic profile (L43-L54).
  2. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/applications/_shared/guardrail_runtime_bridge.py` — copies Bedrock field into `GuardrailBackendOptions` (L24-L29).
  3. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/integrations/contracts/llm_guardrail.py` — vendor-specific `GuardrailBackendOptions` field (L20-L27); `LlmGuardrailBackend` behavior port (L66).
  4. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/integrations/providers/llm_guardrail/bundles/bedrock_guardrails.py` — Bedrock backend consumes policy id (L18-L20, L77-L78).
- **Impact:** Generic shared configuration can accumulate provider-specific fields and forces platform contracts to evolve when providers add unique settings.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-04

**Vendor-import governance explicitly permits the confirmed RAG vendor bypass**

- **Severity:** MEDIUM
- **Category:** TEST GAP
- **Status at publication:** ACCEPTED
- **Confidence:** CONFIRMED
- **Claim falsified:** Mechanical vendor-boundary governance reliably fails when high-level Intergrax code imports/uses concrete vendors outside approved provider boundaries.
- **Observation:** `check_integration_vendor_imports.py` has a finite vendor list. `RAG_ALLOWED_SUFFIXES` explicitly allows `/parser_trace_exporter.py`. That exact file contains the confirmed direct `sentry_sdk` path. Therefore the gate can remain green while the canonical provider boundary is bypassed. `test_vendor_import_governance.py` is marked `no_ci`. The maintenance script is invoked on the full/nightly governance workflow; it is not absent from all CI.
- **Location:**
  - `scripts/maintenance/check_integration_vendor_imports.py:L16-L31,L44-L48` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `tests/unit/integrations/test_vendor_import_governance.py:L12,L15-L17` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `.github/workflows/unit-tests.yml:L166,L196-L207` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Reproduction:**
  1. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:scripts/maintenance/check_integration_vendor_imports.py` — finite `VENDOR_MODULES` including `sentry_sdk` (L16-L31); `RAG_ALLOWED_SUFFIXES` includes `/parser_trace_exporter.py` (L44-L48).
  2. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/rag/document_loaders/observability/parser_trace_exporter.py` — direct `sentry_sdk` import in allowed file (L69-L77).
  3. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:tests/unit/integrations/test_vendor_import_governance.py` — `pytest.mark.no_ci` (L12).
  4. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:.github/workflows/unit-tests.yml` — governance workflow invokes `check_integration_vendor_imports.py` (L207); PR smoke excludes `no_ci` (L166).
- **Impact:** Existing governance can certify a known direct vendor path as acceptable, weakening provider-independence proof and allowing regressions of the same class.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-05

**Experimentation/DX workflow has no provider-neutral experiment persistence port**

- **Severity:** MEDIUM
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Provider classification:** MISSING_ABSTRACTION
- **Related classification note:** VENDOR_LEAK
- **Confidence:** CONFIRMED
- **Claim falsified:** The reusable `ExperimentSession` workflow depends on a provider-neutral experiment persistence contract while SQLite remains only a lab composition choice.
- **Observation:** `intergrax/experiments/store.py` exposes `SQLiteExperimentStore` directly; no `ExperimentPersistence`/`ExperimentStore` port was found in the audited slice. `ExperimentSession` imports/constructs `create_sqlite_experiment_store` and `create_sqlite_trace_store`. Its experiment store field/property are typed as `SQLiteExperimentStore`. Debug experiment factory likewise returns `SQLiteExperimentStore`. Nearby runtime-event/checkpoint/trace debug dependencies already use `RuntimeEventPersistence`, `TaskCheckpointReader` and `RunTraceReader` abstractions. Architecture explicitly scopes Experimentation/DX as laboratory workflow, so the finding is MEDIUM rather than HIGH.
- **Location:**
  - `intergrax/experiments/store.py:L27,L33-L41` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/experiments/workflow.py:L20-L21,L28,L109-L112,L119` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
  - `intergrax/debug/router.py:L76,L110-L115,L393-L440` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`
- **Reproduction:**
  1. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/experiments/store.py` — public `SQLiteExperimentStore` only (L27, L33-L41).
  2. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/experiments/workflow.py` — imports and constructs SQLite stores; field typed as `SQLiteExperimentStore` (L20-L21, L28, L109-L112, L119).
  3. `git show 7570e9b4508554a42bdf5cce2c987c56c6f2b80e:intergrax/debug/router.py` — factory and HTTP dependencies typed as `SQLiteExperimentStore` (L76, L110-L115, L393-L440); contrast with `RunTraceReader` abstraction (L83, L89-L99).
- **Impact:** The reusable experiment workflow and debug surface are tied to SQLite; backend substitution or reuse outside the current lab persistence model requires editing workflow code rather than composition.
- **Confidence:** CONFIRMED

## Provider/backend abstraction classification matrix

| Concern | Canonical abstraction | Observed backend/provider | Classification | Notes |
|---------|----------------------|---------------------------|----------------|-------|
| Long-running checkpoints | `TaskCheckpointPersistence` | SQLite | PAPER_ABSTRACTION | port exists but Nexus/coordinator consume concrete store |
| Human decisions | `HumanDecisionPersistence` | SQLite / memory | ABSTRACTION_PRESERVED | consumer uses port |
| Runtime events | `RuntimeEventPersistence` | SQLite / other implementations | ABSTRACTION_PRESERVED | consumer uses port |
| Task memory | `TaskMemoryPersistence` | SQLite / null | ABSTRACTION_PRESERVED | consumer uses port |
| Agent checkpoints | `AgentCheckpointStore` | SQLite / memory | ABSTRACTION_PRESERVED | concrete selection occurs in wiring |
| Idempotency | `IdempotencyStore` | SQLite / memory / Redis | ABSTRACTION_PRESERVED | high-level consumer uses port |
| Representative vector-store provider | vector-store/RAG provider contract | Qdrant | PROVIDER_LOCAL | vendor code contained in provider-owned package in inspected path |
| Parser observability export | `ObservabilityVendorIntegrationContract` / canonical export boundary | Sentry / Langfuse | VENDOR_LEAK | direct vendor branch bypasses canonical mechanism |
| Adaptive profile persistence | `ProfileVersionStore` etc. | SQLite / memory | ABSTRACTION_PRESERVED | executor depends on ports |
| LLM guardrail | `LlmGuardrailBackend` | Bedrock / other backends | VENDOR_LEAK | behavior port preserved; generic config leaks Bedrock field |
| Experiment registry | no stable experiment persistence port found | SQLite | MISSING_ABSTRACTION | reusable workflow consumes concrete provider |

## Positive / non-falsified evidence

This audit did **not** conclude that "SQLite everywhere is bad" or that provider-owned SDK use is automatically a defect.

1. **`HumanDecisionPersistence`** is a genuine consumer-facing abstraction (`intergrax/runtime/human/persistence_contract.py:L14` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`).
2. **`RuntimeEventPersistence`** is a genuine abstraction (`intergrax/runtime/events/persistence_contract.py:L22` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`).
3. **`TaskMemoryPersistence`** is a genuine abstraction (`intergrax/runtime/task_memory/persistence_contract.py:L14` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`).
4. **`AgentCheckpointStore`** is consumed as an abstraction; concrete SQLite selection occurs in wiring (`intergrax/agents/persistence/checkpoint_store.py:L16` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`).
5. **`IdempotencyStore`** is consumed as an abstraction by reliability wiring (`intergrax/applications/_shared/reliability_wiring.py:L18,L33-L45` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`).
6. **Adaptive core `AdaptationExecutor`** depends on `ProfileVersionStore` / `ProfileActivePointerStore`, not SQLite (`intergrax/runtime/adaptive/adaptation_executor.py:L21-L22,L76-L77` @ `7570e9b4508554a42bdf5cce2c987c56c6f2b80e`).
7. **Provider-owned SDK/dialect use** below a valid port or inside provider-owned packages is not itself a finding when consumers depend on the port.

**FAIL qualification:** verdict means provider/backend discipline is not structurally universal — **not** that all abstractions are absent or every SQLite backend is invalid.

## Falsification log

Targets examined but **not** promoted to findings:

1. **SQLite implementation existing below a valid port** — not automatically a defect; consumers using the port (human decisions, runtime events, task memory, agent checkpoints, idempotency) were inspected and not promoted.
2. **Adaptive persistence** — core `AdaptationExecutor` depends on `ProfileVersionStore` / `ProfileActivePointerStore` ports; not promoted.
3. **Agent checkpoint / idempotency SQLite implementations** — concrete stores exist but high-level consumers use ports; not promoted.
4. **Parser-trace secret exfiltration** — bypass of canonical observability boundary confirmed; no proven secret leak claim for finding 02.
5. **Multiple production providers per category** — no claim that every provider must have multiple production implementations to satisfy abstraction discipline.

## Prior-audit comparison

Legacy provider/integrations audit material remains **historical evidence only**. No prior canonical Protocol v2.2 `PROVIDER_BACKEND_ABSTRACTION` immutable snapshot existed before this layer. Prior INTEGRATIONS-LC and observability contract migration work documents provider package patterns but does not falsify the parser-trace bypass or paper-abstraction observations at `audited_sha`.

## Open questions / blocked items

- Whether long-running checkpoint `build_checkpoint` domain logic should live in runtime-neutral code or a dedicated factory — planning only; not resolved by this audit.
- Whether experiment persistence should share relational-store provider wiring patterns used by other lab stores — deferred to **PBA-FIX-D** planning.
- No operator-disputed findings; no blocked evidence collection.

## Operator acceptance

- **Date:** 2026-08-18
- **Accepted findings:** all 5 (`AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-01` … `AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
