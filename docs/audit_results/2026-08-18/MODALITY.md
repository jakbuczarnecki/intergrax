# MODALITY - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** MODALITY
- **Tier(s):** Tier-0 `intergrax/model_inference/` · Tier-0 `intergrax/tools/providers/` · Tier-1 `intergrax/runtime/modality/` · Tier-1 `intergrax/runtime/nexus/tools/` · Tier-3 `intergrax/applications/_shared/`
- **audited_sha:** `65e2c08f11be1db78f247380bafa4ac3d052a9f7`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 1 CRITICAL / 4 HIGH / 0 MEDIUM / 0 LOW
- **Operator decision:** all 5 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/MODALITY.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/MODALITY.md`
- **Scope in:**
  - three-plane Modality ownership (Plane A LLMAdapter · Plane B media→KnowledgeDocument/RAG · Plane C ToolRuntime + `model_inference`)
  - `ModalityProfile` intersection with `ToolAccessPolicy` before ToolRuntime
  - vision tool contracts and canonical vision tool service (`media_uri` handling)
  - `ModalityExecutionProfile` placement (IN_PROCESS · THREAD_POOL · CELERY)
  - remote vision adapters (Triton · Hugging Face Inference API) and stub fallback paths
  - deterministic CV policy (`require_deterministic_cv`)
  - historical W-ML.0–W-ML.8 and MODALITY-LC **Done** delivery facts (positive control)
- **Scope out:**
  - remediation implementation
  - Plane A LLMAdapter multimodal conformance re-qualification beyond boundary checks
  - Plane B parser/RAG ingest re-audit
  - online training / AutoML platform scope
  - silent runtime fixes in production source
  - universal backend/model support claims
- **Prior audit reference(s):** legacy layer-29 audit [`docs/audit_results/2026-06-18/MODALITY.md`](../../audit_results/2026-06-18/MODALITY.md) - historical only; Protocol v2 snapshot at pinned SHA supersedes for campaign register
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `e23721e809e4aa13cf40c650e9b97e3ab731c057`

## Executive summary

**Verdict: FAIL.** One CRITICAL and four HIGH accepted findings show unrestricted caller-controlled local filesystem media paths crossing remote inference trust boundaries, fail-open empty `ModalityProfile` semantics, paper deterministic-CV controls that do not bind effective adapter/model execution, silent Celery→local placement fallback, and production-named remote provider slugs that silently substitute stub detections. Positive controls: three-plane A/B/C ownership split remains sound; Plane A stays LLMAdapter-owned; Plane B stays media→KnowledgeDocument/RAG; Plane C agent invocation crosses ToolRuntime; `ToolAccessPolicy` applies before ModalityProfile intersection; Modality is not a second generic tool runtime; execution strategy is separated behind `ModalityInferenceExecutor`; conservative plane-specific maturity (A4/I2/P1/E2 aggregate) is honestly documented; no dedicated public production proof is claimed. Residual defects are Protocol-v2 security/governance/execution-integrity gaps distinct from historical W-ML/MODALITY-LC delivery completion - remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** - 1 CRITICAL / 4 HIGH / 0 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-MODALITY-01

- **Severity:** CRITICAL
- **Category:** SECURITY / LOCAL FILE EXFILTRATION
- **Status at publication:** ACCEPTED
- **Remediation block:** MODALITY-MEDIA-TRUST-BOUNDARY
- **Claim falsified:** Agent-facing modality tools consume authorized media references with tenant/scope ownership, allowed source types, and explicit remote egress authorization - not unrestricted local filesystem pointers.
- **Observation:** Vision tool contracts expose caller-controlled `media_uri: str`. The canonical vision tool service passes this `media_uri` into inference after only profile artifact/media-size checks. `_resolve_media_path()` interprets both `file://...` and arbitrary ordinary path strings as local filesystem `Path`s. `TritonVisionServingAdapter` resolves `media_uri` to a local `Path`, reads `Path(...).read_bytes()`, base64-encodes bytes, and sends them to the configured remote Triton endpoint. `HuggingFaceInferenceVisionAdapter` resolves `media_uri` to local `Path`, reads bytes, and sends them to the external Hugging Face Inference endpoint. There is no canonical media-reference boundary proving tenant/scope ownership, allowed media source type, permitted filesystem root, approved blob/object reference, egress authorization, or provenance.
- **Location:**
  - `intergrax/tools/providers/vision/contracts.py` - `media_uri: str` on vision tool contracts
  - `intergrax/tools/providers/vision/service.py` - canonical vision tool service; `_resolve_media_path()`
  - `intergrax/tools/providers/vision/inference_support.py` - media path resolution helpers
  - `intergrax/model_inference/adapters/triton_vision.py` - local `Path.read_bytes()` → remote Triton
  - `intergrax/model_inference/adapters/huggingface_inference_vision.py` - local bytes → HF endpoint
- **Reproduction:** Static inspection at `audited_sha`: trace `vision.detect` / vision tool invocation from contract `media_uri` through `_resolve_media_path()` into adapter `read_bytes()` and remote HTTP dispatch; confirm no tenant/scope or sandbox-root authorization gate precedes byte read.
- **Impact:** A caller authorized for a vision tool can potentially cause arbitrary host-readable local bytes to cross a remote inference trust boundary - exploitable under multi-tenant or partially trusted agent/tool input conditions.
- **Confidence:** CONFIRMED

### AUDIT-20260818-MODALITY-02

- **Severity:** HIGH
- **Category:** GOVERNANCE / CAPABILITY AUTHORIZATION
- **Status at publication:** ACCEPTED
- **Remediation block:** MODALITY-AUTHORITY-INTEGRITY
- **Claim falsified:** `ModalityProfile` only narrows an already-authorized capability set; empty planes and empty explicit tool IDs do not silently mean wildcard authority.
- **Observation:** `ModalityProfile` defaults: `allowed_planes = empty set`, `allowed_tool_ids = empty tuple`. `filter_tool_ids_by_modality_profile()` only restricts by explicit tools when `allowed_tool_ids` is non-empty, and only restricts by plane prefixes when `allowed_planes` yields prefixes. Therefore empty planes + empty explicit tool IDs means all input tool IDs survive - an apparently restrictive empty profile is fail-open. Additionally `GENERATIVE_LLM` maps to `websearch.*` even though Plane A is defined architecturally as LLMAdapter-based and not ToolRuntime-based.
- **Location:**
  - `intergrax/runtime/modality/modality_profile.py` - defaults; `filter_tool_ids_by_modality_profile()`; plane prefix mapping including `GENERATIVE_LLM` → `websearch.*`
  - `intergrax/runtime/nexus/tools/tool_access_policy.py` - `apply_modality_profile()` intersection order (positive: ToolAccessPolicy first)
- **Reproduction:** Instantiate default `ModalityProfile()`; pass a broad tool ID list through `filter_tool_ids_by_modality_profile()`; observe unchanged output. Inspect plane mapping table for `GENERATIVE_LLM`.
- **Impact:** Hosts believing an empty or minimal profile restricts modality capability may unintentionally grant full Plane C (and unrelated websearch) tool surface; undermines monotonic authority intersection with TOOLS findings.
- **Confidence:** CONFIRMED

### AUDIT-20260818-MODALITY-03

- **Severity:** HIGH
- **Category:** POLICY / FALSE SAFETY CONTROL
- **Status at publication:** ACCEPTED
- **Remediation block:** MODALITY-AUTHORITY-INTEGRITY
- **Claim falsified:** When `require_deterministic_cv=True`, effective adapter/model/artifact execution is deterministic and caller-selected adapter/artifact cannot override host determinism requirements.
- **Observation:** `ModalityProfile` exposes `require_deterministic_cv`. The current filter implements this mainly by tool ID: vision tools other than `vision.detect` may be filtered; `vision.detect` remains allowed; an explicitly listed vision tool can bypass that heuristic. But `vision.detect` accepts caller-controlled `artifact_id` and `adapter_slug` and resolves the selected registry artifact and adapter at runtime. `assert_artifact_allowed()` only constrains artifacts when `vision_model_ids` is non-empty. The production deterministic profile does not itself pin a vision model list. Therefore `require_deterministic_cv=True` does not prove that the selected model/adapter execution is deterministic.
- **Location:**
  - `intergrax/runtime/modality/modality_profile.py` - `require_deterministic_cv`; tool-ID heuristic filter
  - `intergrax/tools/providers/vision/service.py` - `vision.detect`; caller `artifact_id`, `adapter_slug`
  - `intergrax/tools/providers/vision/inference_support.py` - `assert_artifact_allowed()`
- **Reproduction:** Configure profile with `require_deterministic_cv=True` and empty `vision_model_ids`; invoke `vision.detect` with non-deterministic adapter slug; observe pass-through without determinism certification failure.
- **Impact:** Safety/policy hosts may believe deterministic CV is enforced when caller can select non-deterministic adapter/model tuples at runtime.
- **Confidence:** CONFIRMED

### AUDIT-20260818-MODALITY-04

- **Severity:** HIGH
- **Category:** EXECUTION PLACEMENT / RESOURCE SAFETY
- **Status at publication:** ACCEPTED
- **Remediation block:** MODALITY-EXECUTION-INTEGRITY
- **Claim falsified:** Explicit CELERY/offload placement choice is policy-truthful; mandatory offload fails closed when broker/task unavailable - it does not silently become local heavy inference.
- **Observation:** `ModalityExecutionProfile` provides explicit IN_PROCESS, THREAD_POOL, and CELERY. When CELERY is selected, `CeleryModalityInferenceExecutor` silently falls back to a ThreadPool executor when broker/app unavailable, task unavailable, dispatch fails, result fails/times out, or result contains an error. Exceptions are swallowed and local fallback executes.
- **Location:**
  - `intergrax/model_inference/execution/profile.py` - `ModalityExecutionProfile`; execution modes
  - `intergrax/model_inference/execution/factory.py` - executor selection
  - `intergrax/model_inference/execution/celery_executor.py` - `CeleryModalityInferenceExecutor` fallback path
- **Reproduction:** Select CELERY mode with broker URL absent or dispatch failure injected; observe ThreadPool local execution without explicit policy failure.
- **Impact:** Hosts requiring heavyweight adapters never in application process (isolation/resource policy) cannot rely on CELERY mode; silent local execution undermines resource safety guarantees.
- **Confidence:** CONFIRMED

### AUDIT-20260818-MODALITY-05

- **Severity:** HIGH
- **Category:** INFERENCE INTEGRITY / SILENT STUB FALLBACK
- **Status at publication:** ACCEPTED
- **Remediation block:** MODALITY-EXECUTION-INTEGRITY
- **Claim falsified:** Production-capable provider slugs represent the provider they name; missing credentials/config fail closed; legitimate zero detection is not provider failure; no synthetic stub substitution on named remote slugs.
- **Observation:** `TritonVisionServingAdapter` falls back to `StubVisionInferenceAdapter` when `INTERGRAX_TRITON_URL`/`base_url` is absent or parsed remote response contains no detections. `HuggingFaceInferenceVisionAdapter` similarly falls back when API key is absent or parsed response contains no detections. Thus a remote provider slug can return synthetic stub detections instead of reporting provider configuration/unavailability/legitimate zero detections. `remote_serving.py` additionally contains legacy `MlInferenceHostAdapter(StubModelInferenceAdapter)` as a compatibility placeholder.
- **Location:**
  - `intergrax/model_inference/adapters/triton_vision.py` - stub fallback on missing URL / empty detections
  - `intergrax/model_inference/adapters/huggingface_inference_vision.py` - stub fallback on missing key / empty detections
  - `intergrax/model_inference/adapters/remote_serving.py` - `MlInferenceHostAdapter(StubModelInferenceAdapter)` compatibility façade
- **Reproduction:** Invoke `vision_serving` or `huggingface_inference` slug without env credentials; observe stub detections returned under production-named slug. Invoke with valid credentials and empty detection response; observe stub substitution conflated with legitimate zero detections.
- **Impact:** Operators and agents cannot distinguish configured remote inference from synthetic stub output; safety-critical CV paths may accept fabricated detections; undermines provider truthfulness and auditability.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Three-plane A/B/C ownership split | NOT falsified - sound |
| Plane A remains LLMAdapter-owned | NOT falsified |
| Plane B remains media→KnowledgeDocument/RAG boundary | NOT falsified |
| Plane C agent invocation crosses ToolRuntime | NOT falsified |
| ToolAccessPolicy applied before ModalityProfile intersection | NOT falsified |
| Modality is not a second generic tool runtime | NOT falsified |
| Execution strategy separated behind ModalityInferenceExecutor | NOT falsified |
| ModalityExecutionProfile uses strict typed configuration | NOT falsified |
| Maturity remains conservative and plane-specific | NOT falsified |
| Domain aggregate A4/I2/P1/E2 does not imply production qualification | NOT falsified |
| No universal backend/model support claim | NOT falsified |
| No dedicated public production proof claimed | NOT falsified |

## Provider / backend abstraction

| concern | canonical abstraction | provider boundary | composition owner | observed provider(s) | classification | evidence/finding |
|---------|-----------------------|-------------------|-------------------|----------------------|----------------|------------------|
| Vision inference | `VisionInferenceAdapter` / registry | Adapter modules (`triton_vision`, `huggingface_inference_vision`, …) | `ModalityInferenceExecutor` factory + host wiring | Triton HTTP, HF Inference API, OpenCV, stub | **VENDOR_LEAK** (media bytes from arbitrary local path) + **PAPER_ABSTRACTION** (remote slug → stub) | MOD-01, MOD-05 |
| Classical ML inference | `ModelInferenceAdapter` | `remote_serving.py` adapters | ToolRuntime → `ml.*` tools | Stub via `MlInferenceHostAdapter` placeholder | **PAPER_ABSTRACTION** | MOD-05 |
| Execution placement | `ModalityExecutionProfile` / `ModalityInferenceExecutor` | Celery broker/task inside executor | Host `RuntimeConfig` / env | Celery with silent ThreadPool fallback | **IMPLEMENTATION DEFECT** (placement truthfulness) | MOD-04 |

## Historical delivery vs Protocol-v2 residual defects

Historical **W-ML.0–W-ML.8**, **MODALITY-LC**, **MOD-SPEECH-ARCH**, and **MOD-MAINT-01…05** **Done** delivery facts remain valid - typed contracts, `model_inference`, tools, profiles, modality metrics, and harness E2E paths were delivered as claimed. The five accepted Protocol-v2 findings document **residual security, governance, placement, and provider-truthfulness gaps** discovered by adversarial falsification at `audited_sha` - they do not reopen or negate historical W-ML/MODALITY-LC closeout rows.

## Root-cause remediation grouping

### MODALITY-MEDIA-TRUST-BOUNDARY - scoped authorized media identity and remote egress boundary

**Findings:** `AUDIT-20260818-MODALITY-01`

Replace arbitrary caller filesystem path semantics with scoped authorized media identity and explicit remote egress boundary. Reuse canonical resource/evidence authority if available - do not create a Modality-specific duplicate authorization subsystem.

### MODALITY-AUTHORITY-INTEGRITY - fail-closed ModalityProfile and deterministic-CV binding

**Findings:** `AUDIT-20260818-MODALITY-02`, `AUDIT-20260818-MODALITY-03`

ModalityProfile becomes fail-closed; deterministic-CV policy binds the actual effective adapter/model/artifact tuple. Cross-link TOOLS authority findings rather than adding a second permission system.

### MODALITY-EXECUTION-INTEGRITY - policy-truthful placement and provider slugs

**Findings:** `AUDIT-20260818-MODALITY-04`, `AUDIT-20260818-MODALITY-05`

Execution placement distinguishes mandatory offload from fallback-permitted modes; named remote provider slugs cannot silently degrade to local/stub inference.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `65e2c08f11be1db78f247380bafa4ac3d052a9f7`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical W-ML / MODALITY-LC plan **Done** rows remain valid delivery facts - not rewritten.

## Open questions / blocked items

- Finding 01: whether canonical media-reference authority lives in TOOLS resource boundary, evidence store, or a shared platform port - deferred to remediation design (reuse before invent).
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 5 (`AUDIT-20260818-MODALITY-01` … `AUDIT-20260818-MODALITY-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
