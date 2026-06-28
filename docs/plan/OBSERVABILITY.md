# Observability — Implementation Plan

**Architecture (1:1):** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

**Cross-plan — Agent layer (ACP):** Dual observability planes (architecture §31) — `AgentRunTrace` on `AgentRunResult` (Plane B) and `ApplicationRunSummary` on Task completion (Plane A). Delivered in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 3** (`ACP-OBS-1`, `ACP-OBS-2`) and **Wave 7** redaction (`ACP-PROD-8`). Trace spine changes MUST keep step records compatible with `AgentStepRecord` tool/RAG/LLM fields.

**Cross-plan — Event catalog (OBS-EVOL-9 · P1-ARCH-02):** Layered spine + `event_kind` (architecture §4.4 · ADR-OBS-003). Developers extend via `emit_domain_signal`, not new `RuntimeEventType`. Pre-release spine consolidation before publication.

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). OBSERVABILITY owns token savings attribution, optimization receipts visibility, typed diagnostic payloads, metrics, and regression-gate reporting through the Harness Observability Spine.

**Last updated:** 2026-06-28 — **INTEGRATIONS-2A** observability_backend category alignment documented.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (OBSERVABILITY plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites/) satellites on demand. **On demand (one max):** [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](plan/satellites/OBSERVABILITY_eval_control_plane.md) (active OECP register), [`plan/satellites/OBSERVABILITY_audit_history.md`](plan/satellites/OBSERVABILITY_audit_history.md) (closed phases). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + rows `TOKEN-OBS-1` / `TOKEN-OBS-2`; use HOS/domain-signal model, do not create private telemetry channel.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/OBSERVABILITY.md`](../guides/audit_slices/OBSERVABILITY.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Architecture documentation (P2)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-07** | Clarify observability event spine and event ownership | **Done** (2026-06-20) |

Architecture: [`OBSERVABILITY.md`](../architecture/OBSERVABILITY.md#observability-event-spine).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](plan/satellites/OBSERVABILITY_eval_control_plane.md) | **OECP** — eval control plane implementation register (active) |
| [`plan/satellites/OBSERVABILITY_audit_history.md`](plan/satellites/OBSERVABILITY_audit_history.md) | audit history (closed phases) |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-OBS — Token optimization telemetry and regression gates (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)  
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)  
**Priority:** P1 after TOKEN-UER-1; TOKEN-OBS-1 may ship before CE/MEM integrations, TOKEN-OBS-2 after first optimized source exists.  
**Delivery rule:** one `TOKEN-OBS-*` row per PR; emit through HOS or approved domain-signal path only.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-OBS-1** | Code | P1 | Planned | `intergrax/runtime/token_optimization/telemetry.py` with typed optimization summary payload, receipt visibility, and counters/spans for saved tokens, failures, fallbacks, source type, strategy, output profile, model/provider | No private telemetry bus; attribution includes run/step/source/model/provider/strategy/profile; redaction rules followed; compatible with unified run journal; `uv run pytest tests/unit/runtime/observability/ -q`; `uv run pytest tests/unit/runtime/token_optimization/ -q` |
| **TOKEN-OBS-2** | Test/Gate | P1 | Planned | Token-vs-quality regression benchmark runner and scripts `check_compression_receipts.py`, `check_token_regression_benchmarks.py` | CI can fail on uncontrolled token growth, missing receipts, protected-region failures, or quality regression; benchmark fixtures cover output policy, tool catalog, and context pack cases; `uv run python scripts/check_compression_receipts.py`; `uv run python scripts/check_token_regression_benchmarks.py` |

**Explicit exclusions:** no new `RuntimeEventType` unless ADR/OBS review requires it; prefer typed payload/domain-signal style consistent with OBS event ownership; no raw prompt/completion persistence in production traces.

---

## Phase OBS-EXPORT — External observability export boundary (In progress)

**Purpose:** Define the platform-level export boundary for external observability sinks such as JSONL/file, OTLP, Elasticsearch, Langfuse, Arize/Phoenix.

**OBS-EXPORT-1 status (2026-06-28):**

- **OBS-EXPORT-1-EXPORT-BOUNDARY** — **Done**
- Normalized **`ObservabilityExportEnvelope`** added (`observability_export_envelope.v1`)
- **`ObservabilityExporter`** async protocol + **`NoOpObservabilityExporter`** + **`InMemoryObservabilityExporter`** / **`TestObservabilityExporter`**
- Typed source models (`RuntimeEventExportSource`, `GatewayCallExportSource`) and safe mapping helpers from `RuntimeEvent`, `ToolCallRecord`, `RagCallRecord`, and `JournalRef`
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix, Elasticsearch, OTLP deferred to OBS-EXPORT-4/5)
- Lifecycle wiring and failure isolation **deferred to OBS-EXPORT-2** (superseded below)

**OBS-EXPORT-2 status (2026-06-28):**

- **OBS-EXPORT-2-EXPORT-POLICY-AND-WIRING** — **Done**
- Typed **`ObservabilityExportPolicy`** with explicit allow/drop/hash posture (`apply_observability_export_policy`, `try_export_observability_envelope`)
- Default local-first posture remains **disabled by default**; **`export_content=false`** by default; strict redaction by default
- Raw prompts, documents, RAG chunks, synthesized content, tool args, secrets, and full local file paths are **not exported by default**
- Exporter failure isolation added — exporter exceptions are logged and never fail product/runtime runs
- Minimal runtime lifecycle wiring implemented via **`make_observability_export_runtime_plugin`** (optional bus subscriber; defaults to **`NoOpObservabilityExporter`**; export runs after canonical bus recording)
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix remain OBS-EXPORT-5)
- JSONL/file exporter remains **OBS-EXPORT-3** (superseded below)
- OTLP/Elasticsearch remains **OBS-EXPORT-4**

**OBS-EXPORT-3 status (2026-06-28):**

- **OBS-EXPORT-3-SAFE-JSONL-FILE-EXPORTER** — **Done**
- **`JsonlObservabilityExporter`** added — local-first, explicit opt-in JSONL/file sink implementing **`ObservabilityExporter`**
- Exporter writes normalized **`ObservabilityExportEnvelope`** records (one JSON object per line, UTF-8, append by default)
- Exporter does **not** register globally; platform bootstrap registration remains deferred unless explicitly planned later
- Redaction/export policy remains upstream — exporter writes the envelope it receives; sanitized metadata-only records when used through **`apply_observability_export_policy`** / **`try_export_observability_envelope`**
- Raw content export remains **unsupported/disabled by default** (`export_content=false`)
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix remain OBS-EXPORT-5)
- OTLP/Elasticsearch remains **OBS-EXPORT-4**

**OBS-EXPORT-4A status (2026-06-28):**

- **OBS-EXPORT-4A-APPLICATION-ATTRIBUTES-CONTRACT** — **Done**
- Typed **`ApplicationObservabilityAttributes`** contract added — application developers extend/customize through inheritance (e.g. `LocalWorkspaceObservabilityAttributes`, `BillingObservabilityAttributes`)
- No arbitrary public `dict[str, Any]` metadata boundary — safe scalar/list values only (`str`, `int`, `float`, `bool`, `None`, `list[str]`)
- Namespaced attribute keys (`{namespace}.{field}`) with stable schema/version fields
- **`sanitize_application_observability_attributes`** applies policy/redaction before export — unsafe/raw/sensitive values rejected, dropped, or path-like values hashed
- **`ObservabilityExportEnvelope`** integrates optional `application_attributes` (pre-policy input) and `sanitized_application_attributes` (post-policy export)
- **`JsonlObservabilityExporter`** consumes already-normalized sanitized attributes via envelope JSON serialization
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix remain OBS-EXPORT-5)
- LKW remains **unchanged**

**OBS-EXPORT-4 status (2026-06-28):**

- **OBS-EXPORT-4-OTLP-ADAPTER** — **Done**
- **`OtlpObservabilityExporter`** added as first remote backend adapter — platform observability package only (not applications, not LKW-specific)
- Adapter consumes normalized **`ObservabilityExportEnvelope`** only; expects policy-approved envelopes from **`apply_observability_export_policy`** / **`try_export_observability_envelope`**
- Adapter consumes **`sanitized_application_attributes`** only — does **not** read or export raw **`application_attributes`**
- Adapter maps sanitized application attributes into OTLP-safe attribute keys (namespaced keys preserved)
- Adapter does **not** export raw content; **`export_content=false`** posture unchanged
- Injectable **`OtlpTransport`** protocol — no vendor SDK coupling; no network in unit tests
- Explicit opt-in — adapter is **not** globally registered in platform bootstrap
- Elasticsearch remains **deferred** unless separately planned
- Langfuse / Arize / Phoenix remains **OBS-EXPORT-5**
- Global bootstrap registration remains **deferred** unless explicitly planned later

**OBS-EXPORT-4B status (2026-06-28):**

- **OBS-EXPORT-4B-OTLP-HTTP-TRANSPORT** — **Done**
- **`OtlpHttpTransport`** added — concrete HTTP POST transport implementing **`OtlpTransport`**
- Transport belongs to platform observability only (not applications, not LKW-specific)
- Transport is explicit opt-in — **not** globally registered in platform bootstrap
- Transport sends only OTLP-safe JSON payloads produced by **`OtlpObservabilityExporter`** from policy-sanitized envelopes
- Redaction and failure isolation remain upstream — **`apply_observability_export_policy`** / **`try_export_observability_envelope`**
- Transport does **not** read or export raw **`application_attributes`**; no raw content export
- No vendor SDK coupling — lightweight **`httpx`** client only; injectable client for tests (no real network in unit tests)
- Operator/bootstrap wiring remains **deferred** unless explicitly planned later
- Langfuse / Arize / Phoenix remains **OBS-EXPORT-5**

**OBS-EXPORT-4C status (2026-06-28):**

- **OBS-EXPORT-4C-EXPLICIT-OTLP-OPERATOR-WIRING** — **Done**
- Explicit OTLP operator wiring helper added — **`build_otlp_observability_exporter`**, **`build_otlp_observability_export_runtime_plugin`**
- Typed operator config added — **`ObservabilityExportOperatorConfig`**, **`OtlpExportOperatorConfig`**
- Disabled by default (`enabled=false`); **`export_content=false`** by default and enforced in runtime plugin wiring
- No global bootstrap registration — operator/platform code must explicitly construct and register the plugin
- No LKW wiring — platform observability package only
- No raw content export; no raw **`application_attributes`** export — policy sanitization remains upstream
- OTLP HTTP export can now be explicitly assembled from operator config (**`ObservabilityExportPolicy`** + **`OtlpObservabilityIntegration`** + **`OtlpObservabilityExporter`** + **`OtlpHttpTransport`** + **`make_observability_export_runtime_plugin`**)
- Injectable transport for tests — no network in unit tests
- Langfuse / Arize / Phoenix remains **OBS-EXPORT-5**
- Elasticsearch remains **deferred** unless separately planned

**INTEGRATIONS-1C status (OTLP observability integration alignment):**

- **INTEGRATIONS-1C** — **Done**
- **`OtlpObservabilityIntegration`** added — first concrete observability vendor integration
- Derives from **`ObservabilityVendorIntegrationContract`**; `provider_id=otlp`
- Wraps existing **`OtlpObservabilityExporter`** / **`OtlpTransport`** as lower-level implementation details
- Operator wiring (**`build_otlp_observability_integration`**) constructs integration-backed OTLP export path explicitly
- Consumes only policy-sanitized envelopes; **`sanitized_application_attributes`** only — never raw **`application_attributes`**
- **`JsonlObservabilityExporter`** unchanged — classified as local file export sink, not remote observability vendor
- No global bootstrap registration; no LKW change; no Langfuse/Arize/Phoenix/Elasticsearch adapters

**INTEGRATIONS-1D status (LKW/local workspace observability platform wiring):**

- **INTEGRATIONS-1D** — **Done**
- **`build_local_workspace_observability_plugins`** added in LKW host — composes platform **`ObservabilityExportOperatorConfig`** only; no LKW-specific exporter
- Disabled by default; **`export_content=false`** enforced via platform **`build_otlp_observability_export_runtime_plugin`**
- LKW factory accepts optional **`observability_export`** and registers returned **`RuntimePlugin`** only at LKW bootstrap — no global registration
- OTLP integration-backed path only (**`build_otlp_observability_export_runtime_plugin`** → **`OtlpObservabilityIntegration`**); no direct **`OtlpHttpTransport`** from LKW
- LKW.2 pipeline unchanged; no vendor SDK in LKW; raw content/local paths not exported by default
- **OBS-EXPORT-5** remains **deferred** until Langfuse/Arize/Phoenix vendor adapters are implemented

**OBS-EXPORT-5 status (dependency on INTEGRATIONS-1A / INTEGRATIONS-1B / INTEGRATIONS-1C):**

- **OBS-EXPORT-5** remains **paused** until remaining observability integration alignment is complete (OTLP done in INTEGRATIONS-1C; Langfuse/Arize/Phoenix deferred)
- Generic **`PlatformIntegrationContract`** added in **INTEGRATIONS-1A** — **Done**
- Observability vendor specialization **`ObservabilityVendorIntegrationContract`** added in **INTEGRATIONS-1B** — **Done**
- OTLP aligned with **`ObservabilityVendorIntegrationContract`** in **INTEGRATIONS-1C** — **Done**
- Concrete Langfuse / Arize / Phoenix adapters remain **deferred** — must subclass **`ObservabilityVendorIntegrationContract`**, not ad-hoc exporter classes
- JSONL remains a local file export sink — not migrated to observability vendor integration contract in INTEGRATIONS-1C
- **No LKW change** in INTEGRATIONS-1C

**INTEGRATIONS-2A status (provider category contracts):**

- **`observability_backend`** provider folder aligns with existing **`ObservabilityVendorIntegrationContract`** — no duplicate observability backend contract
- Runtime **`integration_kind`** for observability vendor integrations remains **`observability_vendor`**; folder name **`observability_backend`** is documented via **`PlatformIntegrationKind.OBSERVABILITY_BACKEND`**
- Existing **`observability_backend`** catalog providers (Langfuse, Arize, Phoenix, Elasticsearch, Datadog, …) are **still awaiting migration/adaptation** to category contracts
- **OBS-EXPORT-5** remains **deferred** until those providers are adapted as concrete **`ObservabilityVendorIntegrationContract`** subclasses

**INTEGRATIONS-2B-LANGFUSE pilot status (2026-06-28):**

- Existing Langfuse **`observability_backend`** provider adapted to **`ObservabilityVendorIntegrationContract`** as pilot migration
- **`LangfuseObservabilityIntegration`** consumes policy-sanitized envelopes only; legacy **`ObservabilityBackend`** query facade unchanged
- **OBS-EXPORT-5** remains **not complete** until migration pattern is approved and remaining providers (Arize, Phoenix, …) are adapted

**Required decisions:**

- External sinks are optional subscribers/export targets, not semantic owners of Intergrax observability.
- Intergrax RuntimeEvent / trace / journal / diagnostic payloads remain the canonical source.
- Vendor SDKs must not be called directly from runtime hot paths, agents, or LKW product code.
- Exporter failure must never fail product runs.
- Raw prompts, raw documents, raw RAG chunks, raw synthesized content, secrets, and full local file paths are not exported by default.
- Redaction/export policy must run before external export.
- Default posture for local-first apps such as LKW: disabled by default, strict redaction, `export_content=false`.
- OBS-EXPORT depends on LKW.2.4 pipeline proof as a representative multi-agent workload.

**Delivery rule:** one `OBS-EXPORT-*` row per PR; export through normalized envelope only; no vendor SDK in runtime hot paths.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **OBS-EXPORT-1** | Code | P2 | **Done** | Normalized export envelope and exporter interface | Defines stable export envelope, exporter interface, no-op exporter, and test exporter. Uses existing spine/journal/runtime metadata as source. No vendor SDK. |
| **OBS-EXPORT-2** | Code | P2 | **Done** | Redaction/export policy and failure isolation | Explicit allow/drop/hash policy for exported fields. Export timeout/failure does not fail the run. Tests prove raw content is not exported. Minimal runtime plugin wiring via `make_observability_export_runtime_plugin`. |
| **OBS-EXPORT-3** | Code | P2 | **Done** | Safe JSONL/file exporter | **`JsonlObservabilityExporter`** writes policy-sanitized **`ObservabilityExportEnvelope`** JSONL records; local-first explicit opt-in; no global bootstrap registration; no vendor SDK; raw content export disabled by default. |
| **OBS-EXPORT-4A** | Code | P2 | **Done** | Typed application observability attributes contract | **`ApplicationObservabilityAttributes`** base + subclass extension; namespaced safe metadata; policy sanitization before export; envelope integration; no arbitrary public dict boundary; no vendor SDK. |
| **OBS-EXPORT-4** | Code | P2 | **Done** | First remote backend adapter: OTLP | **`OtlpObservabilityExporter`** + **`OtlpObservabilityExporterConfig`** + injectable **`OtlpTransport`**; maps policy-sanitized envelopes (including **`sanitized_application_attributes`**) to OTLP-safe log payloads; no vendor SDK; explicit opt-in; no global bootstrap registration. Elasticsearch deferred. |
| **OBS-EXPORT-4B** | Code | P2 | **Done** | OTLP HTTP transport | **`OtlpHttpTransport`** POSTs OTLP JSON to configured endpoint; explicit opt-in; platform observability only; policy-sanitized payloads only; failure isolation via **`try_export_observability_envelope`**; no vendor SDK; no global bootstrap registration. Operator wiring deferred. |
| **OBS-EXPORT-4C** | Code | P2 | **Done** | Explicit OTLP operator wiring | **`ObservabilityExportOperatorConfig`** + **`OtlpExportOperatorConfig`** + **`build_otlp_observability_exporter`** + **`build_otlp_observability_export_runtime_plugin`**; disabled by default; **`export_content=false`** enforced; composes policy + OTLP exporter + HTTP transport + runtime plugin; no global registration; no LKW wiring; no raw content or raw **`application_attributes`** export. |
| **OBS-EXPORT-5** | Code | P3 | **Paused** | Langfuse / Arize / Phoenix adapter | **Paused** — OTLP aligned in **INTEGRATIONS-1C** (Done). Langfuse/Arize/Phoenix remain deferred. Concrete adapters must derive from **`ObservabilityVendorIntegrationContract`**. Adapter consumes normalized export envelope only. No runtime/vendor coupling. No raw content by default. JSONL remains local file sink — not vendor integration. |

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.9, §11 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — AUDIT-IDEAL observability rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-5.3 | §5 Policy | Governance health dashboard (GOV-PROD.1) | P4 | **Done** |
| AUDIT-IDEAL-21.1 | §21 Observability | Causal diagnostics beyond trace bridge (ops tooling) | P2 | **Done** |
| AUDIT-IDEAL-21.2 | §21 Observability | Quality / governance / cost health dashboard contracts | P2 | **Done** |
| AUDIT-IDEAL-21.3 | §21 Observability | Unified product observability dashboard | P4 | **Done** |
| AUDIT-IDEAL-30.2 | §30 Ops | Real deploy SLO window evidence (prod `W_OPS_RELEASE_CYCLES`) | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---
