# OBSERVABILITY — §12+ scenarios & control

**Parent hub:** [`OBSERVABILITY.md`](../OBSERVABILITY.md)

## 12. Tier responsibilities

| Tier | Responsibility | Must NOT |
|------|----------------|----------|
| **Tier-0** | Metrics bridges (LLM, RAG), parser trace export, observability integration providers | Own per-agent trace format |
| **Tier-1** | Spine, bus, bridge, journal, wiring, middleware, debug API | Contain business logic |
| **Tier-2** | Domain `DiagnosticPayload` extensions; consume spine via AgentEngine | Import trace DB, custom buses |
| **Tier-3** | `wire_application_observability`, profile selection, optional sinks | Reimplement RunTraceWriter |

---

## 13. Relationship to other architecture docs

| Document | Relationship |
|----------|--------------|
| [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §33, §42.1, §42.24 | Normative summary; this doc is the deep dive |
| [architecture/NEXUS_EXECUTION_FLOW.md](architecture/NEXUS_EXECUTION_FLOW.md) | Execution narrative; cross-links execution phases |
| [architecture/CRITIC_VERIFICATION.md](architecture/CRITIC_VERIFICATION.md) | Critic trace steps on the spine |
| [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | `ADAPTIVE_*` events |
| [guides/HARNESS_ENVIRONMENT.md](guides/HARNESS_ENVIRONMENT.md) | Lab OTLP, env vars, local backends |
| [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) | LLM metrics plane |
| [guides/AGENT_CREATION_GUIDE.md Appendix Q](guides/AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) | Author wiring checklist |

---

## 14. Maturity model

| Level | Criteria |
|-------|----------|
| **L0** | Ad-hoc logging only |
| **L1** | Trace file per run, no correlation |
| **L2** | TraceEvent + SQLite, partial bus |
| **L3** | Unified journal, DiagnosticPayload guard, wiring CI |
| **L4** | Typed RuntimeEvent payloads, TraceScope tree, catalog emission, extension SDK, journal export (**current — OBS-BUS Done**) |

Audit map §21 score: **L4** (OBS-BUS-7 gate evidence).

---

## 15. Code map (implementation reference)

| Concern | Path |
|---------|------|
| Trace models | `intergrax/runtime/nexus/tracing/trace_models.py` |
| Trace payloads | `intergrax/runtime/nexus/tracing/**/*.py` |
| Runtime events | `intergrax/runtime/events/runtime_event.py` |
| Event catalog (SSOT) | `intergrax/runtime/events/event_catalog.py` |
| Domain signals (target) | `intergrax/runtime/events/signals.py`, `emit_context.py` |
| Journal query (target) | `intergrax/runtime/events/journal_query.py` |
| Event bus | `intergrax/runtime/events/event_bus.py` |
| Trace bridge | `intergrax/runtime/events/trace_bridge.py` |
| Emitter + TraceScope | `intergrax/runtime/observability/emitter.py`, `trace_scope.py` |
| Extension SDK | `intergrax/runtime/observability/extension_sdk.py`, `intergrax/scaffold/tracing_templates.py` |
| Persistence conformance | `intergrax/runtime/observability/persistence_conformance.py`, `events/stores/document_backed_runtime_event_store.py` |
| Unified journal | `intergrax/runtime/events/unified_run_journal.py` |
| Journal export | `intergrax/runtime/observability/journal_export.py`, `export_bridge.py` |
| Nexus wiring | `intergrax/runtime/nexus/observability_wiring.py` |
| App wiring | `intergrax/applications/_shared/observability_wiring.py` |
| Pipeline emit | `intergrax/runtime/nexus/engine/runtime_state.py` |
| Task lifecycle | `intergrax/runtime/task/task_trace.py` |
| Critic trace | `intergrax/runtime/critic/trace.py` |
| Graph trace | `intergrax/runtime/nexus/orchestration/graph_trace_callbacks.py` |
| Middleware | `intergrax/runtime/middleware/trace_middleware.py` |
| Metrics export | `intergrax/runtime/metrics/export.py` |
| Debug API | `intergrax/debug/router.py`, `formatters.py` |
| SQLite stores | `intergrax/runtime/nexus/tracing/sqlite_run_trace_store.py`, `events/store.py` |
| Gates | `scripts/check_observability_gates.py`, `test_observability_layer_depth_gate.py`, emission/schema/persistence audits |

---

## 16. Verification

After harness observability changes:

```bash
uv run pytest -m gate -q
python scripts/check_harness_no_getattr.py
uv run python scripts/check_observability_gates.py
```

`check_observability_gates.py` runs trace-bridge catalog, emission coverage, payload schema registry, persistence conformance, and L4 depth gate tests (CI: `.github/workflows/unit-tests.yml`).

---

## 17. Session closeout

**OBS-BUS (L4):** **Done** (2026-06-08) — unified spine, typed payloads, extension SDK, journal export, CI gates.

**OBS-EVOL-9 (P1-ARCH-02):** **Done** (2026-06-17) — layered identity (`event_kind`, `EventCatalog`, `DOMAIN_SIGNAL`, profile subscriptions, W3C trace context). Publication-ready spine (56 types). See plan OBS-EVOL-9 register and §4.4.7–4.4.13.

**Not in spine scope:** product dashboards (§6.3a), mandatory external APM, per-agent private trace DBs.

---

## 18. Execution Boundary Export (EBE) — optional side channel

**Status:** PoC v1 **Done** · PoC v2 (EBE-8) **Done** (partner validated) · **EBE-9 host signing Done** (partner validated).  
**Reference host:** `applications/attestation_demo/` · **ADR:** [ADR-OBS-002](../adr/entries/2026-06-13/ADR-OBS-002.md) · [ADR-OBS-004](../adr/entries/2026-06-19/ADR-OBS-004.md)

EBE is an **optional** export path for **unsigned, vendor-neutral** tool-boundary facts. It complements — does not replace — the Harness Observability Spine (HOS).

| Principle | Rule |
|-----------|------|
| **Emit at invoker boundary** | `RuntimeToolInvoker` hook after tool execution |
| **Emit at kernel step boundary** | `HarnessKernel.execute_step` hook after trace append (EBE-8) |
| **Event-first, receipt-second** | Platform emits `execution_boundary_event.v1`; external products sign receipts |
| **One event, one receipt** | Partner maps each `boundary_events[]` element to a separate `client_observed` receipt |
| **Non-blocking** | Buffer/sink failures never fail tool invoke |
| **Honest trust** | Unsigned when signing off (`signed: false`); host-signed when EBE-9 enabled (`host_attested`); no implied `server_attested` |
| **HOS unchanged** | Unified journal, trace bridge, middleware — no receipt logic |

### Schema

`execution_boundary_event.v1` — Pydantic model in `intergrax/runtime/attestation/execution_boundary_event.py`.

| Field | Role |
|-------|------|
| `boundary_type` | `tool_execution` (invoker) or `harness_step` (kernel) |
| `event_id` | Stable UUID per event (receipt key) |
| `event_sequence` | Monotonic per `run_id`, assigned by `BoundaryEventBuffer` |
| `policy_verdicts` / `step_outcome` | Harness-step events only |

### Configuration

`ExecutionBoundaryExportProfile` on `ApplicationEnvironmentProfile` (`step_level_enabled` for EBE-8) → `attestation_runtime_bridge.py` → `RuntimeConfig.execution_boundary_export` + optional `BoundaryEventBuffer`. UAEP and ACP session loops copy settings into `StepKernelContext` via `kernel_wiring.py`.

### PoC v2 delivery (EBE-8)

Synchronous API response (`boundary_events[]`) returns **two events per demo run** when `step_level_enabled=true`: `tool_execution` (seq 1) then `harness_step` (seq 2). Webhook sink remains **deferred**.

### PoC v3 delivery (EBE-9 host signing)

When `host_signing_enabled=true`, each event includes `signed: true` and a `host_attestation` envelope. Intergrax:

1. Canonicalizes the unsigned event → `signed_payload_hash`
2. Signs canonical JSON host-attestation statement (`boundaryattest.host-attestation.v1`)
3. Exposes `trust_model.recommended_receipt_role: host_attested`

Unsigned v2 remains available when signing is disabled. Golden vector: `applications/attestation_demo/partner_handoff/ebe9_golden_vector.v1.json`. Spec: `EBE-9_HOST_SIGNING.md`.

**Partner validation — EBE-8 (unsigned v2, 2026-06):** BoundaryAttest adapter confirmed one `client_observed` receipt per event, hash parity, independent verification, and intentional dual claims on the failed-tool fixture. Reference: `agent_experiment_runtime` @ `106aee776fcc6053e8265b9c3656638d107d351d`.

**Partner validation — EBE-9 (host signing, 2026-06):** BoundaryAttest verifier @ `61be9918bc8f91fc8f160e0392d2914f38f3d4cb` passed golden vector byte-for-byte, 39/39 tests, live two-event response from Intergrax @ `96b7f997`, unsigned v2 regression, and negative tamper cases. Host signature verified separately; partner receipts remain `client_observed`. Handoff docs: `agent_experiment_runtime` @ `13102cfaff1a7a9d212c16cd16587477cc533dc0`.

**Trace correlation scope:** `GET /debug/tasks/{run_id}/trace` supports run/task-level journal comparison (agent, capability, graph node, critic, task state). It does **not** expose EBE `event_id`, `step_id`, or `tool_id`; exact per-event correlation uses the live `boundary_events[]` from `POST /poc/run` (or buffered replay endpoint). Enriching HOS trace with EBE identifiers is optional future work, not a PoC v2 requirement.

### Non-goals

- Intergrax receipt product or BoundaryAttest embedding
- Mandatory EBE on all hosts

---

*This document is the canonical observability architecture. Update it when changing spine contracts, emission rules, or persistence profiles. Implementation status: [Phase OBS-BUS — Done](../plan/OBSERVABILITY.md). EBE PoC v1: [Phase EBE](../plan/OBSERVABILITY.md#phase-ebe--execution-boundary-export-partner-poc).*
