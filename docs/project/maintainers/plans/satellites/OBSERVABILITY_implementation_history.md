# OBSERVABILITY — implementation history + LC closeout

**Parent hub:** [`OBSERVABILITY.md`](../OBSERVABILITY.md)

> **Plan ownership:** Implementation phases and LC closeout below. Historical audit findings/verdicts archived at [docs/audit_results/legacy/plan-audit-history/OBSERVABILITY_implementation_history.md](../../../../audit_results/legacy/plan-audit-history/OBSERVABILITY_implementation_history.md).


## Phase IDEAL-L3 — Observability ops depth (Band 2ax)

**Register:** [`plan/IDEAL_HARNESS_L3.md`](IDEAL_HARNESS_L3.md)

| ID | Deliverable | Status |
|----|-------------|--------|
| IDEAL-21.1 | `harness_slos.py` SLO catalog types | **Done** |
| IDEAL-21.2 | Runbook index (HARNESS_ENVIRONMENT ORCH-5.5) | **Done** |
| IDEAL-21.3–21.6 | Cost dashboard, emission audit, OTLP all hosts | **Done** (see [`IDEAL_HARNESS_L3.md`](IDEAL_HARNESS_L3.md) §21) |

---

### 6.1n Harness implementation queue — observability closeout (closed)

**Purpose:** Single ordered list for **Phase OBS** (Band 2t). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **OBS-DOC.1** | Docs | **Done** | Appendix Q + cross-refs | Author map complete |
| 2 | **OBS-1** | Code | **Done** | `observability_runtime_bridge` + `observability_wiring` | `test_harness_observability_wiring.py` |
| 3 | **OBS-2** | Code | **Done** | `observability_assembly_resolver` | wire-time validation tests |
| 4 | **OBS-3** | CI | **Done** | `check_harness_observability_wiring.py` | CI green |

**Suggested PR order (complete):** OBS-DOC.1 → OBS-1 → OBS-2 → OBS-3.### 6.1o Harness implementation queue — reliability closeout (closed)

**Purpose:** Single ordered list for **Phase REL** (Band 2u). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REL-DOC.1** | Docs | **Done** | Appendix R + cross-refs | Author map complete |
| 2 | **REL-1** | Code | **Done** | `reliability_runtime_bridge` + `reliability_wiring` | `test_harness_reliability_wiring.py` |
| 3 | **REL-2** | Code | **Done** | `reliability_assembly_resolver` | wire-time validation tests |
| 4 | **REL-3** | CI | **Done** | `check_harness_reliability_wiring.py` | CI green |

**Suggested PR order (complete):** REL-DOC.1 → REL-1 → REL-2 → REL-3.

---

### 6.1al Harness implementation queue — Unified Observability Spine (closed)

**Purpose:** Single ordered list for **Phase OBS-BUS** (Band 2al). **Closed 2026-06-08** — all OBS-BUS rows **Done**; audit map §21 → **L4**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **OBS-BUS-0** | Docs | **Done** | `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README | Links resolve |
| 2 | **OBS-BUS-1** | Code | **Done** | `RuntimeEventPayload` registry | Payload registry gate |
| 3 | **OBS-BUS-2** | Code | **Done** | `ObservabilityEmitter` + `TraceScope` | Causal tree tests |
| 4 | **OBS-BUS-3** | Code | **Done** | Emission coverage gaps | `check_observability_emission_coverage.py` |
| 5 | **OBS-BUS-4** | Code/Docs | **Done** | Extension SDK + scaffold | Agent tracing template |
| 6 | **OBS-BUS-5** | Code | **Done** | Persistence conformance | Integration tests |
| 7 | **OBS-BUS-6** | Code | **Done** | OTLP/journal dual-write | `test_journal_export.py`, `test_export_bridge.py` |
| 8 | **OBS-BUS-7** | CI | **Done** | L4 §21 gates | `check_observability_gates.py` in CI; audit map §21 → L4 |

**Suggested PR order:** See [Phase OBS-BUS — Execution order](.#obs-bus--execution-order-recommended).

**Explicitly excluded:** Product dashboards (§6.3a); vendor-only APM as sole store.

### 6.1am Harness implementation queue — Memory intelligence depth (closed)

**Purpose:** Single ordered list for **Phase MEM-DEPTH** (Band 2am). **Closed 2026-06-08** — **26/26 Done**. Canonical: [plan/MEMORY.md](plan/MEMORY.md).

**Suggested PR order:** See [§6.2ab](plan/MEMORY.md#62ab-phase-mem-depth-execution-order-band-2am--closed).

**Explicitly excluded:** K.1/K.2, Mem0 SaaS, Redis session default — [§6.3a](.#63a-business-backlog-register-consolidated).

---

## Phase OBS — Observability control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (OBS-DOC.1 + OBS-1–3)


**Priority ladder:** **Band 2t** (§4.0) — closed; default queue = **§6.1** maintenance.

### OBS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| OBS-DOC.1 | OBS0 | **Appendix Q** — observability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| OBS-1 | OBS1 | **`observability_runtime_bridge`** + **`observability_wiring`** | **Done** | `observability_runtime_bridge.py`, `observability_wiring.py`, `runtime_config_bridge.py` | `test_harness_observability_wiring.py` |
| OBS-2 | OBS2 | **`observability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `observability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| OBS-3 | OBS3 | **Host observability CI** — `check_harness_observability_wiring.py` | **Done** | `scripts`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only observability dashboards — [§6.3a](.#63a-business-backlog-register-consolidated).

---

---

## Phase OBS-BUS — Unified Observability Spine

**Status:** **Done** (2026-06-08) — **8/8** deliverables · OBS-BUS-0–7 **Done**

**Purpose:** Implement the full **Harness Observability Spine (HOS)** — one bus for Harness, applications, and agents; typed extension; causal trees; complete catalog emission; L4 audit §21.

**Architecture:** [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **ADR:** [ADR-OBS-001](adr/entries/2026-06-08/ADR-OBS-001.md)


**Priority ladder:** **Band 2al** (§4.0) — runs **after** Phase CRIT-V (Band 2ak) or in parallel §6.1 maintenance slices; **one OBS-BUS ID per PR**.

**Depends on:** Phase OBS (wiring) **Done** · OBS-DEPTH.1/2 **Done** · FAUDIT-OBS.1 **Done**

### OBS-BUS — Master register

| ID | Area | Deliverable | Status | Modules / artifacts | Acceptance |
|----|------|-------------|--------|---------------------|------------|
| OBS-BUS-0 | OBS0 | **Architecture canon** — `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README links | **Done** | `docs/project/architecture/OBSERVABILITY.md`, `docs/project/technical/adr/entries/2026-06-08/ADR-OBS-001.md` | Doc review; links from §33 |
| OBS-BUS-1 | OBS1 | **`RuntimeEventPayload` registry** — typed canonical payloads per `RuntimeEventType` (§42.23.1 families) | **Done** | `intergrax/runtime/events/payload_registry.py`, `payloads`, `schema_guard.py`, `trace_bridge.py`, `context_skill_recording.py` | Gate: `test_runtime_event_payload_registry.py` |
| OBS-BUS-2 | OBS2 | **`ObservabilityEmitter` + `TraceScope`** — single emit API; `parent_event_id` causal tree | **Done** | `intergrax/runtime/observability/emitter.py`, `trace_scope.py`, `runtime_state.py` | `RuntimeState.trace_event` delegates; `test_observability_emitter.py` |
| OBS-BUS-3 | OBS3 | **Emission coverage** — `AGENT_SELECTED`, `STEP_FAILED`, graph typed payloads, critic `evaluator_loop` bridge | **Done** | `agent_router.py`, `graph_trace_callbacks.py`, `task_trace.py`, `trace_bridge.py`, `graph_node_diag.py` | `check_observability_emission_coverage.py` |
| OBS-BUS-4 | OBS4 | **Extension SDK** — agent/app `DiagnosticPayload` scaffold, namespace rules, `PayloadSchemaRegistry` | **Done** | `extension_sdk.py`, `tracing_templates.py`, `new_agent.py`, `new_application.py` | `check_payload_schema_registry.py` |
| OBS-BUS-5 | OBS5 | **Persistence conformance** — Cassandra/ES adapters implement same protocols; profile docs | **Done** | `document_backed_runtime_event_store.py`, `persistence_conformance.py`, profile wiring | `check_observability_persistence_conformance.py` |
| OBS-BUS-6 | OBS6 | **Export sinks** — OTLP dual-write from unified journal; parser trace link | **Done** | `journal_export.py`, `export_bridge.py`, `task_events.py`, `platform_wiring.py` | `TASK_COMPLETED` carries `journal_ref`; export plugin dual-writes OTLP JSON + parser trace |
| OBS-BUS-7 | OBS7 | **CI gates** — emission coverage + schema registry + L4 §21 evidence | **Done** | `scripts/maintenance/check_observability_gates.py`, emission/schema/persistence audits, CI workflow | Gate suite green; audit map §21 → **L4** |

### OBS-BUS — Execution order (recommended)

```text
OBS-BUS-0 (docs) → OBS-BUS-1 (typed payloads)
  → OBS-BUS-2 (emitter + TraceScope)
  → OBS-BUS-3 (coverage gaps)
  → OBS-BUS-4 (extension SDK)
  → OBS-BUS-5 (persistence)
  → OBS-BUS-6 (sinks)
  → OBS-BUS-7 (gates / L4 closeout)
```

**DoD:** All OBS-BUS rows **Done**; `build_unified_run_journal` reproduces full Nexus+AgentEngine path without reading source; every `RuntimeEventType` in §42.1.2 has ≥1 production emitter; `parent_event_id` populated for tool/LLM/delegation; extension scaffold documented; gate green.

**Explicitly excluded:** product-specific dashboards (§6.3a); replacing external APM as mandatory deployment.

---

---

### Phase D — Observability and Experiments



**Goal:** §19, §35 — laboratory tooling (not SaaS UI).



| # | Deliverable | Status | Notes |

|---|-------------|--------|-------|

| D.0 | §42 P4.1 Event Bus wiring | **Done** | `RuntimeEventBus`, `trace_bridge`, NexusLoop |

| D.1 | Debug CLI | **Done** | `python -m intergrax.debug tasks list/|show/|trace` |

| D.2 | Minimal debug API | **Done** | FastAPI `GET /debug/tasks` on trace store |

| D.3 | Experiment registry | **Done** | SQLite registry; CLI + `GET/POST /debug/experiments` |

| D.4 | Experiment workflow API | **Done** | `intergrax/experiments/workflow.py`, `tests/unit/experiments` |

| D.5 | Cost in trace | **Done** | `AgentExecutionResult.cost` from LLM usage / runtime stats |

---

## Phase EBE — Execution Boundary Export (partner PoC)

**Architecture:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) §18 · **ADR:** [ADR-OBS-002](../adr/entries/2026-06-13/ADR-OBS-002.md) · **Reference host:** `applications/attestation_demo`

| ID | Deliverable | Status | Modules / artifacts | Acceptance |
|----|-------------|--------|---------------------|------------|
| EBE-1 | `execution_boundary_event.v1` + invoker hook + memory buffer | **Done** | `intergrax/runtime/attestation` | `tests/unit/runtime/attestation` |
| EBE-2 | `ExecutionBoundaryExportProfile` + wiring bridge | **Done** | `attestation_runtime_bridge.py`, `environment_profile.py` | host runtime wiring |
| EBE-3 | `attestation_demo` host + `POST /poc/run` | **Done** | `applications/attestation_demo` | `attestation_demo_tests` |
| EBE-4 | `boundary_demo_agent` + `records.put` lab wiring | **Done** | `agents/boundary_demo`, `host/tool_wiring.py` | PoC smoke |
| EBE-5 | Partner handoff (README + sample JSON) | **Done** | `partner_handoff` | committed request/response fixtures |
| EBE-6 | Domain doc + harness ADR (trust model) | **Done** | `architecture/OBSERVABILITY.md` §18, ADR-OBS-002 | doc pair + `check_harness_adr.py` |
| EBE-7 | Webhook sink | Deferred | `sinks/webhook.py` | Phase 2 |
| EBE-8 | HarnessKernel step-level events (`harness_step`, `event_sequence`) | **Done** (partner validated) | `harness_boundary_emitter.py`, `HarnessKernel._finish_step` | Live Docker @ `106aee77`; AgentReceipt 28/28 + live example |
| EBE-9 | Host-side event signing (Ed25519 statement) | **Done** (partner validated) | `host_attestation.py`, `canonical_json.py`, profile `host_signing_enabled` | Live Docker @ `96b7f997`; BoundaryAttest `61be9918` 39/39 + golden vector; [ADR-OBS-004](../adr/entries/2026-06-19/ADR-OBS-004.md) |

---

## Phase OBS-EVOL-9 — Layered event catalog (P1-ARCH-02)

**Status:** **Done** (2026-06-17) — M0–M3 register complete; OBS-EVOL-9.9 (`runtime_event.v2`) deferred (low priority, post-publication)
**Goal:** Scale HOS beyond flat `RuntimeEventType` growth — spine + `event_kind` + `EventCatalog` — **before external v1 publication** (no external migration).

**ADR:** [`ADR-OBS-003`](../adr/entries/2026-06-17/ADR-OBS-003.md)
**Architecture:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) §4.4 · UAEP [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.1.6

### OBS-EVOL-9 — Strategic Architecture Review (accepted 2026-06-17)

| SAR | Deliverable | Folded into |
|-----|-------------|-------------|
| SAR-01 | `EmitContext` protocol for emit APIs | OBS-EVOL-9.3 |
| SAR-02 | `retention_class` on `EventCatalogEntry` | OBS-EVOL-9.1 |
| SAR-03 | Declarative `kind_prefix` subscriptions on `ObservabilityProfile` | OBS-EVOL-9.10 |
| SAR-04 | W3C `traceparent` / `tracestate` on `RuntimeEvent` | OBS-EVOL-9.11 |
| SAR-05 | `sample_rate` metadata + bus enforcement | OBS-EVOL-9.1 metadata · OBS-EVOL-9.6 enforcement |
| SAR-06 | Deprecation shim (old spine → `DOMAIN_SIGNAL`) | OBS-EVOL-9.7 |
| SAR-07 | `JournalQuery` read-model filters | OBS-EVOL-9.5 |
| SAR-08 | `LLMStreamEvent.event_kind` namespace lint | OBS-EVOL-9.6 |
| SAR-09 | Mandatory redaction on `emit_domain_signal` | OBS-EVOL-9.3 |
| SAR-10 | Elevate `EventKindRegistry` to P1 | OBS-EVOL-9.4 |
| SAR-11/12 | Per-category buses / hierarchical enum | **Rejected** (ADR-OBS-003) |

| ID | Phase | Deliverable | Status | Priority | Acceptance |
|----|-------|-------------|--------|----------|------------|
| OBS-EVOL-9-DOC | M0 | Architecture §4.4 + plan register + ADR-OBS-003 + author guides | **Done** | **Critical** | This register · ADR · `EXTENSION_AUTHOR_GUIDE.md` §11 · `APPLICATION_CREATION_GUIDE.md` §8 · `AGENT_CREATION_GUIDE.md` §Q.5 |
| OBS-EVOL-9.1 | M1 | `EventCategory` + `EventCatalogEntry` + `event_catalog.py` (`retention_class`, `sample_rate`, `consolidation_kind`; deprecate `phase_coverage.py` as SSOT) | **Done** | **Critical** | `event_catalog.py` · `test_event_catalog.py` |
| OBS-EVOL-9.2 | M1 | `event_kind` + `event_category` + `ops_hint` on `RuntimeEvent`; auto-fill from catalog | **Done** | **Critical** | `runtime_event.py` · `test_runtime_event_kind.py` |
| OBS-EVOL-9.3 | M1 | `EmitContext` + `emit_domain_signal()` (redaction) + `emit_platform_event()` | **Done** | **Critical** | `signals.py` · `emit_context.py` · `test_domain_signals.py` |
| OBS-EVOL-9.4 | M1 | `EventKindRegistry` for extension kinds (agents/apps namespaces) | **Done** | **Critical** | `event_kind_registry.py` · `test_event_kind_registry.py` |
| OBS-EVOL-9.5 | M2 | `RuntimeEventBus.subscribe(categories=, kind_prefix=, ops_hints=)` + `JournalQuery` | **Done** | High | `event_bus.py` · `journal_query.py` · `test_event_bus_taxonomy_subscribe.py` |
| OBS-EVOL-9.6 | M2 | `scripts/maintenance/check_event_catalog.py` + sampling enforcement + LLM `event_kind` namespace lint | **Done** | High | CI script · `test_event_bus_sampling.py` · extend `check_observability_gates.py` |
| OBS-EVOL-9.7 | M2 | **Pre-release spine consolidation** — 74 → 56; `DOMAIN_SIGNAL` + read shim | **Done** | **Critical** | `spine_consolidation.py` · emitters · `test_spine_consolidation.py` · `check_event_catalog.py` |
| OBS-EVOL-9.8 | M2 | Scaffold: `emit_domain_signal` template in `new_agent` / `new_application` | **Done** | Medium | `signal_templates.py` · `test_scaffold_domain_signals.py` |
| OBS-EVOL-9.9 | M3 | Optional `runtime_event.v2` envelope (`event_kind` required) | **Deferred** | Low | Opt-in `schema_version`; v1 indefinite — post-publication backlog |
| OBS-EVOL-9.10 | M2 | Declarative bus subscriptions on `ObservabilityProfile` | **Done** | P2 | `sub_profiles.py` · `event_subscription_registry.py` · `observability_wiring.py` |
| OBS-EVOL-9.11 | M3 | W3C Trace Context (`traceparent` / `tracestate`) on `RuntimeEvent` + OTLP bridge | **Done** | P3 | `w3c_trace_context.py` · `journal_export.py` · `export_bridge.py` |

**Suggested PR order:** OBS-EVOL-9-DOC → OBS-EVOL-9.1 → OBS-EVOL-9.2 → OBS-EVOL-9.3 → OBS-EVOL-9.4 → OBS-EVOL-9.6 → OBS-EVOL-9.5 → OBS-EVOL-9.7 → OBS-EVOL-9.8 → OBS-EVOL-9.10 → OBS-EVOL-9.11.

**Explicitly out of scope:** per-category event buses; hierarchical enums; mandatory external APM.

### OBS-EVOL-9 — Verification gates (verified 2026-06-17)

```bash
uv run pytest tests/unit/runtime/events/ -q
uv run python scripts/maintenance/check_event_catalog.py
uv run python scripts/maintenance/check_observability_gates.py
python scripts/maintenance/check_harness_adr.py
```

---

## Phase OBSERVABILITY-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates OBS-EVOL-9 + OBS-BUS closeout; no open P0/P1
**Prerequisites:** OBS-EVOL-9 M0–M3 **Done** (9.9 deferred) · ADR-OBS-001/003
**Goal:** Formal Full Harness LC closeout — gate verification, journal
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| OBS-LC-S1 | **Re-audit** — OBS register + spine verdict | **Done** | High | No P0/P1 |
| OBS-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| OBS-LC-S3 | **Gate verification** | **Done** | High | 87 event tests · `check_observability_gates` |
| OBS-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** OBS-EVOL-9.9 `runtime_event.v2` · product dashboards §6.3a

### 6.1av Harness implementation queue — Observability audit maintenance (planned)

**Source:** Layer 16 audit (2026-06-18) — `OBSERVABILITY` layers 21, 30 · [`../audit_results/2026-06-18/OBSERVABILITY.md`](../audit_results/2026-06-18/OBSERVABILITY.md)
**Priority ladder:** **Band 1** (§6.1) — post-publication + prompt sync; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **OBS-MAINT-01** | Schema | P3 | **Done** | OBS-EVOL-9.9 — `runtime_event.v2` schema evolution (post-publication) | `PREVIEW_RUNTIME_SCHEMA_VERSIONS` + conformance tests |
| 2 | **OBS-MAINT-02** | Cross-ref | P4 | **Done** | Product dashboards §6.3a — cross-ref [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) Phase K owner | Architecture cross-ref; no duplicate OBS product scope |
| 3 | **OBS-MAINT-03** | Docs | P3 | **Done** | Audit prompt sync — OBS-EVOL-9 M0–M3 **Done** in known gaps | `docs/audit_results/OBSERVABILITY.md` regenerated |
| 4 | **OBS-MAINT-04** | Docs | P3 | **Done** | Pre-release spine consolidation checklist — operator runbook row | Checklist in architecture §pre-release |

**Suggested PR order:** OBS-MAINT-03 → OBS-MAINT-04 → OBS-MAINT-01 → OBS-MAINT-02.

**Cross-domain:** CE-MAINT-01/02 — context assembly OTel/cost (OBS spine consumer).

---

*End of Observability Implementation Plan.*
