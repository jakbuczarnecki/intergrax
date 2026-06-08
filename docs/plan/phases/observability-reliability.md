# Implementation Phases — Observability Reliability

**Hub:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

---

## Phase OBS — Observability control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (OBS-DOC.1 + OBS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §21; complements GOV-AUDIT Appendix H; author map: `AGENT_CREATION_GUIDE.md` **Appendix Q**.

**Priority ladder:** **Band 2t** (§4.0) — closed; default queue = **§6.1** maintenance.

### OBS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| OBS-DOC.1 | OBS0 | **Appendix Q** — observability control plane closeout | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |
| OBS-1 | OBS1 | **`observability_runtime_bridge`** + **`observability_wiring`** | **Done** | `observability_runtime_bridge.py`, `observability_wiring.py`, `runtime_config_bridge.py` | `test_harness_observability_wiring.py` |
| OBS-2 | OBS2 | **`observability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `observability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| OBS-3 | OBS3 | **Host observability CI** — `check_harness_observability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only observability dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

## Phase OBS-BUS — Unified Observability Spine

**Status:** **Done** (2026-06-08) — **8/8** deliverables · OBS-BUS-0–7 **Done**

**Purpose:** Implement the full **Harness Observability Spine (HOS)** — one bus for Harness, applications, and agents; typed extension; causal trees; complete catalog emission; L4 audit §21.

**Architecture:** [`OBSERVABILITY_ARCHITECTURE.md`](OBSERVABILITY_ARCHITECTURE.md) · **ADR:** [ADR-OBS-001](adr/ADR-OBS-001.md)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §21 · complements Phase OBS (wiring closeout) · supersedes residual “live bus emit for all LLM paths” row when OBS-BUS-2 ships.

**Priority ladder:** **Band 2al** (§4.0) — runs **after** Phase CRIT-V (Band 2ak) or in parallel §6.1 maintenance slices; **one OBS-BUS ID per PR**.

**Depends on:** Phase OBS (wiring) **Done** · OBS-DEPTH.1/2 **Done** · FAUDIT-OBS.1 **Done**

### OBS-BUS — Master register

| ID | Area | Deliverable | Status | Modules / artifacts | Acceptance |
|----|------|-------------|--------|---------------------|------------|
| OBS-BUS-0 | OBS0 | **Architecture canon** — `OBSERVABILITY_ARCHITECTURE.md` + ADR-OBS-001 + canon/README links | **Done** | `docs/OBSERVABILITY_ARCHITECTURE.md`, `docs/adr/ADR-OBS-001.md` | Doc review; links from §33 |
| OBS-BUS-1 | OBS1 | **`RuntimeEventPayload` registry** — typed canonical payloads per `RuntimeEventType` (§42.23.1 families) | **Done** | `intergrax/runtime/events/payload_registry.py`, `payloads/`, `schema_guard.py`, `trace_bridge.py`, `context_skill_recording.py` | Gate: `test_runtime_event_payload_registry.py` |
| OBS-BUS-2 | OBS2 | **`ObservabilityEmitter` + `TraceScope`** — single emit API; `parent_event_id` causal tree | **Done** | `intergrax/runtime/observability/emitter.py`, `trace_scope.py`, `runtime_state.py` | `RuntimeState.trace_event` delegates; `test_observability_emitter.py` |
| OBS-BUS-3 | OBS3 | **Emission coverage** — `AGENT_SELECTED`, `STEP_FAILED`, graph typed payloads, critic `evaluator_loop` bridge | **Done** | `agent_router.py`, `graph_trace_callbacks.py`, `task_trace.py`, `trace_bridge.py`, `graph_node_diag.py` | `check_observability_emission_coverage.py` |
| OBS-BUS-4 | OBS4 | **Extension SDK** — agent/app `DiagnosticPayload` scaffold, namespace rules, `PayloadSchemaRegistry` | **Done** | `extension_sdk.py`, `tracing_templates.py`, `new_agent.py`, `new_application.py` | `check_payload_schema_registry.py` |
| OBS-BUS-5 | OBS5 | **Persistence conformance** — Cassandra/ES adapters implement same protocols; profile docs | **Done** | `document_backed_runtime_event_store.py`, `persistence_conformance.py`, profile wiring | `check_observability_persistence_conformance.py` |
| OBS-BUS-6 | OBS6 | **Export sinks** — OTLP dual-write from unified journal; parser trace link | **Done** | `journal_export.py`, `export_bridge.py`, `task_events.py`, `platform_wiring.py` | `TASK_COMPLETED` carries `journal_ref`; export plugin dual-writes OTLP JSON + parser trace |
| OBS-BUS-7 | OBS7 | **CI gates** — emission coverage + schema registry + L4 §21 evidence | **Done** | `scripts/check_observability_gates.py`, emission/schema/persistence audits, CI workflow | Gate suite green; audit map §21 → **L4** |

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

## Phase REL — Reliability control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REL-DOC.1 + REL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §22; H-APP `ReliabilityProfile` **Done**; author map: `AGENT_CREATION_GUIDE.md` **Appendix R**.

**Priority ladder:** **Band 2u** (§4.0) — closed; default queue = **§6.1** maintenance.

### REL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-DOC.1 | REL0 | **Appendix R** — reliability control plane closeout | **Done** | `AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REL-1 | REL1 | **`reliability_runtime_bridge`** + **`reliability_wiring`** | **Done** | `reliability_runtime_bridge.py`, `reliability_wiring.py`, `runtime_config_bridge.py` | `test_harness_reliability_wiring.py` |
| REL-2 | REL2 | **`reliability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `reliability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| REL-3 | REL3 | **Host reliability CI** — `check_harness_reliability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only retry/fallback policies — [§6.3a](#63a-business-backlog-register-consolidated).

---

