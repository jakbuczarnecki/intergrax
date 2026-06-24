# OBSERVABILITY — Eval Control Plane (OECP)

**Parent hub:** [OBSERVABILITY.md](../OBSERVABILITY.md)  
**Architecture:** [architecture/OBSERVABILITY.md](../../architecture/OBSERVABILITY.md#observability--evaluation-control-plane) · [architecture/satellites/OBSERVABILITY_extended_depth.md](../../architecture/satellites/OBSERVABILITY_extended_depth.md)  
**Audit source:** [audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md](../../audit/OBSERVABILITY_EVALUATION_CONTROL_PLANE_AUDIT.md)  
**Status:** Active implementation register — **2026-06-24**

> Closed phases (OBS, OBS-BUS, EBE, OBS-EVOL-9) remain in [OBSERVABILITY_audit_history.md](OBSERVABILITY_audit_history.md). Do not re-open without operator reprioritization.

---

## Delivery model

```text
HOS run -> evidence ledger -> eval snapshot -> metric results -> regression gates -> perturbation suites -> controlled adaptation
```

**Rules:** OECP consumes HOS only; no parallel trace system; external vendors are optional sinks; CVL emits verdicts — OECP stores and gates (see [CRITIC_VERIFICATION.md](../../architecture/CRITIC_VERIFICATION.md#boundary-with-observability--evaluation-control-plane-oecp)).

One phase ID (or cohesive sub-ID batch) per PR unless operator reprioritizes.

---

## Phase register

| ID | Type | Deliverable | Status | Acceptance criteria |
|----|------|-------------|--------|---------------------|
| **OBS-ECP-0** | Docs | OECP architecture canon in hub, extended satellite, CVL cross-ref, Tier-3 profile surfaces, and this plan register | **Done** | Hub states HOS-only spine + OECP scope; extended satellite contains OECP sections; CVL/OECP boundary explicit; Tier-3 §22.1.1 surfaces documented; audit remains separate source at docs/audit/. **Done 2026-06-24** — architecture canon reviewed; implementation phases remain Planned. |
| **OBS-ECP-1** | Code | Trace Completeness Contract (TraceCompletenessProfile, checker, report, gate) | **Planned** | Required eval-grade evidence dimensions validated per profile; gate modes observe / warn / block_release / block_canary_promotion / fail_ci; missing prompt/tool/RAG/critic evidence produces findings with refs |
| **OBS-ECP-2** | Code | Evidence Ledger — eval-ready records derived from HOS/journal | **Planned** | Normalized evidence kinds (prompt, model I/O, tool, RAG, context, policy, critic, custom telemetry) with source_event_id / source_trace_event_id; no full trace duplication; redaction metadata persisted |
| **OBS-ECP-3** | Code | Eval Registry v2 | **Planned** | EvalCase, EvalDataset, EvalRun, EvalRunSnapshot, EvalMetricResult, EvalObservationV2, EvalRegressionResult, perturbation lineage; observations carry evidence refs and version pins |
| **OBS-ECP-4** | Code | Metric and Eval Plugin SDK | **Planned** | EvalMetricPlugin protocol + registry; built-in deterministic/trajectory/ops metrics; custom plugins score without core runtime changes |
| **OBS-CTP-1** | Code | Custom Telemetry Extension Plane — provider registry and profile wiring | **Planned** | TelemetryProvider returns typed DiagnosticPayload / RuntimeEventPayload only; schema_id, namespace, redaction, tenant isolation, retention/export/sampling enforced |
| **OBS-CTP-2** | Code | TelemetryProvider / TelemetryEnricher contracts | **Planned** | Enrichers augment spine events safely; providers integrate through HOS → journal → evidence path; no private logger bypass |
| **OBS-CTP-3** | Code | EventSubscriptionHandler reaction handlers | **Planned** | Declarative profile subscriptions drive typed reactions (memory snapshot, cost anomaly, eval candidate, external sink); handlers cannot bypass HOS |
| **OBS-PERT-1** | Code | Counterfactual Engine | **Planned** | Mutation ops (replace/swap/negate/remove/add constraints, entity/date/role/tool/RAG variants) with full parent lineage |
| **OBS-PERT-2** | Code | Interpolation Engine | **Planned** | Case-blending generators with expected_behavior_delta; parent case refs preserved |
| **OBS-EXT-1** | Code | External Observability Workbench Sync | **Planned** | Non-blocking OTLP/Langfuse/LangSmith (and similar) export; retry + DLQ; redaction before export; vendor deep links stored in Intergrax; canonical semantics remain platform-owned |
| **OBS-GATE-1** | Code/CI | CI and release gates | **Planned** | Trace completeness CI gate; eval regression gate vs baseline; invalid custom schema/plugin rejected at gate |
| **OBS-UX-1** | CLI/API | Debug/workbench surfaces | **Planned** | Run evidence, eval score, regression diff, and custom telemetry views queryable with redaction-safe filters |

---

## Phase detail — OBS-ECP-0 (architecture canon)

| Sub-ID | Type | Deliverable | Status |
|--------|------|-------------|--------|
| OBS-ECP-0.1 | Docs | Hub OECP summary | **Done** |
| OBS-ECP-0.2 | Docs | Extended satellite OECP sections | **Done** |
| OBS-ECP-0.3 | Docs | CVL cross-reference | **Done** |
| OBS-ECP-0.4 | Docs | Tier-3 profile surfaces §22.1.1 | **Done** |
| OBS-ECP-0.5 | Docs | This plan satellite | **Done** |

---

## Cross-plan references

| Domain | Relationship |
|--------|--------------|
| [CRITIC_VERIFICATION.md](../../architecture/CRITIC_VERIFICATION.md) | CVL emits verdicts; OECP stores evidence and regression views |
| [ADAPTIVE_HARNESS_INTELLIGENCE.md](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | May consume eval/regression results; must not invent private eval records |
| [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | Experiments use Eval Registry datasets; no duplicate ledger |
| [TIER3_APPLICATION_ENVIRONMENT.md](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) | Profile-level opt-in surfaces only |

---

*Update this register when closing an OECP phase. Keep historical OBS/OBS-BUS/EBE rows in [OBSERVABILITY_audit_history.md](OBSERVABILITY_audit_history.md).*
