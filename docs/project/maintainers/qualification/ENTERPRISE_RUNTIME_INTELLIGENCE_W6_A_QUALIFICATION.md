# Enterprise Runtime Intelligence — W6-A Architecture Inventory & Qualification

**Task:** W6-A — Enterprise Runtime Intelligence Foundation — Architecture Inventory & Qualification  
**Character:** Inventory + architecture analysis + ADR only (**no production implementation**)  
**Branch:** `development`

## Baseline

| Field | Value |
|-------|--------|
| START_HEAD | `6b51617752c3fda544e1b4d2a29da5ba21c9ded7` |
| ADR | [`ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md`](../architecture/ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md) |
| Architecture hub | [`ENTERPRISE_RUNTIME_INTELLIGENCE.md`](../../architecture/ENTERPRISE_RUNTIME_INTELLIGENCE.md) |

---

## ETAP 1 — Pełny inventory execution plane

### Execution Plane

| Ścieżka | Rola | Kluczowe elementy |
|---------|------|-------------------|
| `intergrax/runtime/execution/` | Root lifecycle (UE-10R1), boundary, budget, decision host ports | `runtime.py`, `host_task.py`, `orchestration.py`, `decision_recovery.py`, checkpoint/finalization persistence |
| `intergrax/runtime/nexus/` | Agent loop, graph execution, session, tracing | `nexus_loop.py`, `orchestration/graph_runner.py`, `execution/graph_executor.py` |
| `intergrax/runtime/task/` | Task model & unified runner | `task.py`, `unified_task_runner.py` |
| `intergrax/runtime/kernel/` | Step kernel | `step_kernel.py` |
| `intergrax/runtime/long_running/` | Task checkpoint, coordinator, scheduler integration | `coordinator.py`, `checkpoint_builder.py`, `runtime_checkpoint.py`, `store.py` |

### Reliability Plane (W1/W2 slice)

| Ścieżka | Rola |
|---------|------|
| `intergrax/runtime/resilience/` | Local adapters: recovery admission, dependency attempt boundary, provider retry budget/rate limit, policy resolver |
| `intergrax/contracts/resilience_policy.py` | Resilience policy contract |
| `intergrax/contracts/dependency_concurrency_admission.py` | W2 admission |
| `intergrax/contracts/execution_capacity_admission.py` | W1 capacity admission |

### Recovery Plane (W3)

| Ścieżka | Rola |
|---------|------|
| `intergrax/runtime/execution/fan_out_partial_recovery.py` | Partial topology recovery |
| `intergrax/runtime/resilience/task_resume_recovery_handoff.py` | Task resume handoff |
| `intergrax/runtime/resilience/decision_durable_recovery_handoff.py` | Decision durable recovery |
| `intergrax/contracts/recovery_admission.py` | Recovery **start** admission (TASK_RESUME, PARTIAL_TOPOLOGY, DECISION_DURABLE) |
| `intergrax/runtime/execution/decision_checkpoint_persistence.py` | Decision snapshot port |

### Cancellation & External Operations (W4)

| Ścieżka | Rola |
|---------|------|
| `intergrax/runtime/cancellation/` | `CancellationCoordinator`, `resume_admission.py` |
| `intergrax/runtime/external_operations/` | Admission gates, provider attempts, recovery gate, diagnostic evidence contributors |
| `intergrax/runtime/enterprise_reliability/` | ERL: uncertainty, reconciliation orchestration, plugin gateway |
| `intergrax/contracts/external_operations/` | Attempt, safety, reconciliation contracts |

### Observability Plane (W5)

| Ścieżka | Rola |
|---------|------|
| `intergrax/runtime/events/` | `RuntimeEvent`, `RuntimeEventBus`, taxonomy, journals |
| `intergrax/runtime/observability/` | Emitters, exporters |
| `intergrax/runtime/observability/event_delivery/` | Bounded sink, export bridge, OTLP/distributed adapters |
| `intergrax/contracts/event_delivery.py`, `observability_export.py` | Export ports |

### Adjacent intelligence (existing — not W6)

| Ścieżka | Rola | Relacja do W6 |
|---------|------|----------------|
| `intergrax/runtime/diagnostics/` | Central Diagnostics, Problem lifecycle | **Authority** — W6 consumes, does not replace |
| `intergrax/runtime/prediction/` | `PredictionEngine`, analyzers, history | Advisory risk — shared plugin pattern |
| `intergrax/runtime/prevention/` | Preventive actions | Action plane with governance |
| `intergrax/runtime/self_healing/` | Self-heal orchestration, workflows | Action plane — decision audit input for W6 |
| `intergrax/runtime/adaptive/` | Harness profile learning (L4) | Separate adaptive **product** surface |
| `intergrax/runtime/replay/` | Run records | Reconstruction input |

---

## ETAP 2 — Mapa obecnego modelu

| Obszar | Owner | Contract | Implementation | State | Gap (W6) |
|--------|-------|----------|----------------|-------|----------|
| **Execution** | `runtime/execution` + `nexus` | `ExecutionRuntime`, identity, boundary, `ExecutionTerminal*` | `runtime.py`, `nexus_loop.py`, `graph_executor.py` | Mature | Brak skorelowanego **outcome explanation** artefaktu |
| **Retry** | `resilience` + contracts | `resilience_policy`, provider retry budget | `local_provider_retry_budget.py`, `policy_resolver.py` | W1/W2 qualified | Retry widoczny w eventach; brak **inteligencji** łączącej retry storm + admission |
| **Recovery** | `long_running` + `execution` + `resilience` | `RecoveryAdmissionPort`, checkpoint ports | `fan_out_partial_recovery.py`, `*_recovery_handoff.py`, coordinator | W3 qualified | Brak **decision intelligence** dla recovery start vs outcome |
| **Cancellation** | `cancellation` + events | Cooperative task keys + `RuntimeEventType` cancel family | `CancellationCoordinator` | W4 patterns | Brak scoring / timeline intelligence |
| **Events** | `runtime/events` + W5 delivery | `RuntimeEvent`, export ports | Bus + bounded export pipeline | W5-H qualified | W6 powinien **czytać** te same fakty, nie duplikować bus |
| **Decisions** | Decision host + governance + execution persistence | `DecisionCheckpoint*`, finalization, verification stages | `decision_recovery.py`, sqlite/in-memory persistence | Partial CAS gap (W3 ADR) | Brak **audytowalnego** runtime decision scorecard |
| **Diagnostics** | `runtime/diagnostics` | Single authority (R1) | Problem lifecycle, read service | Enterprise qualified | Nie rozszerzać o ML narrative — W6 obok |
| **Prediction** | `runtime/prediction` | `PredictiveAnalyzer`, `PredictionEngine` | Registry + engines | R1–R5 docs | Oddzielny cel (risk); W6 dzieli SPI wzorzec |
| **Adaptive (harness)** | `runtime/adaptive` | `adaptive/contracts.py` (models) | Engines, profile stores | L4 R&D | Brak mostu **execution-plane** policy signals |
| **ERL** | `enterprise_reliability` | ERL plugin SPI, reconciliation | Orchestration + gateway | Target arch | W6 powinien korelować UNKNOWN/reconcile w timeline |

---

## ETAP 3 — Runtime Intelligence Model

### Informacje już posiadane (źródła faktów)

| Kategoria | Źródło | Przykłady |
|-----------|--------|-----------|
| Execution events | `RuntimeEvent` + persistence | `STEP_*`, `EXECUTION_FAILED`, `TOOL_*`, `RETRY_*` |
| Failures | Failure evidence recorder, diagnostic analyzers | `ExecutionFailureEvidence`, `EXECUTION_FAILED` |
| Retries | Events + attempt lifecycle | `RETRY_SCHEDULED`, `AttemptLifecycleService` |
| Latency | Events, traces (Nexus tracing), metrics hooks | Step timestamps, OTLP derived |
| Dependency failures | Resilience boundary, external ops | `EXTERNAL_OPERATION_FAILED`, concurrency admission denials |
| Cancellation | Coordinator + events | `CANCELLATION_REQUESTED`, `CANCELLED` |
| Recovery history | Checkpoint revisions, recovery handoffs, admission | `RecoveryAdmissionDecision`, partial snapshots |
| Checkpoint state | Task/decision checkpoint stores | `RuntimeCheckpoint`, `DecisionCheckpointState` |
| Terminal truth | `ExecutionTerminalService` | Dominates resume narrative |
| Problems (derived) | Diagnostic store | Operator-facing pattern identity |
| Predictive signals | Prediction engine | `PredictiveRiskSignal` (advisory) |
| Self-heal / prevention | Workflow + action outcomes | Audit records in contracts |

### Informacje potrzebne (W6 outputs)

| Kategoria | Cel | Przykładowy artefakt |
|-----------|-----|----------------------|
| **Execution Intelligence** | Dlaczego sukces/porażka? | `ExecutionOutcomeExplanation` z `evidence_refs[]` |
| **Runtime Diagnostics** | Automatyczna analiza problemów | `RuntimeDiagnosticFinding` (non-canonical) |
| **Adaptive Policy Foundation** | Czy bezpiecznie dostosować zachowanie? | `AdaptivePolicyRecommendation` (recommend-only) |
| **Decision Intelligence** | Jak runtime decydował i jak ocenić? | `RuntimeDecisionAuditView`, post-hoc score |

### Kontekst budowy (read-only projection)

```text
RuntimeIntelligenceContextBuilder
  ├── ExecutionReconstruction (diagnostics) — read-only
  ├── Terminal + checkpoint pointers
  ├── RecoveryAdmission audit (W3)
  ├── Cancel / external-op timeline
  ├── Optional: DiagnosticReadService slice
  └── Optional: PredictiveInvestigation attachment
```

---

## ETAP 4 — Architecture principles (W6-A freeze)

| Principle | Rule |
|-----------|------|
| Zero god components | No `*Manager` intelligence owner; port + engine + adapters |
| Contract first | `intergrax/contracts/runtime_intelligence/*` before adapters |
| Fail-soft | Intelligence failure never fails execution |
| Single diagnostic authority | W6 findings ≠ Problems |
| Action via existing ports | No direct retry/recovery/cancel from analyzers |
| No global state | Composition-root wiring only |
| Quality bar (future code) | No `Any`, `type: ignore`, `getattr`/`setattr` shortcuts without ADR |

---

## ETAP 5 — Plugin architecture (target)

```text
RuntimeIntelligencePort
        |
        +-- LocalDeterministicAnalyzer
        +-- MLAnalyzer
        +-- ExternalServiceAnalyzer
```

Registry: ordered tuple at wiring time; failure containment per analyzer (pattern: `PredictionEngine`).

---

## ETAP 6 — Ownership matrix (przyszłe funkcje)

| Element | Owner |
|---------|--------|
| **Data collection** | Existing planes (events, checkpoint, terminal, diagnostics read ports) — **no new canonical store in W6-A** |
| **Context projection** | `runtime/runtime_intelligence/` (future) — read-only builder |
| **Analysis** | `RuntimeIntelligenceAnalyzerPort` plugins |
| **Orchestration** | `RuntimeIntelligenceAnalysisEngine` (composed, not global) |
| **Decision (runtime)** | Unchanged: governance, recovery admission, self-heal admission |
| **Action** | Unchanged: existing ports only; W6 recommends |
| **Audit** | Envelope + `analyzer_id`/`analyzer_version` + `evidence_refs` |
| **Failure** | W6 plane: degraded result; execution plane: unchanged |

---

## ETAP 7 — Risks (qualification)

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| R1 | Duplicate diagnostic authority | High | ADR §3.6 + Diagnostics R1 invariant |
| R2 | Hot-path coupling | High | Default off critical path; timeouts |
| R3 | Operator confusion (3 advisory layers) | Medium | Hub doc + investigation projections |
| R4 | External analyzer data exfiltration | High | Governance enablement; ref-only context |
| R5 | Overlap with `adaptive/` harness | Medium | Clear L4 vs execution-plane scope |
| R6 | Incomplete context (missing persistence) | Medium | `INSUFFICIENT_EVIDENCE` in envelope |

---

## ETAP 8 — Implementation phases (post W6-A)

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| **W6-B** | Contract freeze | `intergrax/contracts/runtime_intelligence/` + conformance tests |
| **W6-C** | Local analyzer + context builder | Deterministic analyzer adapter; no ML |
| **W6-D** | Decision intelligence correlation | Recovery/cancel/self-heal audit views |
| **W6-E** | Adaptive policy signals | `AdaptivePolicySignalPort` + governance bridge |
| **W6-F** | Operator investigation projection | Read model attached to diagnostic operator view (derived) |
| **W6-G** | Enterprise qualification | Matrix tests + deployment readiness |

**W6-A:** docs only — this file + ADR + hub.

---

## ETAP 9 — Qualification verdict

| Gate | W6-A |
|------|------|
| Inventory complete | **PASS** |
| Architecture map | **PASS** |
| ADR with ≥3 alternatives | **PASS** |
| Ownership matrix | **PASS** |
| Production code changed | **NO** (required) |
| New managers/schedulers | **NO** (required) |
| Tests added | **NO** (inventory task) |

**Recommendation:** Accept W6-A; proceed to W6-B contract design under [`ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md`](../architecture/ADR_ENTERPRISE_RUNTIME_INTELLIGENCE_ARCHITECTURE.md).

---

## Appendix — Pipeline placement (target)

```text
[Execution hot path]
  publish RuntimeEvent → bus → (handlers) → optional persistence
        |
        | (async / on-demand)
        v
RuntimeIntelligencePort.analyze(context)
        |
        v
Envelope → operator UI / API / optional diagnostic extension evidence
        |
        X (no direct) → Recovery / Cancel / Retry
        |
        v (governance only)
Existing admission & policy ports
```
