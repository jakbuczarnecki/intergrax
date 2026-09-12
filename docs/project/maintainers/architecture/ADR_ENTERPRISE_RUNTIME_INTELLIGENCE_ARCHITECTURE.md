# ADR-ENTERPRISE-RUNTIME-INTELLIGENCE: Enterprise Runtime Intelligence Architecture (W6)

| Field | Value |
|-------|-------|
| **Status** | **Proposed** — architecture + qualification only (W6-A); no production implementation |
| **Date** | 2026-09-12 |
| **Baseline** | W6-A inventory on `development` @ `6b51617752c3fda544e1b4d2a29da5ba21c9ded7` |
| **Related** | W1–W5 execution scale & resilience qualification · [`ENTERPRISE_RUNTIME_INTELLIGENCE_W6_A_QUALIFICATION.md`](../qualification/ENTERPRISE_RUNTIME_INTELLIGENCE_W6_A_QUALIFICATION.md) · [`ENTERPRISE_RUNTIME_INTELLIGENCE.md`](../../architecture/ENTERPRISE_RUNTIME_INTELLIGENCE.md) · [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) · [`ADR-PREDICTIVE-LAYER-AS-DIAGNOSTIC-CONSUMER.md`](ADR/ADR-PREDICTIVE-LAYER-AS-DIAGNOSTIC-CONSUMER.md) |
| **Planned contract modules (W6-B+)** | `intergrax/contracts/runtime_intelligence/` (names frozen here; **no code in W6-A**) |

---

## 1. Context

Intergrax has matured **execution reliability (W1)**, **scale & resilience (W2)**, **checkpoint & recovery (W3)**, **cancellation & external operations (W4)**, and **observability export (W5)**. Operators and autonomous workloads increasingly need answers that span those planes without re-implementing ad hoc analytics in each application:

- **Why** did this run succeed or fail (causal narrative across attempts, dependencies, recovery, and terminal outcome)?
- **What** automated diagnosis is safe to surface during execution (without replacing canonical diagnostic authority)?
- **When** may the platform **recommend or apply** adaptive execution policy (retry budgets, admission, routing) under governance?
- **How** were runtime **decisions** (resume, cancel, reconcile, self-heal) taken and how should they be **scored** after the fact?

Today, facts exist in fragmented owners: `RuntimeEvent` spine, checkpoint stores, terminal records, diagnostic Problems, predictive signals, self-healing workflows, adaptive harness learning, and ERL reconciliation. There is **no single, contract-first, plugin-extensible intelligence plane** that **reads** those facts and emits **bounded, versioned intelligence artifacts** without becoming a god component or a second diagnostic engine.

**W6 (Enterprise Runtime Intelligence)** defines that plane: **consumer of canonical execution facts**, **advisory by default**, **action only through existing ports** (policy admission, recovery admission, governance bridges).

---

## 2. Current limitations (what the platform already has)

| Capability | Owner | Role relative to W6 |
|------------|-------|---------------------|
| Execution lifecycle & identity | `intergrax/runtime/execution/` (`ExecutionRuntime`, boundary, lineage) | **Source of truth** for run/attempt/task IDs and admission hooks |
| Graph orchestration | `intergrax/runtime/nexus/`, `execution/orchestration.py`, `host_task.py` | Emits step/tool/plan events; applies checkpoints |
| Retry & provider budgets | `intergrax/runtime/resilience/` + `intergrax/contracts/resilience_policy.py` | Policy resolution; not holistic “why failed” narrative |
| Recovery & checkpoint | `long_running/`, `execution/decision_recovery.py`, `resilience/*_recovery_handoff.py` | Resume truth; `RecoveryAdmissionPort` (W3-C) gates recovery **starts** |
| Cancellation | `cancellation/coordinator.py`, `RuntimeEventType` cancel family | Cooperative cancel; no intelligence scoring |
| External operations | `external_operations/` + ERL | UNKNOWN/reconcile; diagnostic evidence contributors |
| Events & export | `events/`, `observability/event_delivery/` | Canonical evidence + W5 OTLP pipeline |
| Central diagnostics | `runtime/diagnostics/` | **Single Problem authority** — W6 must not duplicate |
| Predictive layer | `runtime/prediction/` | Advisory risk signals; consumer of diagnostic evidence (ADR-PREDICTIVE) |
| Preventive / self-healing | `prevention/`, `self_healing/` | Action orchestration with governance; not unified execution analytics |
| Adaptive harness | `runtime/adaptive/` | Profile/skill learning (L4 harness); separate product surface from execution-plane intelligence |
| Replay / evidence | `replay/`, `execution_evidence/` | Reconstruction inputs |

**Frozen separations (must not break):**

```text
checkpoint ≠ lineage ≠ evidence ≠ terminal ≠ decision finalization ≠ Problem
diagnostic authority = 1 (Central Diagnostics)
observability export = derived projection (W5)
runtime intelligence = advisory read/analyze plane (W6) — not execution truth
```

**Gaps (W6 motivation):**

1. No **contract-first** `RuntimeIntelligencePort` for plugin analyzers (local / ML / external service).
2. No **unified execution intelligence read model** correlating attempt timeline, recovery admissions, cancellation, checkpoint generation, and terminal outcome.
3. No **decision intelligence** artifact tying governance/self-heal/recovery **decisions** to measurable outcomes (without owning those decisions).
4. **Adaptive policy foundation** is split (`adaptive/`, `resilience/policy_resolver.py`, runtime policy admission) without a shared “safe to adapt?” signal surface.
5. Risk of **silent duplication** if applications embed “mini intelligence” beside Diagnostics or Prediction.

---

## 3. Decision — target architecture

### 3.1 Position in the platform

```text
Execution Runtime (W1) ──facts──► RuntimeEvent / lineage / checkpoint / terminal
        │                                    │
        │                                    ├──► Central Diagnostics (authority)
        │                                    ├──► Observability export (W5, derived)
        │                                    └──► Runtime Intelligence Plane (W6, derived)
        │                                              │
        │                                              ├── ExecutionIntelligence (post-hoc / near-line)
        │                                              ├── RuntimeDiagnostics (bounded auto-analysis)
        │                                              ├── DecisionIntelligence (audit/score)
        │                                              └── AdaptivePolicySignals (recommend-only)
        ▼
Existing action ports only (no new god component):
  RecoveryAdmissionPort · ExecutionCapacityAdmissionPort · DependencyConcurrencyAdmissionPort
  RuntimeExecutionPolicyAdmission · Governance bridges · Self-healing admission gates
```

### 3.2 Zero god components

**Forbidden (without a future ADR that explicitly retires this list):**

- `RuntimeIntelligenceManager`, `AIManager`, `DecisionManager`, `UniversalAnalyzer`
- Global singleton registries for analyzers
- Background “intelligence schedulers” inside W6 core

**Required shape:**

```text
RuntimeIntelligencePort (contract)
        │
        ├── RuntimeIntelligenceAnalyzerPort (plugin SPI)
        │         ├── LocalDeterministicAnalyzer (adapter)
        │         ├── MLAnalyzer (adapter)
        │         └── ExternalServiceAnalyzer (adapter)
        │
        ├── RuntimeIntelligenceContextBuilder (projection from read-only facts)
        └── RuntimeIntelligenceResultEnvelope (versioned, auditable)
```

Orchestration of analyzer **ordering** and **failure containment** uses a **small engine** (e.g. `RuntimeIntelligenceAnalysisEngine`) composed at the **application wiring root** — same pattern as `PredictiveAnalyzerRegistry` + `PredictionEngine`, not a platform-wide manager.

### 3.3 Intelligence categories (semantic, not separate engines)

| Category | Question | Output (examples) | Authority |
|----------|----------|-------------------|-----------|
| **Execution Intelligence** | Why success/failure? | `ExecutionOutcomeExplanation`, timeline correlation | **Advisory**; cites `evidence_refs` only |
| **Runtime Diagnostics** | What automated finding for operators? | `RuntimeDiagnosticFinding` (bounded severity) | **Non-canonical**; may **suggest** Diagnostic extension evidence, never mint Problem |
| **Decision Intelligence** | How was a runtime decision made/scored? | `RuntimeDecisionAuditView`, outcome score | **Audit**; reads decision/recovery/cancel records |
| **Adaptive Policy Foundation** | Safe to change policy? | `AdaptivePolicyRecommendation` | **Recommend-only** until governance approves via existing adaptive/ policy ports |

### 3.4 Contract-first (W6-B scope — design only in W6-A)

Planned package: `intergrax/contracts/runtime_intelligence/`

| Contract | Owner lifecycle | Failure model | Versioning |
|----------|-----------------|---------------|------------|
| `RuntimeIntelligencePort` | Per wiring root; `analyze(context) → Result` | Analyzer failures isolated; `PLUGIN_UNAVAILABLE` per analyzer | `schema_version` on envelope |
| `RuntimeIntelligenceAnalyzerPort` | Plugin registration at compose time | Timeout → degraded partial result | `analyzer_id` + `analyzer_version` |
| `RuntimeIntelligenceContext` | Immutable snapshot built from read ports | Missing facts → explicit `INSUFFICIENT_EVIDENCE` | Context schema semver |
| `RuntimeIntelligenceEvidenceRef` | Pointer to event/checkpoint/terminal ids | Broken ref → omitted + warning in envelope | Stable ref kinds enum |
| `AdaptivePolicySignalPort` (optional W6-C) | Emits signals only | Never throws into execution hot path | Independent minor version |

Each contract documents: **ownership** (runtime intelligence plane), **lifecycle** (request-scoped analyze), **failure** (fail-soft on hot path), **versioning** (envelope + analyzer metadata).

### 3.5 Plugin architecture

```text
                    RuntimeIntelligencePort
                              |
              +---------------+---------------+
              |               |               |
     LocalDeterministic   MLModelAdapter   ExternalHTTPAdapter
        Analyzer            (batch)           (signed requests)
```

Selection via **composed registry** at wiring time (tuple ordering), **not** `if analyzer_type == X` in execution hot path.

### 3.6 Boundaries with W1–W5 and Diagnostics

| Plane | W6 may | W6 must not |
|-------|--------|-------------|
| W1 execution | Read identity, phase, failure evidence | Mutate lifecycle or mint IDs |
| W2 admission | Recommend load signals | Bypass `DependencyConcurrencyAdmissionPort` |
| W3 recovery | Correlate recovery admissions with outcomes | Start recovery without `RecoveryAdmissionPort` |
| W4 cancel / external ops | Explain cancel/reconcile timeline | Cancel or reconcile directly |
| W5 observability | Consume same facts as export | Own export sinks or OTLP transport |
| Diagnostics | Read Problems/read models as **inputs** | Write Problems or compete with `ProblemLifecycleEngine` |
| Prediction | Share analyzer plugin pattern | Become predictive authority |

---

## 4. Alternatives

### Alternative A — Central Intelligence Manager

Single class owns collection, analysis, ML, and policy changes.

| Pros | Cons |
|------|------|
| One place to find logic | Violates zero god component; untestable; blocks enterprise plugins |
| Fast initial demo | Breaks W3/W5 separation; high operational risk |

**Rejected.**

### Alternative B — Event-driven intelligence plane (async bus consumer)

Dedicated consumer subscribes to all `RuntimeEvent`s and maintains intelligence state asynchronously.

| Pros | Cons |
|------|------|
| Decouples from execution hot path | Second moving system; lag; duplicate projection of diagnostics |
| Natural for ML features | Requires durable intelligence store + compaction (new truth risk) |

**Partially accepted** only as **optional adapter** behind `RuntimeIntelligencePort` (near-line mode), not as canonical store. Default W6: **request-scoped analyze** from existing durable facts.

### Alternative C — Embedded runtime intelligence (inline in Nexus loop)

Every step invokes analyzers inside `GraphExecutor`.

| Pros | Cons |
|------|------|
| Lowest latency | Couples intelligence to orchestration; failure modes threaten execution |
| No new plane | Violates failure isolation; complicates tier boundaries |

**Rejected** for default path; optional **explicit hook** (admission-style) may call port with strict timeout in a later ADR.

### Alternative D — Extend Central Diagnostics to own all intelligence (chosen baseline comparison)

Fold execution explanations and adaptive signals into `intergrax.runtime.diagnostics`.

| Pros | Cons |
|------|------|
| Single operator surface | Overloads diagnostic authority; mixes deterministic Problems with ML |
| Less new code | Violates ADR-PREDICTIVE and single-authority scope |

**Rejected.** W6 **feeds** diagnostics via optional extension evidence; Problems remain diagnostic-owned.

---

## 5. Consequences

### Positive

- Clear **advisory plane** for enterprise “why / what / safe-to-adapt” without touching W1–W5 contracts.
- **Plugin extensibility** aligned with prediction and diagnostic extension SPIs.
- Enables **implementation plan** (W6-B contracts → W6-C local analyzer → W6-D decision correlation → W6-E adaptive signals).

### Negative

- Additional **read-model projection** work to avoid N+1 store queries.
- Operators must learn **three advisory layers**: Diagnostics (Problems), Prediction (risk), Runtime Intelligence (execution narrative) — mitigated via unified investigation UI projections (future).

### Neutral

- Reuses existing **governance bridges** for any action; W6 does not shorten approval paths.

---

## 6. Risks

| Risk | Type | Mitigation |
|------|------|------------|
| Second diagnostic authority | Technical / operational | Frozen boundary; Problems only via Diagnostics; W6 findings non-canonical |
| Intelligence on hot path blocks execution | Technical | Default async/off critical path; strict timeouts; fail-soft envelope |
| Analyzer supply-chain (external ML) | Security | Signed adapters; no secrets in envelopes; governance on enablement |
| Store sprawl (intelligence history) | Operational | W6-A: no new durable store; optional history only via existing predictive/history patterns + ADR |
| Confusion with `runtime/adaptive/` harness learning | Product | Document L4 harness vs execution-plane intelligence in hub doc |
| Recovery storm mis-attribution | Operational | Correlate with `RecoveryAdmissionPort` decisions in context builder |

---

## 7. Implementation gate

No production code until:

1. W6-A qualification **Accepted**
2. W6-B contracts merged with versioning + conformance tests
3. Explicit wiring in composition root (no global singleton)

**W6-A deliverable:** this ADR + qualification doc + architecture hub only.
