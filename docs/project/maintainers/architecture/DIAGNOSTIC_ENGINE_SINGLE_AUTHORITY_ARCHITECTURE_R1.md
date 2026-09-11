# Central Diagnostic Engine — Single Authority Architecture (R1)

**Task:** `DIAGNOSTIC-ENGINE-SINGLE-AUTHORITY-ARCHITECTURE-R1`

**Status:** Architecture frozen (no production implementation in this task)

**Repo HEAD at freeze:** `f056d92ea5d58bc4954ede94ab73f65e5cfd7401` (branch `development`; ancestry verified for `2491a724…`, `7a9838fa…`, `130c4715…`)

---

## 1. Executive invariant

```text
MANY SYSTEMS · MANY APPLICATIONS · MANY SCENARIOS · MANY EVIDENCE PRODUCERS
                    ↓
        ONE DIAGNOSTIC ENGINE (intergrax.runtime.diagnostics)
        ONE PROBLEM AUTHORITY (ProblemLifecycleEngine + canonical Problem store)
        ONE DIAGNOSTIC READ SURFACE (DiagnosticReadService)

DIAGNOSTIC_AUTHORITY_COUNT = 1
SECOND_DIAGNOSTIC_ENGINE_ALLOWED = NO
```

Applications and scenarios **may extend** typed evidence and bounded domain analysis. They **must never** instantiate a competing diagnostic engine, Problem lifecycle, or diagnostic read semantics.

**Developer rule (Definition of Done):**

```text
I DO NOT BUILD DIAGNOSTICS.
I PROVIDE TYPED EVIDENCE AND OPTIONAL DOMAIN ANALYSIS TO INTERGRAX CENTRAL DIAGNOSTICS.
```

---

## 2. Confirmed blocker (counterexample driving R1)

```text
root E1
└── child E2
    └── child E3
        └── child E4  → real execution failure
```

**As-built today:**

| Capability | State |
| ---------- | ----- |
| Durable execution lineage (`ExecutionLineagePersistence`, parent chain, segment topology) | **Present** — child admission via `ChildExecutionRunner` + lineage hook |
| Minted `ExecutionId` per child (E4) | **Present** — `mint_child_execution_id()` at admission |
| `CoordinationFailureCode.CHILD_EXECUTION_FAILED` / `ChildExecutionFailedError` | **Present** — orchestration/control-flow only |
| Guaranteed durable path: E4 failure + `ExecutionId=E4` + typed failure semantics → Central Diagnostics → Problem occurrence → `DiagnosticReadService` | **GAP** — not end-to-end guaranteed |

`ChildExecutionFailedError` is **not** canonical diagnostic evidence (ephemeral, not durable, not operator-reconstructable after restart, does not carry five-ID failure contract by itself).

`CHILD_EXECUTION_FAILED` proves **delegation failed**, not **which ExecutionId failed** unless explicit canonical linkage exists.

---

## 3. Authority model (frozen)

```text
Execution Runtime              → owns execution facts (identity, boundary, lineage admission)
Decision System              → owns decision facts and decision lifecycle evidence
Governance                     → owns authorization / HITL outcome facts
Subsystems / Applications      → own business/domain facts (via approved evidence SPI)
Observability / Evidence layer → records canonical evidence (append-first where contracted)
Central Diagnostic Engine      → interprets/correlates canonical evidence only
ProblemLifecycleEngine         → owns derived Problem lifecycle (mint/reconcile occurrences)
DiagnosticReadService          → owns canonical diagnostic read projection
```

Central Diagnostics **must not** invent execution facts, invent decision facts, invent application facts, or rewrite lineage.

```text
TELEMETRY PRODUCERS = MANY  (logs, metrics, traces, OTel, vendor backends, health checks)
DIAGNOSTIC AUTHORITY = ONE
```

Telemetry and vendor observability are **evidence inputs or derived projections**, not Problem truth or root-cause truth.

---

## 4. Evidence-first pipeline (frozen)

```text
FACT
  → CANONICAL EVIDENCE (immutable append where applicable)
  → CENTRAL DIAGNOSTIC ANALYSIS (deterministic core + bounded plugins)
  → FINDING / ASSESSMENT
  → PROBLEM (reconciled durable state)
  → DIAGNOSTIC READ MODEL
```

**Forbidden:**

```text
exception string → LLM guess → Problem
```

---

## 5. Decision — canonical execution failure evidence carrier

**Chosen: OPTION B (composition), with OPTION A as the execution-failure spine.**

| Layer | Role |
| ----- | ---- |
| **`RuntimeEvent`** | **Universal execution-scoped failure evidence carrier** for platform runtime paths. Carries five-ID identity (`tenant_id`, `task_id`, `run_id`, `attempt_id`, `execution_id`), `event_id`, typed `event_type` / `event_kind`, `phase`, `severity`, versioned `schema_version` (`runtime_event.v2`), bounded `payload` with explicit failure taxonomy (not raw exception object as identity). |
| **`PlatformCausalEvidence`** | **Supplementary typed causal edges** (transport→execution today; extended relation kinds in later tasks for tool/decision/integration/propagation). Does **not** replace `RuntimeEvent` for “E4 failed”. |
| **`PlatformFunctionalEvidence`** | **Domain / pipeline functional evidence** consumed by `FunctionalDiagnosticAnalyzer` — not execution terminal failure by itself. |
| **OPTION C (`DiagnosticEvidence` umbrella)** | **Rejected for R1** — would duplicate `RuntimeEvent` + existing causal/functional stores without clearing the E4 gap. |

**Rationale:** `RuntimeEvent` already models five-ID execution identity and rich failure `RuntimeEventType` values (`STEP_FAILED`, `TASK_FAILED`, `TOOL_FAILED`, `RUNTIME_HANDLER_FAILED`, …). The gap is **writer coverage and ordering** on child failure paths, not absence of a carrier type.

---

## 6. Five-ID target (frozen)

```text
TenantId · TaskId · RunId · AttemptId · ExecutionId
```

Rules:

- Do **not** overload `RunId`, `AttemptId`, or `event_id` as `ExecutionId`.
- **New writers (post–R2 rollout):** every canonical execution-scoped terminal/failure `RuntimeEvent` **must** include the active `ExecutionId` (including child E4).
- **Legacy:** events without `execution_id` remain valid at **lower precision** (`RUN_LEVEL` / `ATTEMPT_LEVEL`); diagnostics must not fabricate E7 from heuristics.

---

## 7. Failure evidence field contract (execution-scoped)

Diagnostics must be able to obtain (from canonical stores, not from coordination messages alone):

| Field | Source (target) |
| ----- | ---------------- |
| `tenant_id` | `RuntimeEvent` / scope |
| `task_id`, `run_id`, `attempt_id`, `execution_id` | `RuntimeEvent` envelope |
| Event/evidence identity | `event_id` |
| Failure kind/category | `event_type` + typed payload family (category/code), not exception type name as sole key |
| Component/source | `agent_id`, `step_id`, `node_id`, `ops_hint`, bounded payload |
| `observed_at` | `timestamp` |
| Causal relationship | `PlatformCausalEvidence` + future typed edges; optional `parent_event_id` is **not** causality |

Optional `parent_execution_id` in lineage is for **topology lookup**, not substitute for failure evidence on E4.

---

## 8. Child failure ordering and crash safety (frozen semantics)

```text
child lineage admission durable (ExecutionId E4)
  → child delegate runs under ExecutionBoundary
  → on failure: canonical failure RuntimeEvent persist attempt (execution_id=E4)
  → only then: failure propagates (exception / coordination outcome)
```

If failure evidence persistence **fails**:

| Policy | Semantics |
| ------ | --------- |
| **Execution truth** | Failure still propagates (business/control-flow) — diagnostics does not block execution |
| **Diagnostic truth** | Mark assessment **DEGRADED** / **PARTIAL**; `DiagnosticCompleteness` reflects missing failure evidence; never claim `PROVEN` failure boundary at E4 |
| **Evidence store outage** | Distinct state **UNAVAILABLE** vs “no failure” |

`ChildExecutionFailedError` may still surface to parents; it is **not** persisted diagnostic evidence.

---

## 9. Coordination vs execution (frozen)

```text
CoordinationFailureCode.CHILD_EXECUTION_FAILED  →  orchestration outcome
RuntimeEvent (execution_id=E4, failure kind)     →  execution evidence
```

Diagnostics correlates them when **explicit linkage** exists (shared five-ID scope, causal edge, or occurrence refs). Collapsing coordination code into execution failure identity is **forbidden**.

---

## 10. Failure boundary, cause, impact (frozen model)

New derived concepts for R2+ implementation (names frozen at architecture level):

| Concept | Meaning |
| ------- | ------- |
| **`FailureBoundary`** | Deepest **deterministically proven** failing execution/component (e.g. E4 SAP connector), not the root E1 |
| **`ProvenanceCauseKind`** | `PROVEN_CAUSE` · `PROBABLE_CAUSE` · `RELATED_EVIDENCE` · `UNKNOWN_BEYOND_BOUNDARY` |
| **Impact** | Ancestors/siblings affected by propagation (E1 user request failed) — **not** labeled as root cause |
| **Symptom** | Observable outcome without proven causal link |

Rules:

- `parent_execution_id` in lineage = **topology only**, not “parent caused by child”.
- Temporal order ≠ causal order; causal edges require typed evidence.
- Sibling isolation: E3 fails → boundary E3; E2/E4 unaffected unless evidence says otherwise.
- External boundary (E4→SAP HTTP 500): `LAST_PROVEN_BOUNDARY = SAP API`; provider-internal cause = `UNKNOWN_BEYOND_BOUNDARY` — no fabrication.

Maps to existing `DiagnosticCertainty` (`PROVEN`, `INSUFFICIENT_EVIDENCE`) with **extensions** in implementation for `SUPPORTED` / `INCONCLUSIVE` / `UNAVAILABLE` where evidence quality demands it.

**Precision levels (frozen):** `RUN_LEVEL` · `ATTEMPT_LEVEL` · `EXECUTION_LEVEL` · `EXTERNAL_BOUNDARY`.

Plugins **cannot** upgrade precision without canonical correlation evidence.

---

## 11. Diagnostic completeness and confidence (frozen, multi-dimensional)

Do **not** conflate into one boolean.

| Dimension | Examples |
| --------- | -------- |
| Evidence availability | store reachable vs `UNAVAILABLE` |
| Evidence coverage | failure event present vs missing |
| Lineage completeness | full tree vs truncated discovery |
| Causal completeness | edges proven vs unknown |
| Assessment completeness | all analyzers succeeded vs degraded |

**Confidence** derives from evidence quality only — not LLM self-confidence.

---

## 12. Security, redaction, immutability, versioning

- **Admission-time** redaction/validation for contributed evidence; do not rely on UI filtering.
- Payload families: **failure category**, **failure code**, **safe summary**, **internal detail**, **raw sensitive detail** — separate tiers; default persistence minimizes PII/secrets.
- Persisted evidence: **append-only / immutable** where contracted; Problems may evolve; **never rewrite** historical evidence because diagnosis changed.
- Every evidence/extension contract: **typed, frozen/immutable models, explicit `schema_version`**, tenant-scoped, `extra=forbid` style; unknown versions → typed unsupported handling.
- **Provider neutral** semantics in `intergrax.runtime.diagnostics` and evidence contracts — Mongo/Postgres/Kafka/OTel are adapters.

---

## 13. Platform-wide extension SPI (R1 freeze)

Extensions have **no diagnostic authority**. Core ports (`DiagnosticOrchestrator`, `ProblemLifecycleEngine`, `DiagnosticReadService`) are **not** implementable by applications.

| Port (frozen names) | Responsibility | Forbidden |
| ------------------- | -------------- | --------- |
| **`DiagnosticEvidenceContributor`** | Emit typed, versioned, tenant-scoped domain evidence (e.g. `payment_id`, workflow step) into canonical functional/evidence stores | Mint `ProblemId`, set root cause, override platform evidence, change completeness |
| **`DiagnosticAnalyzer`** (domain) | Map known domain codes to bounded **`DiagnosticFindingCandidate`** (or functional analysis results) | Direct `ProblemPersistence` / `reconcile` |
| **`DiagnosticSubjectResolver`** | Resolve optional subjects (integration op, tool invocation) **with** canonical execution identity when possible | Mint competing subject authority |
| **`DiagnosticTaxonomyContributor`** | Register **namespaced** failure/evidence kinds in registries — not a global mega-enum | Platform-wide enum of every future app failure |

**Registry:** single **`DiagnosticAnalyzerRegistry`** (and existing **`DiagnosticScopeDiscoveryProviderRegistry`**) under central bootstrap — deterministic ordering, conflict markers, no import-order precedence.

**Plugin contracts:** typed, bounded, deterministic where canonical, tenant-aware; **no** `dict[str, Any]` bags, reflection, or dynamic magic.

**Analyzer failure / timeout:** containment — faulty plugin → partial/degraded assessment only; cannot erase evidence or change execution outcome.

**No recursive diagnostics loop:** subsystem failure evidence uses existing isolation (does not re-enter orchestrator write path).

**Evidence kind namespaces (illustrative):** `execution.failure`, `decision.rejected`, `integration.failure`, `application.domain` — extensible via registry, not one giant central enum.

---

## 14. Adoption contracts (frozen)

| Actor | Must | Must not |
| ----- | ---- | -------- |
| **Application** | Use canonical execution runtime; emit typed evidence; optional analyzer/subject/taxonomy registration; read via `DiagnosticReadService` | `DiagnosticOrchestrator`, local Problem store, local diagnostic engine |
| **Scenario** | Supply domain facts via SPI only | Configure diagnostic storage, instantiate orchestrator, own `ProblemLifecycleEngine` |
| **Multi-agent / Agent distribution** | Emit/propagate typed facts | `MultiAgentDiagnosticEngine` or root-cause authority |
| **Decision system** | Own decision lifecycle evidence | Duplicate decision lifecycle inside diagnostics |
| **Tools / RAG / Connectors** | Emit execution-scoped + boundary-safe failure evidence | Connector-specific Problem DB or `RagDiagnosticEngine` |
| **Governance / HITL** | Own DENIED / REQUIRES_HUMAN / ALLOW truth | Diagnostics re-labeling DENIED as infra failure |
| **LLM** | Summarize, suggest investigation steps | Create canonical Problem, select root cause without deterministic evidence, upgrade confidence |

Canonical Problem path:

```text
DiagnosticOrchestrator → ProblemGroupingEngine → ProblemLifecycleEngine
```

---

## 15. Problem store and read model

- **Single logical authority:** `ProblemPersistence` + `ProblemOccurrencePersistence` (physical sharding allowed; **no** semantic duplication per app/scenario).
- Occurrences hold refs to reconstruct tenant, task/run, attempt, **proven failing execution**, subject — **no** duplicate execution tree inside Problem.
- **No duplicate lineage snapshot** in Problem store — read canonical `ExecutionLineage` via ports.
- `DiagnosticReadService` answers: WHAT/WHERE/WHEN/WHICH execution/WHY (proven vs unknown)/evidence/completeness — bounded reads, no tenant-wide replay for one incident.

---

## 16. As-built audit (code-backed summary)

| Area | Package / type | CURRENT | TARGET | GAP | NEXT TASK |
| ---- | -------------- | ------- | ------ | --- | --------- |
| Execution identity | `RuntimeEvent` (`runtime_event.v2`) | Five-ID on model; many writers | All failure paths emit E4 | Child/delegate failure may lack durable terminal event before propagate | R2 evidence writers |
| Child execution | `ChildExecutionRunner` | Lineage admission + boundary | Failure event persist before propagate | Ordering not architecturally enforced in runner | R2 |
| Lineage | `ExecutionLineagePersistence` | Durable tree | Read for diagnostics | Not causal proof | R3 correlation |
| Causal | `PlatformCausalEvidence` | Minimal `CausalRelationKind` | Rich edges for tool/decision/propagation | Sparse relation set | R4+ |
| Coordination | `ChildExecutionFailedError` | Control-flow | Link to E4 evidence | No ExecutionId in error | R2 + correlation |
| Reconstruction | `ExecutionReconstructor` | Bounded RuntimeEvent + causal reads | Failure boundary at ExecutionId | No explicit FailureBoundary type yet | R2/R3 |
| Assessment | `DiagnosticAssessmentBuilder`, `DiagnosticCertainty` | Lifecycle anomalies | Cause/boundary/impact split | Coarser certainty model | R3 |
| Grouping | `ProblemGroupingEngine` / deterministic strategy | Structural signatures | Include failure boundary in signature | May group without E-level precision | R3 |
| Lifecycle | `ProblemLifecycleEngine` | Canonical reconcile | Occurrence refs to E4 | Depends on evidence | R2 |
| Read | `DiagnosticReadService` | Central read | Full operator questions | Partial without E4 evidence | R2 |
| Functional / domain | `FunctionalDiagnosticAnalyzer`, `PlatformFunctionalEvidence` | Deterministic checks | Domain plugins via SPI formalization | App registration path incomplete | R5 |
| Scope discovery | `DiagnosticScopeDiscoveryProviderRegistry` | Bounded discovery | Central composition | — | R5 |
| Subjects | `DiagnosticSubjectKind` (EXECUTION, APPLICATION_INSTANCE) | Two kinds | Composed subjects (tool, integration) | Extend via resolver, not app Problem store | R5 |
| Scenario gates | `test_scenario_runtime_baseline` AST | Forbids orchestrator symbols in scenarios | Single authority | Extend to apps/production modules | R6 gates |
| Telemetry | HOS / export | Derived | Non-authority | Already aligned | — |

---

## 17. Forbidden ownership

| Assignment | Verdict |
| ---------- | ------- |
| Application → Problem lifecycle | **FORBIDDEN** |
| Scenario → `DiagnosticOrchestrator` | **FORBIDDEN** |
| Agent → root-cause authority | **FORBIDDEN** |
| Connector → Problem Store | **FORBIDDEN** |
| LLM → canonical finding | **FORBIDDEN** |
| Vendor observability → truth | **FORBIDDEN** |
| `ApplicationDiagnosticEngine` / `ScenarioDiagnosticEngine` / … | **FORBIDDEN** |
| `ApplicationProblemStore` / local `DiagnosticReadService` semantics | **FORBIDDEN** |

---

## 18. Ownership table

| Component | Owns |
| --------- | ---- |
| Execution Runtime | Execution facts, boundary, child admission |
| Execution Lineage | Topology persistence (parent chain, segment) |
| RuntimeEvent persistence | Canonical execution evidence spine |
| Causal Evidence persistence | Typed cross-boundary causal facts |
| Central Diagnostics (`intergrax.runtime.diagnostics`) | Interpretation, assessment, orchestration of analysis |
| Problem lifecycle | Problem identity, occurrences, reconciliation |
| Application plugins | Domain evidence + bounded analyzer candidates |
| Scenario plugins | Same SPI as apps — no extra authority |
| Observability export | Derived projections |
| LLM investigation | Non-canonical narrative |

---

## 19. Rollout (frozen dependencies)

```text
R1  Single-authority architecture (this document)
R2  DIAGNOSTIC-ENGINE-EXECUTION-FAILURE-EVIDENCE-R1 — E4 durable failure RuntimeEvent + diagnostic E2E
R3  Multi-agent failure localization (boundary, siblings, grouping)
R4  Decision → execution diagnostic correlation
R5  Application extension SPI (contributor, analyzer registry, gates)
R6  Cross-system enterprise qualification + composition hardening
```

No big-bang migration: legacy attempt-level evidence stays readable; new writers adopt stronger contract.

---

## 20. Qualification strategy (future tests)

| ID | Scenario |
| -- | -------- |
| E1 | Single child failure → exact child `ExecutionId` |
| E2 | Nested E4 failure → failure boundary E4 |
| E3 | Sibling E3 fail → E2/E4 unaffected |
| E4 | External API fail → boundary known, internal cause unknown |
| E5 | Legacy attempt-only failure → attempt-level precision only |
| E6 | Domain analyzer adds deterministic business interpretation |
| E7 | Domain analyzer unavailable → central engine degraded, still operational |
| E8 | Invalid plugin cannot create Problem directly |
| E9 | App/scenario cannot instantiate local diagnostic authority |
| E10 | LLM cannot alter Problem truth |

---

## 21. Architecture gates (R6 implementation)

Static composition checks (AST/import-aware, not class-name grep alone) ensuring `applications/`, `agents/`, `platform_proofs/scenarios/` do **not** construct outside approved central wiring:

- `DiagnosticOrchestrator`
- `ProblemLifecycleEngine`
- `ProblemPersistence` / `ProblemOccurrencePersistence`
- `ExecutionReconstructor`

Extend existing `test_scenario_runtime_baseline` pattern to production application modules.

**Central composition:** register analyzers/contributors via existing platform plugin/bootstrap — no per-scenario service locator, no uncontrolled global mutable registry.

---

## 22. Architecture decision record (explicit answers)

| Decision | Answer |
| -------- | ------ |
| `CANONICAL_DIAGNOSTIC_AUTHORITY` | `intergrax.runtime.diagnostics` (Central Diagnostics) |
| `CANONICAL_PROBLEM_AUTHORITY` | `ProblemLifecycleEngine` + canonical Problem persistence |
| `CANONICAL_READ_AUTHORITY` | `DiagnosticReadService` |
| `CANONICAL_EXECUTION_FAILURE_EVIDENCE` | **`RuntimeEvent`** (failure types) **+** supplementary **`PlatformCausalEvidence`** / functional evidence as composed model (**OPTION B**) |
| `EXECUTION_ID_REQUIRED_FOR_NEW_EXECUTION_EVENTS` | **YES** for new canonical writers after R2 rollout |
| `LEGACY_WITHOUT_EXECUTION_ID` | Honest lower precision; no fabrication |
| `APPLICATION_EXTENSION_SPI` | Contributor + Analyzer + optional SubjectResolver + TaxonomyContributor (registry-backed) |
| `SCENARIO_EXTENSION_SPI` | Same as application — facts/analyzers only |
| `AGENT_EXTENSION_SPI` | Emit facts only; no diagnostic engine |
| `DOMAIN_ANALYZER_AUTHORITY` | Bounded `DiagnosticFindingCandidate` / functional analysis only |
| `CAN_DOMAIN_ANALYZER_CREATE_PROBLEM` | **NO** |
| `FAILURE_BOUNDARY_MODEL` | `FailureBoundary` + proven cause kinds (§10) |
| `CAUSE_VS_IMPACT` | Separated (§10) |
| `DIAGNOSTIC_COMPLETENESS_MODEL` | Multi-dimensional (§11) |
| `DIAGNOSTIC_CONFIDENCE_MODEL` | Evidence-derived; extends current `DiagnosticCertainty` |
| `EXTERNAL_BOUNDARY_UNKNOWN_CAUSE` | `UNKNOWN_BEYOND_BOUNDARY` / last proven boundary |
| `PLUGIN_ORDERING` | Central registry, deterministic order + conflict semantics |
| `PLUGIN_FAILURE_ISOLATION` | Degraded/partial assessment; no evidence corruption |
| `EVIDENCE_REDACTION` | At canonical admission |
| `LLM_AUTHORITY` | **NON_CANONICAL** |
| `SINGLE_PROBLEM_STORE` | **YES** (logical) |
| `SECOND_DIAGNOSTIC_ENGINE_ALLOWED` | **NO** |
| `CHECKPOINT_FORENSIC_AUTHORITY` | Checkpoints are recovery state — **not** diagnostic truth (existing platform stance preserved) |
| `VENDOR_TELEMETRY_AUTHORITY` | **NO** |

---

## 23. Related documentation

- Operator/domain hub: [`docs/project/architecture/DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)
- Observability evidence: [`docs/project/architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)

---

## 24. Final architecture block

```text
STATUS: FROZEN (R1 ARCHITECTURE)

CANONICAL_DIAGNOSTIC_AUTHORITY:
  intergrax.runtime.diagnostics (Central Diagnostic Engine)

CANONICAL_PROBLEM_AUTHORITY:
  ProblemLifecycleEngine + ProblemPersistence / ProblemOccurrencePersistence

CANONICAL_DIAGNOSTIC_READ_AUTHORITY:
  DiagnosticReadService

SECOND_DIAGNOSTIC_ENGINE_ALLOWED:
  NO

MANY_EVIDENCE_PRODUCERS:
  YES (RuntimeEvent, CausalEvidence, FunctionalEvidence, domain contributors, telemetry adapters)
ONE_DIAGNOSTIC_AUTHORITY:
  YES

CANONICAL_EXECUTION_FAILURE_EVIDENCE:
  RuntimeEvent (typed execution failure) composed with PlatformCausalEvidence and functional evidence where applicable (OPTION B)

EXECUTION_ID_NEW_WRITER_REQUIREMENT:
  REQUIRED on all new canonical execution-scoped failure/terminal RuntimeEvents after R2 rollout

LEGACY_EXECUTION_ID_SEMANTICS:
  Attempt/run-level diagnosis only when execution_id absent; mark precision explicitly; no heuristic upgrade

APPLICATIONS_CAN_DEFINE_EVIDENCE:
  YES (typed contributor → canonical stores)
APPLICATIONS_CAN_DEFINE_ANALYZERS:
  YES (registry, bounded output)
APPLICATIONS_CAN_CREATE_PROBLEMS:
  NO
APPLICATIONS_CAN_OWN_DIAGNOSTIC_STORE:
  NO (semantic authority)

SCENARIOS_CAN_CREATE_DIAGNOSTIC_ENGINE:
  NO

AGENTS_CAN_CREATE_DIAGNOSTIC_ENGINE:
  NO

FAILURE_BOUNDARY:
  FailureBoundary at deepest proven execution/component (e.g. E4)

CAUSE_AND_IMPACT_SEPARATED:
  YES

EXECUTION_LINEAGE_IS_CAUSALITY:
  NO (topology only; causality requires typed evidence)

EXTERNAL_UNKNOWN_CAUSE:
  LAST_PROVEN_BOUNDARY + UNKNOWN_BEYOND_BOUNDARY

DIAGNOSTIC_COMPLETENESS:
  Multi-dimensional (availability, coverage, lineage, causal, assessment)

DIAGNOSTIC_CONFIDENCE:
  Evidence-derived (PROVEN / SUPPORTED / INCONCLUSIVE / UNAVAILABLE — implementation aligns with DiagnosticCertainty)

LLM_CANONICAL_AUTHORITY:
  NO

PROBLEM_STORE_AUTHORITY_COUNT:
  1 (logical)

CHECKPOINT_FORENSIC_AUTHORITY:
  NO

VENDOR_TELEMETRY_AUTHORITY:
  NO

PLUGINABLE:
  YES (bounded SPI)
MODULAR:
  YES
PROVIDER_NEUTRAL:
  YES
TENANT_SAFE:
  YES
BOUNDED:
  YES (reads, analyzer budgets, payload limits)

NEXT_TASK:
  DIAGNOSTIC-ENGINE-EXECUTION-FAILURE-EVIDENCE-R1
  (child E4 fails → durable failure evidence references E4 → central Problem occurrence → DiagnosticReadService)
```
