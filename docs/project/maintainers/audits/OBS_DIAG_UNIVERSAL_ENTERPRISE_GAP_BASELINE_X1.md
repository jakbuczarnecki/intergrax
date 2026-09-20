# OBS-DIAG-X1 — Universal Enterprise Gap Baseline

> **Maintainer audit evidence — not architecture SSOT.**
> **CURRENT AUTHORITY:** [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) · [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)
>
> **Historical freeze:** This X1 artifact records the gap state at the audited SHA below.
> Do **not** rewrite rows as if later closures never existed. OBS-DIAG-X2 was closed later —
> see [`OBS_DIAG_DIAGNOSTIC_COMPOSITION_PLUGINABILITY_X2.md`](OBS_DIAG_DIAGNOSTIC_COMPOSITION_PLUGINABILITY_X2.md).
> Composition replaceability remains **PARTIAL** in this document’s matrices by design (X1 truth).

| Field | Value |
| ----- | ----- |
| **Program** | OBS-DIAG-X1 |
| **Audited code SHA** | `19307f1e4c3bfdcdc9ea675f4909383ab1d70878` |
| **Audit start SHA** | `8ada8d72dd3d78f89048aaef177488f255fb3a64` (= `origin/development` at session start) |
| **Branch** | `development` (HEAD at docs reconcile; may be ahead of `origin/development`) |
| **Clean-gate historical pin** | `4633a7ab9b24194b31b525b1908201ec5f7c51e7` (prior clean-gate; superseded for current truth) |
| **Production code changes in X1** | NONE |
| **Test changes in X1** | NONE |

**Verdict framing (architecture vs universal spine):**

```text
CORE ARCHITECTURE = STRONG / ENTERPRISE
FULL UNIVERSAL OBS + DIAG SPINE = NOT YET FULLY CLOSED
```

---

## Status taxonomy (mandatory)

Use only: `PROVEN` · `IMPLEMENTED_NOT_FULLY_PROVEN` · `PARTIAL` · `NOT_PROVEN` · `NOT_APPLICABLE` · `PLANNED` · `OPEN`

## Proof-level taxonomy

| Level | Meaning |
| ----- | ------- |
| **P1** | Unit / contract |
| **P2** | Composition |
| **P3** | In-process end-to-end through real platform spine |
| **P4** | External / provider / process-boundary proof |

**Mock rule:** A test may use a deterministic LLM test double when LLM is not the qualified boundary. It must **not** mock transport, persistence, worker boundary, diagnostic persistence, or vendor endpoint when the proof declares those as real qualification targets.

---

## A. Current-state capability matrix

| Capability | Architecture | Implementation | Proof | Current status | Source |
| ---------- | ------------ | -------------- | ----- | -------------- | ------ |
| RuntimeEvent canonical evidence | A4 | I4 | P1–P3 gates | **PROVEN** | HOS + OBS-COVERAGE-1 P1 |
| five-ID identity | A4 | I4 | contract + writers | **PROVEN** | `RuntimeEvent` contract |
| RuntimeEvent persistence | A4 | I4 | SQLite default + stores | **PROVEN** | `RuntimeEventPersistence` |
| EventDeliveryBufferPort | A4 | I4 | P1B-R3 | **PROVEN** | ADR-OBS-005 |
| bounded event delivery | A4 | I4 | P1B-R3 | **PROVEN** | Plane B |
| causal evidence | A4 | I4 | contract + paths | **PROVEN** | CausalEvidencePersistence |
| functional evidence | A4 | I4 | contract + Mongo D1-R1 durability | **IMPLEMENTED_NOT_FULLY_PROVEN** | scale **NOT_PROVEN** |
| execution reconstruction | A4 | I4 | OBS-RECONSTRUCTION-1 | **PROVEN** | `ExecutionReconstructionReader` |
| execution lineage read | A4 | I4 | lineage ports | **PROVEN** | Execution lineage |
| historical reconstruction | A4 | I4 | ASOF + BITEMP | **PROVEN** | `HistoricalReconstructionService` |
| central diagnostics | A4 | I4 | HARDEN M1–M24 | **PROVEN** | `intergrax.runtime.diagnostics` |
| Problem grouping | A4 | I4 | strategy registry | **PROVEN** | `ProblemGroupingEngine` |
| Problem lifecycle | A4 | I4 | HARDEN | **PROVEN** | `ProblemLifecycleEngine` |
| Problem persistence | A4 | I4 | DocumentStore wiring | **PROVEN** | `ProblemPersistence` |
| Problem occurrence persistence | A4 | I4 | DocumentStore wiring | **PROVEN** | occurrence port |
| DiagnosticReadService | A4 | I4 | read suites | **PROVEN** (contract) | shared read wiring |
| terminal diagnostics | A4 | I4 | production E2E | **PROVEN** | terminal trigger path |
| async/background execution | A4 | I4 | universal spine async | **PROVEN** (P3 in-process) | `test_obs_universal_spine_async_e2e.py` |
| scenario runtime adoption | A4 | I3–I4 | architecture gate + partial E2E | **PARTIAL** | 4 initialized; E2E not universal |
| PRODUCT host adoption | A4 | I4 write path | factory gates | **PARTIAL** | write NATIVE; read varies |
| HITL restart/resume | A4 | I4 mechanisms | dedicated P3 | **PARTIAL** | see § HITL |
| cross-process topology | A4 | I4 | DG-005 | **PROVEN** (scoped) | see § DG-005 |
| Kafka ingress | A4 | I4 transport | transport P4 separate | **IMPLEMENTED_NOT_FULLY_PROVEN** | full spine **NOT_PROVEN** |
| OTLP export | A4 | I3–I4 | DIAG-FINAL OTLP slice | **PARTIAL** | vendor hardening open |
| Mongo Problem persistence | A4 | I4 | P4 Mongo path | **PROVEN** (path slice) | DIAG platform P4 |
| operator read exposure | A4 | I4 contract | host HTTP uneven | **PARTIAL** | CORE PROVEN / host PARTIAL |
| vendor integrations | A4 | I2–I3 adapters | OTLP slice only | **PARTIAL** | see vendor matrix |
| OECP | documented | not shipped | none | **PLANNED** / **OPEN** | eval control plane |

---

## B. Ownership matrix

| Concern | Single semantic owner | Competing owner found? |
| ------- | --------------------- | ---------------------- |
| execution lifecycle | Execution | **NO** |
| execution identity | Execution | **NO** |
| execution tree | Execution | **NO** |
| RuntimeEvent evidence | Observability | **NO** |
| evidence persistence | Observability / Evidence Plane | **NO** |
| factual reconstruction | Shared Evidence Plane (`ExecutionReconstructionReader`) | **NO** |
| diagnostic interpretation | Central Diagnostics | **NO** |
| Problem grouping validation | Central Diagnostics | **NO** |
| Problem lifecycle | Central Diagnostics | **NO** |
| Problem persistence contract | Diagnostics | **NO** |
| vendor telemetry | Derived integration adapter | **NO** |
| operator dashboards | Projection / read layer | **NO** |

**Duplication audit (production `intergrax/` + `applications/`):**
`DiagnosticOrchestrator(` / `ProblemLifecycleEngine(` / `attach_terminal_diagnostic_trigger(` resolve to shared composition in `diagnostic_runtime_wiring.py` (canonical). `ExecutionReconstructor(` / `wire_problem_persistence(` appear at composition roots and legitimate helpers — **no duplicate semantic authority** found. `RuntimeEventBus(` instances are runtime-scoped, not a second evidence SSOT.

---

## C. Contract / pluginability matrix (X2 input)

| Mechanism | Contract exists? | Engine injection exists? | Host composition injection exists? | External provider configurable? | Default hard-wired at composition root? | Status |
| --------- | ---------------- | ------------------------ | ---------------------------------- | ------------------------------- | --------------------------------------- | ------ |
| ProblemPersistence | YES | YES (via lifecycle engine) | YES (`wire_problem_persistence`) | via DocumentStore / provider profile | DocumentStore default | **PARTIAL** host replaceability |
| ProblemOccurrencePersistence | YES | YES | YES (`wire_problem_occurrence_persistence`) | via DocumentStore | DocumentStore default | **PARTIAL** |
| ExecutionReconstructionReader | YES | YES (orchestrator ctor) | YES but default class fixed | contract-conformant reader | `ExecutionReconstructor()` hard-wired | **PARTIAL** |
| CausalEvidencePersistence | YES | YES (reconstructor) | YES (`wire_causal_evidence_persistence`) | DocumentStore / memory | DocumentStore default | **PARTIAL** |
| ProblemGroupingStrategy | YES | YES (registry) | YES (`registry.register`) | register alternate strategy | Deterministic default registered | **PROVEN** seam |
| DiagnosticAssessmentBuilder | class/module | YES (orchestrator ctor) | **NO** public host override seam | N/A | `DiagnosticAssessmentBuilder()` hard-wired | **PARTIAL** |
| LifecycleAnomalyAnalyzer | class/module | YES (orchestrator ctor) | **NO** public host override seam | N/A | `LifecycleAnomalyAnalyzer()` hard-wired | **PARTIAL** |

```text
ENGINE CONTRACT PLUGINABILITY = PROVEN
STANDARD HOST COMPOSITION REPLACEABILITY = PARTIAL
```

Do **not** label the subsystem `NOT_PLUGINABLE`.

---

## D. Adoption matrix (current discovery @ audited SHA)

### PRODUCT hosts (write-path NATIVE = 4)

| Surface | Role | Diagnostics write | Operator HTTP read | Status |
| ------- | ---- | ----------------- | ------------------ | ------ |
| `governed_contractor_application` | PRODUCT | Yes | Yes (dashboard wiring) | **NATIVE** |
| `legal_application` | PRODUCT | Yes | Write only (no HTTP read routes) | **NATIVE** write / **PARTIAL** read |
| `dispute_sim_application` | PRODUCT | Yes | Write only | **NATIVE** write / **PARTIAL** read |
| `local_workspace_application` | PRODUCT reference | Yes | Optional / uneven | **NATIVE** write / **PARTIAL** read |
| `research_application` | PRODUCT prototype | Yes | Write only / DEV | **CONDITIONAL** |
| LKW / background worker | PRODUCT worker | Yes (shared harness) | No separate read API | **NATIVE** write |

```text
CORE READ CONTRACT = PROVEN
UNIVERSAL HOST EXPOSURE = PARTIAL
```

### Initialized scenarios (`discover_initialized_scenario_slugs` = **4**)

| Slug | Lifecycle (SCENARIO_SPEC) |
| ---- | ------------------------- |
| `ai_incident_investigation` | `EXECUTABLE` |
| `indirect_prompt_injection` | `EXECUTABLE` |
| `enterprise_payment_uncertainty_recovery` | `IMPLEMENTATION_INITIALIZED` |
| `verified_product_identification` | `IMPLEMENTATION_INITIALIZED` |

Design-only packages = **NOT_APPLICABLE** until `IMPLEMENTATION_INITIALIZED`.

Historical qualification text saying “1 initialized scenario” is a **snapshot**, not current truth.

---

## E. E2E proof matrix

| Flow | P-level | Real external boundary | Mocked qualified boundary? | Status |
| ---- | ------: | ---------------------: | -------------------------: | ------ |
| HTTP → DIAG (governed contractor) | P3/P4 slice | HTTP host + persistence | LLM may be double | **PROVEN** (host slice) |
| async worker → DIAG | P3 | in-process worker ingress | no transport mock required | **PROVEN** |
| Kafka → worker → execution → DIAG | — | would require Kafka + worker process | N/A | **NOT_PROVEN** as single P4 spine |
| Kafka transport alone | P4 | Kafka broker | no | **PROVEN** separately |
| Mongo Problem persistence | P4 | Mongo | no | **PROVEN** (path slice) |
| OTLP export | P4 | Docker OTLP collector | no | **PARTIAL** (slice; not full vendor matrix) |
| HITL restart → DIAG | P3 | durable stores; in-process rebuild | FakeLLMAdapter (LLM not qualified) | **PARTIAL** |
| scenario → DIAG | P2/P3 | lab DocumentStore | varies | **PARTIAL** (not all 4 E2E-qualified) |
| DG-005 cross-process evidence | P4 process | separate OS processes + shared SQLite file | no bus sharing | **PROVEN** (scoped) |

### Kafka classification (do not merge proofs)

```text
Kafka transport P4: PROVEN separately
in-process async OBS→DIAG spine P3: PROVEN
single external Kafka→worker→execution→diagnostics P4 spine: NOT_PROVEN
```

### DG-005 — reconciled claim

**PROVEN:** process-isolated writer / reader / diagnostics over shared durable `EvidencePersistencePort` (provider-neutral harness; currently qualified backend = `sqlite-file`); reconstruction/diagnostics do not share writer `RuntimeEventBus` history.

**Non-claims:** multi-region, network partition tolerance, HA failover, cross-DC consistency.

Proof: `tests/unit/runtime/architecture/test_obs_dg005_distributed_topology_qualification.py` + `testing_support/obs_distributed_topology/`.

### HITL / restart — reconciled claim

**PROVEN (P3):** durable checkpoint round-trip across store reinstantiation; GR-5 pause/approve/resume spine; runtime rebuild resume → terminal `TASK_COMPLETED` + diagnostics read path clean (`test_obs_universal_spine_hitl_restart_e2e.py`, `hitl_restart_harness.py`).

**NOT_PROVEN:** real independent OS process crash/restart of full host; real external HITL service; Kafka-coupled HITL spine.

---

## F. Zero-bypass matrix

| Surface | Proof | Status |
| ------- | ----- | ------ |
| application factory composition | adoption / architecture gates | **PROVEN** (PRODUCT BYPASS = 0 on factory composition) |
| request runtime path (all entry points) | no universal entry-path AST/runtime gate set | **NOT_PROVEN** |
| scenario | initialized architecture gate | **PROVEN** for initialized packages; design-only N/A |
| worker | shared harness wiring proofs | **PARTIAL** (LKW NATIVE write; not every worker topology) |
| local DiagnosticOrchestrator | production search → shared wiring only | **PROVEN** (no competing product orchestrator) |
| direct NexusLoop as root authority | architecture gates + docs | **PROVEN** as forbidden pattern on qualified surfaces; universal runtime scan **NOT_PROVEN** |

Do **not** claim `GLOBAL BYPASS = 0` for every runtime entry path.

---

## G. Vendor matrix (OBS)

| Vendor / sink | Contract exists | Adapter exists | Real proof exists | Failure/recovery proof | Production hardening |
| ------------- | -------------- | -------------- | ----------------- | ---------------------- | -------------------- |
| OTLP | YES | YES | P4 collector slice | collector-down truth retained | **PARTIAL** |
| JSONL / journal snapshot | YES (export path) | YES | unit/export | limited | **PARTIAL** |
| Datadog | YES | YES (provider package) | **NOT_PROVEN** live | **NOT_PROVEN** | **OPEN** |
| Sentry | YES | YES | **NOT_PROVEN** live enterprise | **NOT_PROVEN** | **OPEN** |
| Langfuse | YES | YES | **NOT_PROVEN** live | **NOT_PROVEN** | **OPEN** |
| Phoenix | YES | YES | **NOT_PROVEN** live | **NOT_PROVEN** | **OPEN** |

```text
OECP — PLANNED / OPEN
```

---

## H. Enterprise gap table

| ID | Gap | Current status | Why not closed | Priority | Required closure task |
| -- | --- | -------------- | -------------- | -------- | --------------------- |
| X2 | Diagnostic composition replaceability | **PARTIAL** | Host composition hard-wires several engines | **P0** | OBS-DIAG-X2 |
| X3 | Global entry-path zero-bypass proof | **NOT_PROVEN** | Factory gates ≠ every runtime entry | **P0** | OBS-DIAG-X3 |
| X4 | External Kafka full spine E2E | **NOT_PROVEN** | Transport + async proofs are separate | **P0** | OBS-DIAG-X4 |
| X5 | HITL full restart qualification | **PARTIAL** | Missing real process / external HITL | **P1** | OBS-DIAG-X5 |
| X6 | Universal PRODUCT/scenario E2E adoption | **PARTIAL** | 4 initialized; not all E2E; read uneven | **P1** | OBS-DIAG-X6 |
| X7 | Real provider qualification matrix | **OPEN** | Adapters ≠ production proofs | **P1** | OBS-DIAG-X7 |
| X8 | Universal operator read backbone | **PARTIAL** | Contract shared; HTTP/dashboard host-specific | **P1** | OBS-DIAG-X8 |
| X9 | OBS vendor hardening | **OPEN** / **PARTIAL** | Only OTLP slice proven | **P2** | OBS-DIAG-X9 |
| X10 | Final universal enterprise spine | **OPEN** | Depends on X2–X9 | **P0** | OBS-DIAG-X10 |
| — | OECP | **PLANNED** | Architecture only | **P3** | OBS-ECP phases |

---

## I. Maturity (reconciled)

| Axis | Level | Notes |
| ---- | ----- | ----- |
| Architecture (A) | **A4** | Frozen ownership + spine |
| Implementation (I) | **I4 core** | OECP / full host replaceability not I4 |
| Production (P) | **P2 / P3 mixed** | Strong core; uneven host exposure & external spines |
| Evidence (E) | **E3** | Strong gates; not universal E4 |

---

## J. Roadmap X2–X10

| Etap | Cel | Status |
| ---- | --- | ------ |
| **OBS-DIAG-X1** | Canonical SSOT reconciliation + enterprise gap baseline | **COMPLETE (this artifact)** |
| OBS-DIAG-X2 | Diagnostic composition pluginability | **CLOSED** — see [`OBS_DIAG_DIAGNOSTIC_COMPOSITION_PLUGINABILITY_X2.md`](OBS_DIAG_DIAGNOSTIC_COMPOSITION_PLUGINABILITY_X2.md) (X1 matrices above remain historical PARTIAL) |
| OBS-DIAG-X3 | Universal entry-point / zero-bypass qualification | PENDING |
| OBS-DIAG-X4 | Real Kafka cross-process OBS→DIAG spine E2E | PENDING |
| OBS-DIAG-X5 | Full HITL restart → diagnostics E2E qualification | PENDING |
| OBS-DIAG-X6 | Universal PRODUCT/scenario adoption qualification | PENDING |
| OBS-DIAG-X7 | Real provider qualification matrix | PENDING |
| OBS-DIAG-X8 | Universal operator read backbone | PENDING |
| OBS-DIAG-X9 | OBS vendor hardening | PENDING |
| OBS-DIAG-X10 | Final zero-bypass / real-spine enterprise qualification | FINAL |

---

## K. Independent audit requirement

All documentation claims in X1 must be independently verified against real repository documentation, production code, tests, and the GitHub commit. A Cursor AI session report alone is insufficient evidence of correctness.
