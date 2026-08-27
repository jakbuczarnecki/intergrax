# SCENARIO-PLATFORM-1 — Platform-Native Scenario Scaffold Audit

**Audit ID:** SCENARIO-PLATFORM-1  
**Date:** 2026-08-27  
**Branch:** `development`  
**Mode:** audit / documentation only — no implementation  
**Repository:** jakbuczarnecki/intergrax

---

## 1. Executive verdict

**Platform-native scenario scaffolding today is design-only.** `scripts/proof/create_scenario_proof.py` generates a high-quality design-stage package (`README.md` + `SCENARIO_SPEC.md`) with normative observability, survival, and APPLICATION vs PROOF contracts — but **no implementation, execution, observability, diagnostic, or proof scaffold is generated or enforced at runtime**.

The only executable Scenario package is `platform_proofs/scenarios/ai_incident_investigation/`, built manually. It demonstrates valuable patterns (`runtime_composition.py`, `scenario_composition.py`, typed `DiagnosticPayload`, `GraphExecutor` + Critic + ToolRuntime) but each future scenario would currently **rediscover** the same wiring.

**Target:** layered scaffold — universal execution/observability/provenance baseline (AUTO) + typed capability seams (SEAM) + explicit opt-in (OPT-IN) + domain ownership (DOMAIN) + proof packaging (PROOF). **Reject a god-scaffold** that injects diagnostics investigator, Critic, HITL, memory, RAG, or hosting into every scenario by default.

**Biggest systemic gap:** documentation and design templates are strong; **enforcement and shared composition are absent**. Requirements for observability, identity, and platform reuse are **documented** and partially **tested for one scenario**, not **automatic** for new scenarios.

---

## 2. Current scaffold architecture

### 2.1 Scaffold categories

| Category | Status | Evidence |
|----------|--------|----------|
| **A. Design scaffold** | **EXISTS** | `scripts/proof/create_scenario_proof.py`; tests in `tests/unit/scripts/proof/test_create_scenario_proof.py`; `platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md` § Design-stage Scenario scaffold |
| **B. Implementation scaffold** | **ABSENT** | No generator for `proof.json`, `run_proof.py`, `runtime_composition.py`, agents, or providers. Authoring Guide documents post-implementation shape only. |
| **C. Execution scaffold** | **PARTIAL** | Documented target (`Task` → `ExecutionGraph` → `GraphExecutor` / Nexus). Hand-built in `ai_incident_investigation/scenario.py`. No shared `intergrax` or `platform_proofs/_shared` scenario entry module. Products use `build_nexus_loop_from_environment` (`intergrax/applications/_shared/nexus_factory.py`) — not exposed to scenario generator. |
| **D. Observability scaffold** | **PARTIAL** | Design template § Observability / Explainability / Diagnostics Contract in generated `SCENARIO_SPEC.md`. Reference pattern: `investigation_observability.py`, `RuntimeState.trace_event`, `DiagnosticPayload`. No generator placeholder or shared base. |
| **E. Diagnostic scaffold** | **PARTIAL** | Platform contracts: `intergrax/runtime/diagnostics/investigation_contracts.py`, `DiagnosticReadService`. Scenario seam: `scenario_composition.py` (ai_incident only). Architecture gate: `test_diagnostic_architecture_gate.py` (ai_incident only). No generic diagnostic-context consumer scaffold. |
| **F. Proof / evaluator scaffold** | **PARTIAL** | Generic proof infrastructure: `scripts/proof/*` (`PlatformProofEvidence` v3, runner, renderer, verifier). Per-scenario `evaluator.py`, `evidence_builder.py`, `run_proof.py` hand-built for ai_incident. Design generator **forbids** these at design stage (`DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES`). |

### 2.2 Generator inventory

| Path | Role |
|------|------|
| `scripts/proof/create_scenario_proof.py` | **Only** scenario scaffold CLI — design stage |
| `tests/unit/scripts/proof/test_create_scenario_proof.py` | Design package contract tests |
| `platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md` | Canonical workflow + package shape |
| `platform_proofs/PLATFORM_PROOF_PROTOCOL.md` | Governance + descriptor contract |
| `scripts/proof/intergrax_platform_proof_descriptor.py` | `proof.json` schema |
| `scripts/proof/intergrax_proof_runner.py` | Discovery + execution (not scenario-specific) |

No other scenario template, cookiecutter, or promote/initialize command exists.

### 2.3 Lifecycle constants

`create_scenario_proof.py` defines:

- `LIFECYCLE_DESIGN_NOT_ACCEPTED` — written into generated docs
- `LIFECYCLE_ACCEPTED_FOR_IMPLEMENTATION` — **defined but not enforced** by any generator or gate

Quality gate remains **human** (Scenario Quality Gate in SCENARIO_SPEC § A). Tests verify section presence, not approval status.

---

## 3. Guide vs enforcement gap

### 3.1 Matrix

| Requirement | Guide | Scaffold (generated) | Gate (test/CI) | Runtime (automatic) |
|-------------|-------|----------------------|----------------|---------------------|
| execution identity (`TaskId`, `RunId`, `AttemptId`) | yes | prompt only | ai_incident tests only | **partial** — `Task()` mints `task_id`; run/attempt require manual `mint_*` + `bind_active_execution_identity` at orchestration boundary |
| Task / Run lifecycle | yes | prompt only | ai_incident integration tests | **partial** — via `GraphExecutor`; full `NexusLoop` lifecycle not default |
| `GraphExecutor` / Nexus entry | yes | no | ai_incident only | **no** — scenario-owned wiring |
| `RuntimeState` / `trace_event` | yes | prompt only | ai_incident unit tests | **no** — only when agent path wires `RuntimeState` |
| `TraceEvent` / `ToolCallTrace` | yes | prompt only | ai_incident tests | **partial** — via ToolRuntime path when composed |
| `ObservabilityEmitter` | yes | no | no | **no** for GraphExecutor-only scenario path unless explicitly bridged |
| `DiagnosticPayload` (typed) | yes | prompt only | ai_incident observability tests | **no** — scenario must author payloads |
| `RuntimeEvent` canonical execution truth | yes | no | DIAG-8C integration tests | **no** for ai_incident canonical path — uses `GraphExecutor` without persisted runtime event spine + terminal diagnostic bridge |
| causal evidence | yes | conditional prompt | platform tests only | **no** — opt-in via hosting/Nexus composition |
| `DiagnosticOrchestrator` / Problem write | yes | no | platform tests only | **no** on GraphExecutor-only scenario path; **yes** on full `NexusLoop` with terminal bridge |
| `DiagnosticReadService` | yes | no | ai_incident DIAG-8C tests | **no** — composition-only seam |
| `Problem` / `ProblemId` | yes | no | DIAG-8C tests | **no** — opt-in via `scenario_composition.py` |
| `IncidentInvestigationInput` | yes (DIAG-8) | no | DIAG-8C tests | **no** — domain-specific seam, not universal |
| `EvidenceBackedClaim` / `EvidenceClaimSet` | yes | prompt only | ai_incident tests | **no** — domain wiring |
| Critic hooks | yes | conditional prompt | ai_incident tests | **no** — explicit `build_critic_graph_hooks` |
| `EvaluatorLoop` | yes | conditional prompt | ai_incident tests | **no** — explicit `EvaluatorLoopSpec` |
| `RuntimePolicyBundle` | yes | no | product wiring tests | **partial** — ai_incident passes empty bundle via `ApplicationBuildContext` |
| HITL / approvals | conditional | conditional prompt | no | **no** |
| `ToolRegistry` / `ToolRuntime` | yes | no | ai_incident tests | **no** — scenario declares tools + builds invoker |
| LLM (`resolve_llm_adapter`) | yes | no | ai_incident runtime_composition tests | **no** — scenario builds `ApplicationEnvironmentProfile` |
| tenant identity | yes (implicit) | no | DIAG-8C tenant isolation tests | **no** — `STANDALONE_SCENARIO_TENANT_ID` hardcoded in ai_incident |
| provider lifecycle | yes | prompt only | no | **no** |
| persistence | conditional | no | no | **no** — scenario-local stores unless composed |
| `PlatformProofEvidence` projection | yes | forbidden at design | ai_incident + `scripts/proof` verifier | **no** — proof layer hand-built |
| report / evidence packaging | yes | forbidden at design | PP-SUITE tests | **no** — `evidence_builder.py` per scenario |
| redaction (`DiagnosticPayload.redact`) | yes | prompt only | payload unit tests | **no** — per-payload author responsibility |
| retry / recovery | conditional | conditional prompt | platform only | **no** unless Nexus/hosting wired |
| budget / limits | conditional | no | platform only | **no** unless profile wired |
| application hosting | conditional | no | HOST-DIAG-1 gate (hosting only) | **no** — not scenario default |
| Application Survival Test | yes | **generated section** | docs contract test | **no** |
| Application Observability Test | yes | **generated section** | docs contract test | **no** |
| no FakeLLM in canonical path | yes | prompt only | acceptance checklist (manual) | **no** automated scenario gate |
| APPLICATION vs PROOF separation | yes | **generated table** | ai_incident architecture gate (diagnostics only) | **no** structural enforcement |

**Pattern:** Guide + design template = strong. Runtime automation = weak. Enforcement = **one reference scenario** + documentation contract tests, not universal architecture gates.

---

## 4. Platform capability matrix (current → target)

| # | Capability | Current owner | Current scaffold | Target default | Why |
|---|------------|---------------|------------------|----------------|-----|
| 1 | execution identity | `intergrax.contracts.execution_identity`; scenario orchestrator mints at boundary | prompt only | **AUTO** | Legitimate mint at orchestration boundary; authors must not invent IDs ad hoc |
| 2 | Task / Run lifecycle | `intergrax.runtime.task`; `GraphExecutor` / `NexusLoop` | none | **AUTO** (minimal Task+graph) | Every autonomous scenario needs a task envelope |
| 3 | GraphExecutor / Nexus | platform runtime | none | **SEAM** | GraphExecutor sufficient for single-agent scenarios; Nexus for multi-step/hosted |
| 4 | RuntimeState | Nexus agent path | none | **AUTO** when execution runs | Trace owner for material decisions |
| 5 | TraceEvent | `RuntimeState.trace_event` | prompt | **AUTO** via runtime path | Baseline observability |
| 6 | ToolCallTrace | ToolRuntime / runtime state | none | **AUTO** when tools enabled | Tool scenarios must not rebuild telemetry |
| 7 | ObservabilityEmitter | platform observability bridges | none | **SEAM** | Full emitter profile is product-grade; scenarios need minimal bridge |
| 8 | DiagnosticPayload | scenario-typed + platform base | prompt | **SEAM** | Domain payloads; scaffold provides module convention not generic filler |
| 9 | RuntimeEvent execution truth | platform event store + Nexus terminal bridge | none | **SEAM** | AUTO only when scenario opts into hosted/Nexus persistence path |
| 10 | causal evidence | platform observability persistence | none | **OPT-IN** | Investigation/hosting scenarios only |
| 11 | DiagnosticOrchestrator | platform diagnostics | none | **SEAM** (write via spine) | Do not clone; wire through normal terminal bridge when execution spine present |
| 12 | DiagnosticReadService | platform diagnostics | ai_incident `scenario_composition.py` | **SEAM** | Generic read consumer at composition boundary |
| 13 | Problem / ProblemId | platform diagnostics | none | **OPT-IN** | Not every scenario investigates a Problem |
| 14 | IncidentInvestigationInput | `investigation_contracts.py` | none | **OPT-IN** | Domain-specific investigation contract; pattern for other typed inputs |
| 15 | EvidenceBackedClaim / EvidenceClaimSet | `intergrax.contracts.evidence_claims` | prompt | **SEAM** | Import contract; wire when claims material |
| 16 | Critic hooks | `intergrax.runtime.critic` | conditional prompt | **OPT-IN** | Required only when critic semantics exist |
| 17 | EvaluatorLoop | `intergrax.runtime.critic.evaluator_loop_spec` | conditional prompt | **OPT-IN** | Revision loop when critic challenges apply |
| 18 | governance / RuntimePolicyBundle | `intergrax.runtime.policy` | none | **AUTO** (default empty bundle) | Normal runtime expects policy slot; empty default is fine |
| 19 | HITL / approvals | platform governance | conditional prompt | **OPT-IN** | Problem-dependent |
| 20 | ToolRegistry / ToolRuntime | `intergrax.tools` | none | **SEAM** | Declare `ToolProfile`; scaffold wires runtime, not reimplementation |
| 21 | LLM provider resolution | `resolve_llm_adapter` + `ApplicationEnvironmentProfile` | none | **OPT-IN** | Capability-driven; non-AI scenarios skip |
| 22 | tenant identity | caller / composition / `Task.tenant_id` | none | **AUTO** (explicit contract) | Propagate declared tenant; no ContextVar; synthetic tenant only for standalone proof |
| 23 | provider lifecycle | application providers | none | **DOMAIN** + **SEAM** | Typed provider interface in `providers/` |
| 24 | persistence | platform services or scenario fixtures | none | **OPT-IN** | Fixture/store only when domain requires |
| 25 | PlatformProofEvidence projection | `scripts/proof` + scenario `evidence_builder.py` | forbidden at design | **PROOF** | Proof layer consumes runtime artifacts |
| 26 | report / evidence packaging | `run_proof.py`, renderer | forbidden at design | **PROOF** | Never in application core |
| 27 | redaction | `DiagnosticPayload.redact()` | prompt | **SEAM** | Placeholder module + lint gate for payloads |
| 28 | retry / recovery | Nexus retry engine | none | **OPT-IN** | Material only for resilience scenarios |
| 29 | budget / limits | `RunBudget` / profiles | none | **OPT-IN** | Material only when limits are claim under test |
| 30 | application hosting | `intergrax/hosting` | none | **OPT-IN** | LKW lesson: hosting scenarios only; composition boundary for diagnostics |

---

## 5. Automatic baseline (MUST-BE-AUTOMATIC)

Derived from platform ownership and ai_incident lessons — **automatic means scaffold wires it; author does not rediscover**.

| Baseline | Mechanism |
|----------|-----------|
| Execution identity minting at orchestration boundary | `mint_run_id`, `mint_attempt_id`, `bind_active_execution_identity` in generated `scenario.py` skeleton |
| `Task` envelope with explicit `tenant_id` | Generated task factory taking `tenant_id` parameter — no hidden default except documented standalone proof constant |
| `ApplicationEnvironmentProfile.lab_defaults` + `ApplicationBuildContext.for_manifest` | Generated `runtime_composition.py` minimal profile |
| Default `RuntimePolicyBundle()` | Empty bundle slot in build context |
| Tool profile declaration → ToolRuntime wiring | Generated seam: `ToolProfile` + registry registration hook |
| `RuntimeState.trace_event` path for agent execution | Via `build_runtime_context_from_environment` / standard agent UAEP path |
| `ToolCallTrace` when tools invoked | Through platform ToolRuntime — not custom invoke wrappers without trace |
| Proof descriptor discovery | `proof.json` template with v3 schema fields |
| Provenance metadata in proof run | `run_proof.py` skeleton: git SHA, artifact dir env |
| PHYSICAL separation application vs proof modules | Directory convention enforced by generator (see §15) |

**Not automatic (explicitly):** Problem read, Critic, EvaluatorLoop, LLM, hosting, causal evidence, NexusLoop full orchestration.

---

## 6. Standard seams (MUST-HAVE-STANDARD-SEAM)

Typed integration points scaffold provides; scenario activates when relevant.

| Seam | Pattern source | Generalization |
|------|----------------|----------------|
| Runtime composition | `runtime_composition.py` | `build_scenario_environment_profile()` + `build_scenario_runtime_composition(registry, environment)` |
| Platform diagnostic read | `scenario_composition.py` | `resolve_<capability>_input(diagnostic_read_service, tenant_id, ...)` — **not** hardcoded `IncidentInvestigationInput` in every scenario |
| Typed diagnostic context rendering | `platform_diagnostic_context.py` | Optional `format_*_context_lines(input)` beside domain contract |
| Domain observability payloads | `investigation_observability.py` | `observability.py` or `<domain>_observability.py` with `DiagnosticPayload` subclasses |
| Critic configuration | `scenario.py` `build_critic_graph_hooks` | Capability flag generates critic hook wiring stub |
| Evidence claims | `incident_reasoning.convert_proposal_to_pending_claims` | SEAM: pending-claim conversion pattern, domain-owned kinds |
| Provider boundary | `fixtures.py` + `tools.py` | `providers/` with shared typed contract; synthetic + real implement same interface |
| Execution entry | `scenario.py` `execute_*` | `execute_scenario(bundle) -> ScenarioExecutionResult` — GraphExecutor default, Nexus opt-in |
| LLM resolution | `resolve_scenario_llm_adapter` | Generated only when `llm_required=true` in capability declaration |

**Diagnostic read generalization:** expose a **generic composition module** (`scenario_composition.py`) that imports `DiagnosticReadService` and maps to a **scenario-declared** input type (e.g. `IncidentInvestigationInput`). Investigation scenarios use existing contract; other scenarios define their own bounded read DTOs in platform when reuse emerges — **do not** put `IncidentInvestigationInput` in universal scaffold.

---

## 7. Opt-in capabilities

| Capability | Why opt-in |
|------------|------------|
| `IncidentInvestigationInput` | Only investigation scenarios consume Problem context |
| `DiagnosticReadService` | Read side meaningless without platform Problems |
| Critic / EvaluatorLoop | Scenarios without verifier semantics should not pay revision-loop cost |
| HITL / approvals | Governance scenarios only |
| LLM / `resolve_llm_adapter` | Non-AI scenarios exist (deterministic workflow proofs) |
| RAG / web / memory | `ApplicationEnvironmentProfile` profiles — off by default |
| causal evidence / RuntimeEvent persistence | Requires hosted execution spine |
| `DiagnosticOrchestrator` write | Owned by platform terminal bridge — scenario must not clone; opt-in by choosing Nexus/hosted path |
| application hosting | LKW-style product hosting ≠ scenario default |
| root-cause promotion / adjudication | Future platform capability; not universal |
| long-term persistence | Fixture/session-local default for proofs |

---

## 8. Scenario-owned responsibilities (DOMAIN)

Scaffold **must not** absorb:

- business problem, workflow story, adversarial worlds
- domain hypotheses and competing explanations
- domain tools and provider semantics
- domain prompts and reasoning proposals
- domain-specific `DiagnosticPayload` semantics (schema IDs, fields)
- domain claim kinds and resolution predicates (`validation.py`, `domain_reasoning.py`)
- success/failure / RESOLVED / UNRESOLVED **domain** meaning (`InvestigationConclusion` — scenario authority)
- fixture hidden truth for evaluator oracle
- synthetic evidence stores (`ScenarioEvidenceStore`)

Platform supplies rails; scenario supplies intelligence.

---

## 9. Proof-owned responsibilities (PROOF)

| Artifact | Owner |
|----------|-------|
| `evaluator.py` / falsification assertions | proof |
| `evidence_builder.py` → `PlatformProofEvidence` | proof |
| `evaluator_evidence.py` projection helpers | proof |
| `run_proof.py` subprocess entry | proof |
| `reproduction.py` / shell commands | proof |
| `report_sections.py` / HTML domain sections | proof |
| `proof.json` descriptor | proof (package metadata) |
| fixture variant selection for adversarial cases | proof harness config |
| PASS/FAIL oracle comparing to hidden truth | proof |

**Invariant:** proof projection **consumes** application/runtime observability — never the sole producer (Application Observability Test).

---

## 10. Observability standard

### 10.1 Minimum baseline (every executable autonomous Scenario)

| Layer | Owner | Auto-wired target |
|-------|-------|-------------------|
| execution correlation | platform identity | run/attempt on orchestration boundary |
| sequencing | `RuntimeState` | agent step traces |
| tool invocation | ToolRuntime | `ToolCallTrace` |
| material decision summaries | scenario `DiagnosticPayload` | typed payloads via `trace_event` |
| terminal outcome facts | scenario domain payload + critic verdict | structured dict / claim set — not prose-only |
| proof export | `PlatformProofEvidence` steps/graph | projects from above |

### 10.2 Canonical owners (do not reimplement)

- `RuntimeState.trace_event(...)` / `trace_events`
- `TraceEvent`, `TraceComponent`, `TraceLevel`
- `ToolCallTrace` / `RuntimeState.tool_traces`
- `DiagnosticPayload` + `redact()`
- Critic trace via `CriticTraceEmitter`
- `intergrax.platform_proof_evidence.v3` for proof packaging

### 10.3 Domain extension convention

Recommend **`<domain>_observability.py`** (ai_incident: `investigation_observability.py`) — not an empty generated `observability.py`. Scaffold should:

- document the convention in generated `SCENARIO_SPEC.md`
- optionally create the file **only** when observability capability is declared
- never generate meaningless generic payloads

### 10.4 Anti-patterns (forbidden)

- proof-only event bus
- custom correlation ID scheme
- report-synthesized tool calls / decisions
- chain-of-thought capture

---

## 11. Diagnostics standard

### 11.1 Write side (current coverage)

| Path | Diagnostic write automatic? |
|------|----------------------------|
| Full `NexusLoop` + runtime event persistence + terminal bridge | **Yes** — `invoke_terminal_execution_diagnostics` in `nexus_loop.py` |
| `GraphExecutor`-only scenario (ai_incident canonical) | **No** — emits some `RuntimeEvent` types via executor but **does not** invoke `DiagnosticOrchestrator` / Problem materialization |
| Scenario-local diagnostic write pipeline | **Forbidden** — do not clone orchestrator |

**Recommendation:** scaffold documents two execution profiles:

1. **Light graph profile** — GraphExecutor, observability via `RuntimeState`; diagnostics write **not** claimed unless wired to event store + terminal bridge.
2. **Hosted/spine profile** — Nexus/hosting composition; diagnostics write **automatic** at platform terminal boundary.

Scenarios must not create scenario-local diagnostic write pipelines.

### 11.2 Read side (DIAG-8C lesson)

| Element | Status |
|---------|--------|
| `DiagnosticReadService` | platform canonical read surface |
| `IncidentInvestigationInput` | platform bounded contract for investigation scenarios |
| `scenario_composition.py` | **only** composition file may import read service (enforced in ai_incident gate) |
| reasoning modules | must not import `DiagnosticReadService` |

**Generalize as:** `scenario_composition.py` = **diagnostic context consumer seam** — maps read service → typed bounded input. Capability-specific contract (`IncidentInvestigationInput` or future types), not universal scaffold import.

### 11.3 Non-execution subjects

`DiagnosticProblemOccurrenceView` carries `application-instance` subject refs and `NON_EXECUTION_SUBJECT` unavailable reasons (DIAG-8B). Scaffold must:

- not require `TaskId`/`RunId` on generic scenario contracts
- use `tenant_id` + `ProblemId` at composition boundary when diagnostics read is enabled
- document standalone vs platform-attached modes (ai_incident: `STANDALONE_SCENARIO_TENANT_ID` vs `investigation_input.tenant_id`)

---

## 12. Tenant / execution identity standard

### 12.1 Tenant

| Mode | Source | Acceptable |
|------|--------|------------|
| Platform-attached | `IncidentInvestigationInput.tenant_id` / caller-supplied | production-like |
| Standalone proof | documented synthetic constant (e.g. `scenario-tenant`) | proof-only when standalone mode explicit |
| Production-attached | host/application manifest tenant | required for hosting scenarios |

**Rules:**

- tenant flows through `Task.tenant_id` and tool/runtime calls
- no ContextVar / global tenant magic
- design scaffold should add **Tenant contract** subsection in SCENARIO_SPEC (currently implicit)
- hardcoded tenant without mode documentation → architecture gate violation

### 12.2 Execution identity audit (ai_incident)

| Usage | Classification |
|-------|----------------|
| `Task(...)` auto `task_id` | platform-owned legitimate |
| `mint_run_id()`, `mint_attempt_id()` + `bind_active_execution_identity` in `scenario.py` | platform-owned legitimate at orchestration boundary |
| `mint_*` in DIAG-8C tests building runtime events | test-only legitimate |
| `STANDALONE_SCENARIO_TENANT_ID = "scenario-tenant"` | scenario-owned — acceptable for standalone proof with documented contract |
| synthetic user_id `"scenario-user"` | scenario-owned — acceptable for proof |

**Gate:** forbid `mint_task_id` / `mint_run_id` in domain reasoning, tools, evaluator, evidence_builder.

---

## 13. Runtime / composition standard

### 13.1 Reference pattern (ai_incident — reusable in part)

```text
runtime_composition.py     → ApplicationEnvironmentProfile + ApplicationBuildContext
scenario_composition.py    → optional platform capability wiring (diagnostics read)
<agent>.py                 → UAEP agent, uses build_agent_runtime_context
scenario.py                → Task + ExecutionGraph + GraphExecutor + optional Critic
proof: run_proof.py        → adversarial variants + evidence projection
```

### 13.2 Not cargo-cult

- `IncidentInvestigatorAgent` domain logic
- dual RESOLVED/UNRESOLVED worlds (proof adversarial design)
- `_IncidentToolCallIdAdapter` (provider quirk workaround)
- `IncidentInvestigationInput` in every scenario

### 13.3 Product contrast (LKW)

LKW is **product proof** under `applications/local_workspace_application/` — not a scenario scaffold template. Lessons:

- tenant must come from real operation/host context
- do not synthesize execution identity for non-execution subjects
- hosting meets diagnostics at **composition boundary** (`HOST-DIAG-1` gate — hosting must not bypass platform diagnostics bridge)

### 13.4 Existing contracts to reuse (no `ScenarioPlatformCapabilities` duplicate)

| Contract | Role |
|----------|------|
| `ApplicationEnvironmentProfile` | capability/profile declaration (LLM, RAG, web, memory, critic, …) |
| `ApplicationBuildContext` | registry, policy, tool profile wiring |
| `ToolProfile` | allowed tools |
| `IntegrationProfile` | external integrations |
| `RuntimePolicyBundle` | governance |
| `proof.json` / descriptor v3 | proof package metadata |
| `ProofManifestEntry` | runner discovery |

**Recommendation:** extend scenario implementation scaffold to generate a **minimal** `ApplicationEnvironmentProfile` with explicit toggles — not a parallel `ScenarioPlatformCapabilities` type unless audit proves profile fields insufficient.

---

## 14. Application vs proof harness

### 14.1 Documentation enforcement

- Generated `SCENARIO_SPEC.md` includes APPLICATION vs PROOF table
- `DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES` blocks premature proof files
- Authoring Guide Public Library acceptance gates (14 YES/NO checks) — **manual**

### 14.2 Structural enforcement today

**Weak.** ai_incident mixes application and proof modules in one directory (authoring-guide-allowed). Only diagnostic import boundary is gated.

### 14.3 Recommended physical shape (future generator)

```text
platform_proofs/scenarios/<slug>/
├── README.md
├── SCENARIO_SPEC.md
├── proof.json
├── run_proof.py                 # PROOF entry
├── reproduction.py              # PROOF
├── report_sections.py           # PROOF optional
├── proof/
│   ├── evaluator.py
│   ├── evidence_builder.py
│   └── evaluator_evidence.py
├── application/
│   ├── scenario_contract.py
│   ├── runtime_composition.py
│   ├── scenario_composition.py  # optional
│   ├── scenario.py
│   ├── observability.py         # when declared
│   ├── agent(s).py
│   ├── tools.py
│   └── providers/
├── fixtures/                    # proof-isolated hidden truth
└── output/
```

Flat layout remains valid per Authoring Guide; **subpackages recommended** for gates (forbid `proof/` imports from `application/`).

---

## 15. Architecture gates (proposed)

### 15.1 Universal (every scenario package)

| Gate | Rationale |
|------|-----------|
| no `testing_support` / `FakeLLM` / `MagicMock` in `application/` | production-capable path |
| no `evaluator` / `evidence_builder` imported by application core | proof separation |
| no proof-only event bus / logger as sole trace | observability test |
| no `ProblemPersistence` / `DiagnosticOrchestrator` / `ExecutionReconstructor` in application domain modules | one-spine |
| `DiagnosticReadService` only in `scenario_composition.py` (or `*_composition.py`) | DIAG-8C boundary |
| no synthetic execution identity in domain modules | identity ownership |
| `proof.json` + `run_proof.py` present for executable packages | discovery contract |
| generated observability payloads implement `redact()` when containing sensitive fields | security |

### 15.2 Capability-specific

| Gate | When |
|------|------|
| Critic hooks required | `critic_enabled` capability |
| `DiagnosticReadService` composition tests | `diagnostics_read` capability |
| no standalone hardcoded tenant in platform-attached mode | `diagnostics_read` capability |
| LLM resolution smoke test | `llm_required` capability |
| HOST-DIAG-style bridge | hosting scenarios only |

---

## 16. Target layered model

| Layer | Name | Contents |
|-------|------|----------|
| **L0** | Design scaffold | `README.md`, `SCENARIO_SPEC.md` (exists) |
| **L1** | Execution baseline | `runtime_composition.py`, `scenario.py` skeleton, tenant + identity helpers |
| **L2** | Observability baseline | UAEP agent path, `ToolRuntime` trace, observability module convention |
| **L3** | Platform capability seams | diagnostics read composition, critic, LLM, governance, hosting |
| **L4** | Domain application | agents, tools, providers, domain reasoning, validation |
| **L5** | Proof / evaluator | `proof/*`, `run_proof.py`, evidence projection, reproduction |

---

## 17. Real scenario comparison

### 17.1 `ai_incident_investigation` (only full Scenario package)

| Aspect | Automatic | Manual / duplicated | Missing | Should become scaffold |
|--------|-----------|---------------------|---------|------------------------|
| Design docs | generated template (retrofit) | full manual implementation | — | design generator (exists) |
| `runtime_composition` | — | copied pattern from product bridges | shared module | L1 generator |
| `GraphExecutor` + identity | — | hand-written `scenario.py` | — | L1 generator |
| ToolRuntime + traces | via platform when wired | custom `_IncidentScopedToolInvoker` | — | L2 tool seam template |
| Observability payloads | — | hand-written `investigation_observability.py` | convention only | L2 convention + optional stub |
| Diagnostics read | — | hand-written `scenario_composition.py` | generic helper | L3 seam template |
| Critic / EvaluatorLoop | — | explicit in `scenario.py` | — | L3 opt-in stub |
| Tenant | — | `STANDALONE_SCENARIO_TENANT_ID` | explicit contract in design docs | L1 tenant section + gate |
| Proof packaging | generic `scripts/proof` | hand-written evidence_builder | — | L5 generator |
| Architecture gates | — | one diagnostic gate file | universal scenario gates | L6 |

### 17.2 LKW (`applications/local_workspace_application/`)

Product proof — **not** scenario scaffold. Lessons imported: tenant from operations, hosting/diagnostics composition boundary, non-execution subjects. **Do not** migrate LKW in this program.

### 17.3 Other packages

| Package | Role | Scenario relevance |
|---------|------|-------------------|
| `platform_proofs/test_domain/fake_publish/` | conformance artifact residue | not a scenario pattern |
| `proof_infrastructure/governed_hybrid_knowledge_proof/` | legacy proof infrastructure | pre-scenario; docker harness — not platform-native scenario model |
| Design-only future slugs | — | only `create_scenario_proof.py` output |

**No other implemented Scenario packages exist** — ai_incident is the sole reference.

---

## 18. DIAG-8 lessons — what new scenarios should not rediscover

| Phase | Lesson | Scaffold responsibility |
|-------|--------|-------------------------|
| **Before DIAG-8** | Scenario was runtime-native but diagnostically standalone | document standalone vs attached modes |
| **DIAG-8B** | Platform owns `IncidentInvestigationInput` / `InvestigationConclusion` | import contracts — do not redefine |
| **DIAG-8C** | `DiagnosticReadService` only at composition boundary | generate `scenario_composition.py` seam + architecture gate |
| **Terminology** | RESOLVED ≠ Problem closure | generate InvestigationConclusion guidance in spec |
| **Identity** | Do not synthesize execution IDs for app-instance subjects | tenant + subject ref documentation |
| **Write path** | Do not clone orchestrator | document GraphExecutor vs Nexus spine choice |

---

## 19. Design vs implementation generator decision

### 19.1 Options

| Option | Assessment |
|--------|------------|
| **A. Extend `create_scenario_proof.py`** | Risk: blurs design vs implementation lifecycle |
| **B. Separate `promote_scenario_implementation` command** | **Recommended** — preserves human Quality Gate |
| **C. External cookiecutter** | Duplicates repo conventions; avoid |

### 19.2 Recommendation: **B**

```text
create_scenario_proof.py          → design only (unchanged role)
init_scenario_implementation.py   → new; requires ACCEPTED FOR IMPLEMENTATION in SCENARIO_SPEC
```

Implementation generator checks:

- `SCENARIO_SPEC.md` contains `ACCEPTED FOR IMPLEMENTATION` (or future machine-readable frontmatter)
- `INTERGRAX FIT` ≠ `NOT YET PERFORMED`
- `GAP DECISION` resolved
- APPLICATION vs PROOF section filled
- Observability contract filled

**Must not** generate runnable domain logic — only rails, stubs, and gates.

### 19.3 Quality gate behavior

| Stage | Generator allowed | Human gate |
|-------|-------------------|------------|
| DESIGN / NOT YET ACCEPTED | design only | Scenario Quality Gate |
| ACCEPTED FOR IMPLEMENTATION | implementation skeleton | Intergrax Fit + Gap Decision |
| Executable | proof.json + run_proof stub | Public Library acceptance (14 gates) |

Constant `LIFECYCLE_ACCEPTED_FOR_IMPLEMENTATION` already exists — wire it in SCENARIO-PLATFORM-2/3.

---

## 20. Migration strategy

| Rule | Action |
|------|--------|
| Existing scenarios | opportunistic migration — **no mass rewrite** |
| New scenarios | must use new scaffold after SCENARIO-PLATFORM-3 |
| Reference | **`ai_incident_investigation`** — prove patterns, then extract shared seams |
| LKW | out of scope — product, not scenario |
| Architecture gates | prevent **new** divergence (import boundaries, proof separation) |

**First migration candidate:** ai_incident — add `application/` vs `proof/` subpackages when implementing SCENARIO-PLATFORM-6 (optional flatten compatibility).

---

## 21. Proposed implementation slices

| Slice | Scope |
|-------|-------|
| **SCENARIO-PLATFORM-2** | Lifecycle contracts: tenant section in design template; capability toggles on `ApplicationEnvironmentProfile` authoring guidance; `ACCEPTED FOR IMPLEMENTATION` enforcement spec; universal architecture gate test module template |
| **SCENARIO-PLATFORM-3** | `init_scenario_implementation.py` — gated skeleton (L1 + L5 stubs, no domain logic) |
| **SCENARIO-PLATFORM-4** | Shared `platform_proofs/_shared/scenario_runtime_baseline.py` (or `intergrax/applications/_shared/scenario_baseline.py`) — identity bind helper, minimal composition, ToolRuntime wiring |
| **SCENARIO-PLATFORM-5** | Diagnostic read seam template + generalized composition gate; document GraphExecutor vs Nexus spine selection |
| **SCENARIO-PLATFORM-6** | Universal scenario architecture gates in CI; refactor ai_incident as reference; optional directory split |

---

## 22. Risks

| Risk | Mitigation |
|------|------------|
| God-scaffold | capability flags on existing profiles; layered L0–L5 |
| Cargo-cult ai_incident | document what is domain-specific vs rail |
| GraphExecutor path skips diagnostic write | explicit spine profile documentation; do not fake Problem write in scenario |
| Flat package structure | import-boundary gates even without subdirs |
| Design/implementation gate bypass | separate CLI + status check |
| `IncidentInvestigationInput` overgeneralization | keep investigation-specific; add new platform contracts when second consumer exists |

---

## 23. Open questions

1. Should scenario baseline live under `platform_proofs/_shared/` or `intergrax/applications/_shared/`? (Tier boundary: scenarios are Tier-3-like packages inside `platform_proofs/`.)
2. Is `GraphExecutor`-only the default execution profile for v1 implementation scaffold, with Nexus as opt-in seam?
3. Machine-readable lifecycle frontmatter in `SCENARIO_SPEC.md` vs string search for `ACCEPTED FOR IMPLEMENTATION`?
4. When second non-investigation scenario needs diagnostic read, introduce generic `DiagnosticContextInput` protocol or per-domain contracts only?
5. Should conformance proofs under `platform_proofs/<domain>/` share any scenario L1 baseline, or remain separate harness model?
6. Minimum CI gate: run universal scenario architecture tests on every `platform_proofs/scenarios/*` package?

---

## Appendix A — Default matrix (summary)

See §4 for full table. Summary counts:

| Target default | Count (of 30) |
|----------------|---------------|
| AUTO | 6 |
| SEAM | 12 |
| OPT-IN | 9 |
| DOMAIN | 1 (provider lifecycle domain portion) |
| PROOF | 2 |

---

## Appendix B — P0 findings

1. **Implementation scaffold absent** — every executable scenario is hand-rolled.
2. **Enforcement gap** — strong docs, minimal automated gates beyond ai_incident.
3. **No shared runtime baseline** — `runtime_composition.py` pattern not extracted.
4. **Diagnostic write not automatic** on GraphExecutor-only path — must be explicit in scenario docs.
5. **Tenant contract undocumented** in design generator — leads to hardcoded `scenario-tenant`.
6. **Lifecycle constant unused** — `ACCEPTED FOR IMPLEMENTATION` not enforced before implementation codegen.
7. **Single scenario reference** — high risk of tacit knowledge loss for next author.

---

## 24. SCENARIO-PLATFORM-2 resolution (2026-08-27)

**Authoritative contract:** [`docs/project/maintainers/plans/SCENARIO_RUNTIME_BASELINE.md`](../plans/SCENARIO_RUNTIME_BASELINE.md)

| Question (§23) | Resolution |
|----------------|------------|
| Baseline location | `intergrax/applications/_shared/scenario_runtime_baseline.py` (planned 3A) — **not** `platform_proofs/_shared` for runtime |
| Default execution spine | **C — shared facade → NexusLoop** (not GraphExecutor direct) |
| Lifecycle gate | Recommend YAML frontmatter on `SCENARIO_SPEC.md` in 3B; string constant transitional |
| Diagnostic read generalization | Per-domain DTO at composition seam; no universal `IncidentInvestigationInput` |
| Conformance proofs | Separate harness model; may use GraphExecutor for low-level conformance only |

**§5 correction:** manual `mint_run_id` / `bind_active_execution_identity` in generated `scenario.py` is **not** AUTO baseline — platform facade owns identity.

---

*End of SCENARIO-PLATFORM-1 audit (updated through SCENARIO-PLATFORM-2).*
