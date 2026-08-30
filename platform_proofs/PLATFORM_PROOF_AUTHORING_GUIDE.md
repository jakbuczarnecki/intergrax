# Intergrax Proof Library — Authoring Guide

**Status:** Canonical  
**Audience:** Independent Scenario Proof and Conformance Proof author sessions

This is the **single canonical practical instruction** for independent proof-author sessions. Future sessions receive only:

1. this guide; and
2. a description of the real-world scenario or mechanism under test.

The guide supplies the rest of the context and rules. Do not create a second overlapping authoring guide.

**Normative companion:** [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) — classification, falsification philosophy, evidence contracts. This guide is the operational workflow; the protocol is the governance layer.

**Never:** classify a product (`applications/`) as a platform domain or migrate product proofs into `platform_proofs/`.

---

## Proof Library public model

The Intergrax Proof Library has two public classes. Both are **executable falsification attempts** — not demos. **Platform proof ≠ product proof.** Product proofs stay under `applications/`.

### SCENARIO

The **primary public Proof Library unit**. A Scenario Proof is a **production-capable autonomous application component** that solves a concrete real-world problem through normal Intergrax runtime and platform contracts. The Proof Library layer executes adversarial cases, falsifies claims, captures evidence, evaluates outcomes, and renders reports; **it does not replace the application itself**.

Starts from:

```text
REAL PROBLEM
→ PRODUCTION-CAPABLE APPLICATION
→ INTERGRAX PLATFORM MECHANISMS
→ REAL CONFIGURED AI / SYSTEM BOUNDARIES
→ ADVERSARIAL EXECUTION
→ FALSIFICATION
→ EVIDENCE
→ VERDICT
→ REPRODUCTION
```

A Scenario Proof may exercise **multiple** Intergrax mechanisms and domains. The problem owns the Scenario; platform domains participate according to the guarantees required by that problem. Participating domains are discovered during Intergrax Fit — not chosen before the scenario exists. The scenario must **never** be selected merely to demonstrate a feature.

#### production-capable vs production-validated

| Term | Meaning |
|------|---------|
| **production-capable** | Architecture and canonical execution path are suitable for real deployment; no test-only/fake application shortcuts; normal configurable provider/runtime contracts; replaceable controlled data sources. **Required** for Scenario Proof acceptance. |
| **production-validated** | Real production users, data, or environment validation. **Not** automatically established by Scenario Proof acceptance. |

Use **production-capable** as the precise engineering term. Do **not** write "production-proven" or "production-validated" when describing Scenario Proof acceptance.

### CONFORMANCE

Mechanism-level proof used for:

- CI
- regression
- contract verification
- architecture confidence
- platform development

Canonical path:

```text
PLATFORM MECHANISM
→ CONTROLLED HARNESS
→ CONTRACT / INVARIANT
→ EVIDENCE
```

Conformance proofs are **secondary** in the public Proof Library. They are mechanism-first; Scenario proofs are problem-first. Conformance may legitimately use controlled harnesses and test doubles **outside** the claimed mechanism boundary. SCENARIO canonical execution has **production-capable application** requirements that Conformance does not.

Existing domain-oriented paths may remain for Conformance proofs under `platform_proofs/<domain>/…`. Scenario proofs use `platform_proofs/scenarios/<scenario_slug>/`. Do not mandate mass folder migration for symmetry.

---

## Scenario Proof — production-capable application contract

These rules are **normative** for every Scenario Proof. They apply to **canonical Scenario execution** — the path documented for proof runs and Proof Library acceptance — not to every unit or integration test file.

### Application Survival Test

**Mandatory Scenario acceptance question:**

> If proof infrastructure, evaluator, evidence packaging, and report generation are removed, does a useful autonomous application component remain that still solves the underlying problem?

| Answer | Effect |
|--------|--------|
| **YES** | Required to qualify as SCENARIO |
| **NO** | **Not acceptable** as SCENARIO — may qualify as CONFORMANCE or requires redesign |

### Scenario runtime LAB profile (generated proofs)

Generated scenario application skeletons use `build_scenario_lab_runtime` from `intergrax.applications._shared.scenario_runtime_profiles`. Authors of ordinary proof runs **do not** manually configure `runtime_events_db_path`, `trace_db_path`, `use_in_memory_trace`, `require_runtime_event_persistence`, or diagnostic storage rules. LAB provides automatic scoped workspace storage, explicit synthetic tenant, the same Nexus baseline, RuntimeEvent persistence, and default diagnostics via shared `InMemoryDocumentStore`. Production-attached deployment requires explicit manifest, tenant, durable storage, and document store when diagnostics are required.

### Application Observability Test

**Mandatory Scenario acceptance question** (alongside Application Survival Test):

> If the proof evaluator, evidence packaging, and HTML report are removed, does the application/runtime still produce enough structured execution information to reconstruct its material decisions, actions, observations, challenges, recoveries, diagnostics, and terminal result?

| Answer | Effect |
|--------|--------|
| **YES** | Required to qualify as SCENARIO |
| **NO** | **Not acceptable** as SCENARIO — Proof cannot be the sole recorder |

**Ownership rule:**

| APPLICATION / PLATFORM | PROOF |
|------------------------|-------|
| emits canonical structured execution trace | validates, projects, packages, renders |
| autonomous decision trace | falsification |
| tool/action provenance | invariant verification |
| action rationale / objective | evidence projection |
| diagnostic facts | report rendering |
| claim/challenge lifecycle facts | reproduction metadata |
| terminal decision facts | |

**PROOF DOES NOT OWN:** fabricated rationale; reconstructed model intent absent from runtime artifacts; post-hoc explanation generated by another LLM.

### Mandatory observability, explainability, and diagnostics

**Non-negotiable rule:**

> A Scenario Proof **MUST NOT** be a black box. Every material autonomous decision, external action, evidence acquisition, claim transition, challenge, recovery action, and terminal outcome **MUST** be reconstructable from structured artifacts emitted by the canonical application/runtime path.

**Explainability rule:**

> Explainability **MUST** use explicit decision summaries, objectives, evidence references, selected actions, and bounded action rationales. Hidden chain-of-thought is neither required nor accepted as an observability artifact.

**Proof/report rule:**

> The Proof/report layer **MUST** consume application/platform observability. It **MUST NOT** invent execution explanations that were not emitted by the canonical application path.

The Proof layer may **VERIFY**, **PROJECT**, **PACKAGE**, and **RENDER** — it **MUST NOT** reconstruct or invent missing execution reasoning post hoc.

#### Three mandatory pillars

Every accepted executable **SCENARIO** with material autonomous behavior **MUST** satisfy all three pillars below. Do not require irrelevant event classes for every Scenario — the requirement is that all **material** workflow decisions and actions are reconstructable.

**A. Observability — what happened**

Material events **MUST** be reconstructable where applicable, including when relevant to the Scenario:

- autonomous/model/planner round
- selected action/tool
- validated action arguments/scope
- actual tool execution and outcome
- evidence created/acquired
- policy denial; approval/HITL state
- claim proposal and resolution transition
- challenge creation and resolution
- revision/recovery
- budget state / limit reached
- completion proposal; terminal gate
- operational failure

Reuse canonical production observability owners: `RuntimeState.trace_events`, `RuntimeState.trace_event(...)`, `ObservabilityEmitter`, typed `TraceEvent`, typed `DiagnosticPayload` (with schema/version and `redact()`), `ToolCallTrace`, `RuntimeState.tool_traces`, execution identity/sequencing, and runtime/planner/critic components. Do **not** introduce a Proof-only event bus or parallel logging system.

**B. Explainability — why it acted**

Material autonomous decisions **MUST** expose operator-facing rationale — **not** chain-of-thought. Acceptable structured semantics include: decision objective; action purpose; evidence basis; selected action/tool; expected information need; concise decision rationale; challenged claim / reason for revision. Scenario applications may own typed domain-specific `DiagnosticPayload` implementations inside the `TraceEvent` envelope. Do **not** mandate a premature universal decision DTO unless cross-domain invariants justify it.

**C. Diagnostics — why it failed / stopped**

Material failure/stop paths **MUST** emit structured diagnostic evidence, including when applicable: model/provider failure; malformed structured response; invalid action/tool request; scope validation failure; policy denial; tool failure; retry/recovery; budget exhaustion; missing evidence; unsupported claim; Critic rejection; terminal rejection. Distinguish **operational failure** from **epistemic unresolved outcome** — do not narrate operational failure as epistemic uncertainty.

#### Chain-of-thought hard rule

Scenario observability **MUST NOT** require, persist, expose, or publish hidden/private model chain-of-thought. Explainability relies on observable structured outputs: decision summaries; action objectives; evidence references; action rationales; claim proposals; challenges; tool calls; outcomes; diagnostics; terminal reasons. Do **not** encourage prompts such as “show your full reasoning”, “write every thought”, or “dump chain of thought”.

#### Model / tool correlation

For autonomous Scenario applications, acceptance **MUST** require correlation between model/planner decision and actual runtime action:

```text
declared selected tool/action
→ corresponding ToolCallTrace / runtime event
→ resulting observation
→ evidence ref (when material)
```

Model prose alone is **not** proof of action.

#### Challenge / revision correlation

When Critic/verifier/reviewer lifecycle exists: challenge **MUST** reference the challenged claim/action where applicable; revision/recovery **MUST** be traceably linked to the challenge; final accepted state **MUST** show whether the challenge was satisfied, rejected, or superseded.

#### Redaction / security

Observability must be production-safe: no secrets or credentials; no unrestricted prompt dumps; no raw private chain-of-thought; bounded output previews where appropriate; use `DiagnosticPayload.redact()` or equivalent platform redaction. Public report projection may be stricter than internal runtime diagnostics. Observability ≠ dumping everything.

#### Machine-readable execution artifact

For an **accepted executable SCENARIO** with material autonomous behavior, a machine-readable observability/decision provenance artifact **MUST** be available to the proof/report pipeline. **Reuse** existing projection where possible:

- runtime `TraceEvent` / `ToolCallTrace` export
- typed `PlatformProofEvidence` (`intergrax.platform_proof_evidence.v3`) — especially `scenarios[].steps`, `evidence_graph`, evaluator/failure fields
- report evidence structure per [PLATFORM_PROOF_REPORT_STANDARD.md](../docs/project/proofs/PLATFORM_PROOF_REPORT_STANDARD.md)

Do **not** require `decision-trace.json` or a new schema unless an audit proves existing contracts cannot serve the role. Design-stage packages do not need runtime artifacts. Do **not** break existing CONFORMANCE packages.

#### Design-stage declaration

Before implementation, every new Scenario design **MUST** declare: material autonomous decisions; observability coverage; explainability coverage; diagnostics coverage; redaction policy; expected machine-readable execution artifact; report projection plan. See scaffold § Observability / Explainability / Diagnostics Contract.

### Canonical Scenario execution path

**Non-negotiable rule:**

> The canonical Scenario Proof execution **MUST** exercise the same production-capable application path intended for real deployment.

| Forbidden | Required |
|-----------|----------|
| production application → real path; proof → separate fake/scripted path | same application core → controlled/synthetic provider implementation for proof **or** → production provider implementation for deployment |

The proof may swap controlled data providers. It **MUST NOT** swap the application itself for a test harness.

### No proof-only application logic

Canonical Scenario execution **MUST NOT** contain business-decision logic that exists only to make the proof pass.

**Prohibited in canonical path** (examples):

- proof-only diagnosis engine
- proof-only routing
- hardcoded expected outcome
- direct variant-to-answer mapping
- scripted evidence sequence pretending to be autonomous investigation
- proof harness deciding instead of application core
- proof-specific bypass of normal runtime/provider contracts

Proof code **may**: configure; invoke; observe; falsify; evaluate; package evidence; render.

Proof code **MUST NOT** own the actual application intelligence/workflow.

### Fake / mock policy (Scenario canonical path)

| Context | Fake/mock policy |
|---------|------------------|
| **Scenario canonical application path** | **PROHIBITED** for material app/runtime behavior |
| **Scenario unit/integration tests** | Allowed |
| **Conformance proof** | Allowed only **outside** claimed boundary |
| **Mechanism under proof** | **Never** fake |

**Prohibited in canonical Scenario application execution** (minimum):

- `FakeLLMAdapter` or equivalent fake model adapter
- scripted model response
- fake model provider
- `Mock` / `MagicMock` in production application path
- `testing_support` imports in canonical Scenario application path
- test-only session/runtime manager replacing normal application dependency
- proof-only runtime adapter
- fake decision engine
- hardcoded agent output
- direct provider execution bypassing normal Intergrax runtime contract

A future author receiving only this guide and a problem description **MUST NOT** reasonably conclude that a fake model is acceptable because "the LLM itself is not the platform mechanism under proof" when AI/agent behavior is material to the Scenario claim.

### Synthetic data policy

> **Synthetic data is allowed. Fake application behavior is not.**

| Allowed | Forbidden |
|---------|-----------|
| synthetic organization; fictional incidents; deterministic datasets; controlled local provider implementation; synthetic telemetry; synthetic documents; synthetic API service — **provided they enter through normal production-capable provider/tool contracts** | returning pre-written final answer instead of exercising application logic; fixture directly deciding outcome; fake model output replacing AI behavior when AI behavior is part of the Scenario claim |

### Real AI / model boundary

If the Scenario's public problem/claim materially depends on AI/agent behavior, canonical Scenario execution **MUST** use a real configured model/provider boundary through normal Intergrax runtime mechanisms.

Examples requiring real model/provider in canonical execution:

- "AI investigates incident"
- "AI analyzes contract"
- "AI evaluates vendor risk"

A fake/scripted model **MAY** be used only in tests. If model behavior is irrelevant to the claim, reconsider whether the entry should be **CONFORMANCE** instead of an AI-framed **SCENARIO**. Provider-neutral wording — no specific vendor is required.

### Bounded autonomy

The application **MUST** autonomously perform the meaningful workflow within its declared scope. For an investigative application this may include: forming candidate hypotheses; deciding what evidence to request; deciding whether evidence is sufficient; revising a conclusion; refusing unsupported completion.

Governance may constrain allowed actions/outcomes. The proof harness **MUST NOT** manually orchestrate business reasoning while claiming the application/AI performed it. Do not demand unlimited autonomy — use **bounded autonomy**.

**Design principle:** real autonomous model/application behavior **plus** platform-enforced admissibility/governance/evidence boundaries. The model may propose hypotheses, interpretation, tool requests, and candidate decisions. Platform mechanisms may enforce allowed tools, evidence requirements, policy, challenge, terminal state, approvals, and bounded recovery.

### Provider abstraction / replaceability

Controlled proof inputs **MUST** be provided through the same typed provider/tool contracts that real deployment can use. Both controlled/synthetic and production implementations **MUST** satisfy the same application-facing contract. The normative requirement is **replaceability without rewriting application core** — not any specific class name.

### Platform reuse (Scenario applications)

**Proof-local clones of missing platform capabilities are forbidden.**

A Scenario application **MUST** consume normal reusable Intergrax mechanisms where the capability is generic. If a generic capability is missing:

```text
scenario
→ STOP
→ platform gap analysis
→ reusable platform implementation
→ verification
→ resume scenario
```

Do **not** build local substitutes for LLM/runtime lifecycle, ToolRuntime, retry/recovery, state/persistence, approvals, governance, evidence storage/contracts, Critic lifecycle, observability, or provider lifecycle when an Intergrax owner exists or should exist.

### External boundaries vs production-capability

**production-capable** does **not** mean every external boundary must be live. A Scenario **MAY** use controlled local/synthetic provider implementations when the external system itself is not the claim (e.g. `SyntheticTelemetryProvider` through the same typed contract for epistemic investigation behavior).

If the claim explicitly includes "works against provider X", "real database semantics", or "real external SaaS behavior", the relevant real boundary **MUST** be exercised.

### No hidden test infrastructure in application core

Canonical Scenario application modules **MUST NOT** depend on: `tests/`; `testing_support`; `unittest.mock`; `pytest`; fixtures whose role is to emulate application behavior rather than provide data. Unit tests, proof runner tests, and evaluator tests **MAY** depend on those.

### Stability from invariants, not fake model outputs

Scenario reproducibility **MUST NOT** be achieved by replacing the real AI boundary with deterministic fake model output. Separate **semantic/model variability** from **deterministic system invariants**. A valid Scenario may tolerate model variability while requiring deterministic platform guarantees (e.g. unsupported final decision rejected; required evidence present; forbidden action does not occur; missing decisive evidence cannot become SUPPORTED; terminal state contract holds).

### Ready-to-use component expectation

A Scenario application **SHOULD** be usable as a component immediately after normal configuration of its required provider/model dependencies.

| Not required | Required |
|--------------|----------|
| polished end-user UI; commercial packaging; real-user validation; every production integration | real application core; production-capable contracts; normal runtime path; bounded configuration surface; no fake/test-only canonical dependencies |

---

## Proof Library philosophy

> **The Intergrax Proof Library is not a catalog of Intergrax features. It is a catalog of difficult real-world AI system problems that Intergrax can solve and falsifiably demonstrate.**

> **Scenario selection precedes mechanism selection.**

> **A missing platform capability does not automatically invalidate a strong scenario. If the capability is reusable and architecturally justified, implement it in Intergrax first and then resume the Scenario Proof.**

> **One independent author session owns one Scenario Proof from qualification through accepted evidence.**

Every Scenario Proof session is a **falsification attempt** against a real problem — not a feature demo, marketing slide, or architecture tour.

---

## One session owns one Scenario Proof

**Normative rule:**

```text
One independent Scenario Proof author session owns exactly one Scenario Proof
from scenario qualification through accepted evidence/publication.
```

A session **may** inspect:

- relevant architecture;
- reusable platform components;
- existing proofs;
- tests and contracts;

—but its implementation and publication **ownership** remains **one scenario**.

A single session **MUST NOT**:

- design multiple Scenario Proofs in parallel;
- opportunistically modify unrelated Scenario Proofs;
- broaden into general Proof Library redesign.

Conformance proof work follows the same shared-development rules but is outside this one-session-one-scenario contract unless the operator explicitly scopes otherwise.

---

## Canonical Scenario Lifecycle

Scenario Proof authoring is a **two-stage lifecycle** with **two canonical commands**. Do not collapse them.

```text
IDEA
↓
create_scenario_proof.py
↓
DESIGN / NOT YET ACCEPTED
↓
README.md + SCENARIO_SPEC.md
↓
A/B/C/D/E DESIGN
↓
Scenario Quality Gate
↓
ACCEPTED FOR IMPLEMENTATION
↓
init_scenario_implementation.py
↓
platform-native implementation skeleton
↓
IMPLEMENTATION
↓
TEST
↓
REAL PROOF EXECUTION
↓
EVIDENCE / REPORT
↓
LIBRARY ACCEPTANCE
```

### Do not skip stages

```text
create_scenario_proof.py ≠ init_scenario_implementation.py
```

```text
DESIGN ≠ ACCEPTED FOR IMPLEMENTATION ≠ EXECUTABLE ≠ PROOF ACCEPTED
```

A future session **MUST NOT** infer:

- “folder exists, so implementation may start”; or
- “README + SCENARIO_SPEC exist, so the scaffold is complete.”

Existence of `platform_proofs/scenarios/<slug>/` means only that a design package may exist — not that implementation is authorized or that runtime artifacts are present.

### Phase 1 — Create design package

**Canonical command:**

```bash
uv run python scripts/proof/create_scenario_proof.py \
  --slug <slug> \
  --title "<title>"
```

**Result:**

```text
platform_proofs/scenarios/<slug>/
├── README.md
└── SCENARIO_SPEC.md
```

**Lifecycle after this command:**

```text
DESIGN / NOT YET ACCEPTED
```

> **This command creates a design package only.**

It does **not** mean the scenario is:

- implementation-ready;
- executable;
- accepted; or
- qualified.

It **MUST NOT** generate runtime or proof implementation artifacts (`proof.json`, `run_proof.py`, `application/`, `proof/`, `fixtures/`, `.env.example`, evaluator/evidence modules, or other design-stage forbidden artifacts).

`SCENARIO_SPEC.md` includes YAML frontmatter — the **canonical machine-readable lifecycle source**. README keeps human-readable status wording.

### Phase 2 — Design the scenario

After the design package exists, complete the deep contract in `README.md` and `SCENARIO_SPEC.md`:

```text
A. SCENARIO
B. SOLUTION
C. INTERGRAX FIT
D. GAP DECISION
E. PROOF BUILD
```

Before the implementation initializer, the session **MUST** resolve at minimum:

- real problem; user / operator; stakes; failure consequences;
- naive failure; WOW factor; Skeptic Challenge; adversarial conditions;
- Application Survival Test; Application Observability Test;
- observability / explainability / diagnostics contract;
- APPLICATION vs PROOF HARNESS separation;
- bounded claim; PASS; FAIL; excluded claims; limitations;
- Intergrax Fit; missing-capability / gap decisions;
- proof build plan (§ E — workflow only at this stage; no implementation yet).

This phase is documentation and design only — see § Mandatory session conversation format and § Five-stage Scenario Proof session model (Stages 1–4).

### Phase 3 — Scenario Quality Gate

After honest design work, the session records an explicit gate decision:

| Outcome | Meaning |
|---------|---------|
| **REJECT / REDESIGN** | Scenario too weak, dishonest, or incomplete — return to Phase 2 |
| **ACCEPTED FOR IMPLEMENTATION** | Scenario concept accepted — **implementation stage may begin** |

**Hard rule:**

> `init_scenario_implementation.py` **MUST NOT** be used while the Scenario lifecycle is `DESIGN / NOT YET ACCEPTED`.

**Acceptance for implementation does not mean:**

- proof PASS;
- executable qualification; or
- public Proof Library acceptance.

It means only: **permission to initialize the implementation skeleton and begin Phase 4+**.

Record acceptance in `SCENARIO_SPEC.md` frontmatter (not README prose alone). Example after gate pass:

```yaml
---
scenario_slug: <scenario_slug>
lifecycle: ACCEPTED_FOR_IMPLEMENTATION
implementation_status: NOT_INITIALIZED
intergrax_fit: COMPLETED
gap_decision: RESOLVED
observability_contract: COMPLETED
application_vs_proof_ownership: COMPLETED
---
```

### Phase 4 — Initialize implementation scaffold

**Only after** `ACCEPTED FOR IMPLEMENTATION` and completed frontmatter gates.

**Canonical command:**

```bash
uv run python scripts/proof/init_scenario_implementation.py \
  --slug <slug>
```

This command creates the **platform-native implementation skeleton**. Authors **MUST NOT** manually recreate the structure owned by `init_scenario_implementation.py`.

The generator is fail-closed: it does not create a design package, does not overwrite existing implementation files, and updates frontmatter to `lifecycle: IMPLEMENTATION_INITIALIZED` / `implementation_status: INITIALIZED` on success.

> **The initializer is the source of truth for the exact current generated file set; documentation describes architectural responsibilities and lifecycle.**

**Architectural shape** (representative — see initializer for the authoritative file list):

```text
platform_proofs/scenarios/<slug>/
├── README.md
├── SCENARIO_SPEC.md
│
├── application/
│   ├── __init__.py
│   ├── runtime_composition.py   # build_scenario_lab_runtime + agent registration
│   ├── scenario.py              # application execution entry
│   ├── agent.py                 # domain agent / workflow skeleton
│   ├── observability.py         # domain DiagnosticPayload / observability seams
│   └── tools.py                 # platform tool bindings
│
├── proof/
│   ├── __init__.py
│   ├── evaluator.py             # falsification assertions
│   └── evidence_builder.py      # evidence projection
│
├── fixtures/
│   └── __init__.py              # controlled external data when required
│
├── proof.json
├── run_proof.py
└── .env.example
```

#### Directory roles

| Path | Role |
|------|------|
| **`application/`** | Production-capable application core — runtime composition, execution entry, domain agent/workflow, platform tool bindings, observability seams. **MUST NOT** import proof evaluator/report code. |
| **`proof/`** | Proof-owned layer — evaluator, evidence projection, falsification assertions, proof-only packaging. **MUST NOT** own application decision logic. |
| **`fixtures/`** | Controlled external data / synthetic providers / scenario-owned fixtures when required. **MUST NOT** replace application business logic with fake decision engines. |
| **`proof.json`** | Machine-readable proof package descriptor (`intergrax.platform_proof_descriptor.v3`). |
| **`run_proof.py`** | Thin proof-owned execution entrypoint — configure, invoke application, evaluate, write artifacts. |
| **`.env.example`** | Documented configuration surface (no secrets committed). |

#### Runtime baseline

The generated `application/runtime_composition.py` uses the shared scenario runtime baseline (`build_scenario_lab_runtime` from `intergrax.applications._shared.scenario_runtime_profiles`) so authors do not rediscover standard execution identity, Nexus/runtime composition, storage, and baseline observability wiring. See § Scenario runtime LAB profile (generated proofs).

### Phases 5–10 — Implement through library acceptance

After initialization:

```text
IMPLEMENTATION → TARGETED TESTING → REAL PROOF EXECUTION
→ EVIDENCE / REPORT VERIFICATION → LIBRARY ACCEPTANCE
```

Technical contracts for descriptor, execution, evidence, and report remain in the sections below (§ Technical Proof Library lifecycle onward). Do not treat initialization as proof qualification.

---

## Two-layer working model

Every Scenario Proof session operates in two conceptual layers. **Do not collapse them.**

### LAYER 1 — SCENARIO / PROBLEM

Before discussing Intergrax implementation, determine whether the real-world scenario is strong enough to deserve entry in the Proof Library.

Focus on:

- real problem;
- real user / operator / business pain;
- failure consequences;
- why the problem is difficult for AI systems;
- adversarial conditions;
- skeptical-engineer challenge;
- public “wow” value.

**Do not begin mechanism selection before the scenario itself is understood and accepted.**

### LAYER 2 — SOLUTION / IMPLEMENTATION

Only after scenario quality is accepted:

1. design how the system should solve the problem;
2. define desired behavior and guarantees;
3. map required Intergrax mechanisms;
4. perform capability-fit / gap analysis;
5. implement reusable missing capability if justified;
6. package / run / evaluate / report under the canonical Proof Library contract (see § Technical Proof Library lifecycle).

---

## Five-stage Scenario Proof session model

Every Scenario Proof session **MUST** explicitly work through these stages **in order**. Do not skip stages. Do not implement before earlier stages are honest.

```text
STAGE 1 — Qualify and strengthen the scenario
STAGE 2 — Design the scenario solution and proof semantics
STAGE 3 — Map required Intergrax mechanisms
STAGE 4 — Missing capability decision
STAGE 5 — Apply canonical Proof Library engine
```

Conformance proofs follow the same engineering discipline in Stages 3–5 but may omit problem-first public framing where mechanism verification is the sole purpose.

### Stage 1 — Qualify and strengthen the scenario

Discuss the scenario in **plain problem language** before any implementation.

**Required questions:**

- What is the concrete real-world problem?
- Who suffers from it?
- What happens if the AI / system gets it wrong?
- Why is this not a toy problem?
- What makes it difficult?
- What is the naive / simple solution likely to get wrong?
- What adversarial situation exposes that weakness?
- What makes the result interesting even to someone who does not know Intergrax?

The session **must actively strengthen** weak scenario proposals rather than immediately implement them.

#### Scenario Quality Gate

A Scenario Proof should satisfy **most** of these:

- real operational / business / engineering pain exists;
- failure has meaningful consequences;
- problem was not invented merely to exercise an Intergrax feature;
- scenario contains uncertainty, conflict, failure, recovery, temporal complexity, governance, side-effect risk, adversarial input, or another meaningful systems challenge;
- negative / adversarial falsification is possible;
- outcome can be evaluated;
- story is understandable without reading Intergrax internals;
- result demonstrates more than basic LLM orchestration;
- public demonstration has a credible “this is a system, not a chatbot” effect;
- **Application Survival Test** passes (see § Application Survival Test);
- **Application Observability Test** passes (see § Application Observability Test);
- material autonomous execution is reconstructable from production-path observability (see § Mandatory observability, explainability, and diagnostics);
- plausible real operator/user exists for the application component;
- canonical application path is production-capable;
- AI framing requires real model boundary when material to the claim;
- autonomy is genuine rather than proof-scripted;
- controlled data sources are replaceable through normal provider/tool contracts.

**Weak scenario examples that require strengthening** (capabilities may appear inside a strong scenario, but are not strong scenarios by themselves):

```text
AI answers a question from RAG.
AI uses memory.
AI calls several tools.
Two agents discuss an answer.
AI summarizes documents.
```

#### Skeptic Challenge

Every Scenario Proof **must** explicitly answer:

> Could a skeptical Staff / Principal Engineer reasonably say:
>
> “I can build the same thing with an LLM + memory + RAG + a few LangGraph nodes in 10 minutes”?

**If YES** — the scenario is currently too weak. **Do not** automatically discard the underlying real problem. First attempt to strengthen it by introducing the actual difficult guarantees naturally required by the real problem, such as:

- conflicting evidence;
- missing evidence;
- stale evidence;
- temporal reconstruction;
- independently verified decisions;
- bounded recovery;
- durable state;
- exactly-once / duplicate side-effect protection;
- policy enforcement;
- human approval;
- adversarial input;
- auditability;
- evidence provenance;
- multi-step evidence dependency;
- interruption / resume;
- disagreement requiring further evidence;
- explicit unresolved outcome when certainty is impossible.

Do **not** add complexity arbitrarily.

**If NO** — document why the scenario requires guarantees beyond trivial orchestration.

#### WOW criterion

```text
WOW does not mean “more agents” or “more infrastructure”.
WOW means a difficult, credible guarantee is visibly achieved and evidenced.
```

Reject feature stuffing such as adding many agents, many databases, Kafka, RAG, Memory, Critic, HITL, or Governance **solely** to make a proof look impressive.

The scenario should remain the **smallest credible system** that demonstrates the difficult guarantee.

**Strong WOW example:**

```text
An external refund succeeds.
The process crashes before local completion is recorded.
Execution resumes.
The system proves that a duplicate refund did not occur and preserves an auditable recovery history.
```

This can be more impressive than a seven-agent demo.

#### Scenario rejection criteria

Reject or redesign a scenario if, after strengthening attempts:

- there is no meaningful real-world pain;
- failure is inconsequential;
- PASS is subjective prose only;
- no credible falsification exists;
- the model is directly given the expected answer / ground truth;
- the proof is only a feature demo;
- simple orchestration genuinely provides equivalent guarantees;
- complexity exists only for visual impression;
- the scenario cannot be reproduced honestly;
- the claim would require overstating evidence;
- **Application Survival Test** fails;
- **Application Observability Test** fails;
- material autonomous action lacks provenance;
- model-selected action cannot be correlated to actual execution;
- material decision has no explicit bounded rationale/objective;
- claim cannot be linked to evidence;
- challenge cannot be linked to revision;
- terminal state cannot be reconstructed;
- failure path lacks diagnostic reason;
- report invents explanation post hoc;
- Proof-specific logger is the only source of execution trace;
- hidden chain-of-thought is required or collected as evidence;
- explanation is generated after execution by an unrelated LLM;
- canonical path uses proof-only application logic or fake/test-only canonical dependencies.

Also reject proofs whose primary motivation is:

- “show that tool calling works”
- “show that RAG works”
- marketing demos
- hello-world agents
- arbitrary multi-hop requirements with no real failure risk

### Stage 2 — Design the scenario solution and proof semantics

Once the Scenario Quality Gate passes, design the solution in **plain language** before discussing APIs, classes, or package layout.

**Required story structure:**

```text
WHAT HAPPENS
→ WHAT THE SYSTEM RECEIVES
→ WHAT IT MUST DETERMINE / ACHIEVE
→ WHAT CAN GO WRONG
→ WHAT THE SYSTEM DOES
→ HOW IT REACTS TO FAILURE / CONFLICT
→ WHAT FINAL RESULT IS ACCEPTABLE
```

Where useful, use:

```text
INPUT
→ ACTION
→ OBSERVATION
→ DECISION
→ CHALLENGE / FAILURE
→ RECOVERY / VERIFICATION
→ FINAL RESULT
```

Do not force phases that do not naturally apply to the scenario.

#### APPLICATION vs PROOF HARNESS (mandatory separation)

Before implementation, document explicit ownership:

| APPLICATION / PLATFORM OWNS | PROOF OWNS |
|-------------------------------|------------|
| business workflow | controlled adversarial input configuration |
| autonomous reasoning / decision flow | evaluator |
| runtime execution trace | falsification assertions |
| autonomous decision trace | invariant verification |
| tool/action provenance | canonical evidence projection |
| action rationale / objective | report rendering |
| diagnostic facts | reproduction metadata |
| claim/challenge lifecycle facts | |
| terminal decision facts | |
| provider / tool consumption | |
| production configuration surface | |
| domain output | |

**PROOF DOES NOT OWN:** fabricated rationale; reconstructed model intent not present in runtime artifacts; post-hoc explanation generated by another LLM.

This separation is **REQUIRED** before implementation. The proof harness **MUST NOT** own business reasoning while claiming the application/AI performed it.

#### Proof semantics

Before mechanism selection, define:

| Item | Required |
|------|----------|
| **Problem** | The real pain |
| **Risk** | What harmful / incorrect system behavior is being tested |
| **Desired outcome** | What correct system behavior looks like |
| **Claim** | One bounded falsifiable claim |
| **PASS** | Explicit conditions required for success — prefer machine-checkable invariants |
| **FAIL** | Explicit conditions that invalidate the claim |
| **Adversarial cases** | Concrete attempts to break the claim |
| **Excluded claims** | What the scenario does not prove |
| **Limitations** | Known boundaries |

### Stage 3 — Map required Intergrax mechanisms

Only after the problem and solution story are clear, determine technical realization.

Create a mapping:

```text
APPLICATION NEED
→ PLATFORM MECHANISM
→ CURRENT PLATFORM OWNER
→ STATUS
```

Also required for each need:

| Audit dimension | Values |
|-----------------|--------|
| **TEST-ONLY SUBSTITUTE PRESENT?** | YES / NO — if **YES** in canonical Scenario path: **BLOCKER** |

Legacy equivalent (still valid for guarantee-centric discussion):

```text
REQUIRED GUARANTEE
→ REQUIRED CAPABILITY
→ EXISTING INTERGRAX COMPONENT
→ STATUS
```

Statuses:

- **AVAILABLE**
- **AVAILABLE BUT NEEDS WIRING**
- **MISSING**

Examples of mechanisms — **only when naturally required**:

- bounded execution
- tool runtime
- durable state
- persistence
- recovery
- idempotency
- side-effect coordination
- Critic / verification
- HITL
- governance
- policy enforcement
- bitemporal knowledge
- RAG
- Memory
- observability
- Unified Run Journal
- evidence identity / dependencies
- multi-agent coordination
- budget enforcement
- security / adversarial input handling

```text
Problem chooses mechanisms.
Mechanisms do not choose the problem.
```

**Do not** start from platform mechanisms and invent a scenario around them. Never add components solely to make the proof appear sophisticated.

Verify reuse at current repository HEAD before claiming **AVAILABLE**.

### Stage 4 — Missing capability decision

A **MISSING** capability does **not** automatically invalidate a strong scenario.

When a required capability is **MISSING**, determine:

1. Is the requirement genuinely necessary for the real scenario?
2. Is the missing capability reusable beyond this one proof?
3. Does it belong naturally in Intergrax architecture?
4. Can it be implemented cleanly as a typed reusable platform mechanism?
5. Would implementing it materially strengthen Intergrax?

**If YES to the above:**

- **STOP** Scenario implementation temporarily;
- define the architecture gap;
- prepare / approve a bounded platform implementation task;
- implement and independently verify the reusable capability;
- return to the Scenario Proof.

**If NO:**

- do **not** create proof-local fake infrastructure;
- redesign the scenario realization, narrow the claim, or reject the scenario if the guarantee cannot honestly be demonstrated.

```text
The Proof Library is also a discovery mechanism for meaningful platform gaps.
```

**Proof-local clones of missing platform capabilities are forbidden.**

#### STOP ≠ abandon scenario

**STOP** means:

> do not improvise or create proof-local platform substitutes.

It does **not** necessarily mean:

> abandon the Scenario Proof.

For strong scenarios, STOP may trigger:

```text
scenario
→ architecture gap
→ reusable platform enhancement
→ verification
→ resume scenario
```

This distinction is mandatory. Report the gap; do not silently weaken the scenario or fork a proof-local substitute.

### Stage 5 — Apply canonical Proof Library engine

Only after:

- scenario accepted;
- solution designed;
- APPLICATION vs PROOF HARNESS separation documented;
- claim defined;
- mechanisms mapped;
- gaps resolved;
- author confirms:
  - production-capable application exists;
  - canonical application path contains no prohibited fake/test shortcuts;
  - controlled providers use normal application contracts;
  - real model boundary is configured if material to claim;
  - application can execute independently of evaluator/report layer;
  - **Application Observability Test** passes;
  - observability / explainability / diagnostics contract documented in SCENARIO_SPEC;

proceed to the **technical pipeline** defined in § Technical Proof Library lifecycle and the sections that follow (package structure, descriptor, execution, evidence, report, acceptance).

```text
proof package
→ proof.json
→ .env.example
→ optional proof-owned Compose / fixtures
→ run_proof.py
→ targeted tests
→ real execution
→ typed evidence
→ proof-result.json
→ report.html
→ manual report / evidence audit
→ Proof Library acceptance
```

Do not duplicate those technical contracts here — they remain canonical in their existing sections.

---

## Mandatory session conversation format

Structure user-facing discussion in this order:

### A. SCENARIO

Discuss:

- REAL PROBLEM
- WHY IT MATTERS
- FAILURE CONSEQUENCES
- WOW FACTOR
- SKEPTIC CHALLENGE
- ADVERSARIAL CONDITIONS

**No implementation yet.**

### B. SOLUTION

Discuss:

- DESIRED BEHAVIOR
- STEP-BY-STEP STORY
- GUARANTEES
- CLAIM
- PASS
- FAIL
- ADVERSARIAL ATTACKS
- EXCLUDED CLAIMS

### C. INTERGRAX FIT

Present a clear matrix:

```text
APPLICATION NEED
→ PLATFORM MECHANISM
→ CURRENT PLATFORM OWNER
→ STATUS
```

Also audit **TEST-ONLY SUBSTITUTE PRESENT?** — **YES** in canonical Scenario path is a **BLOCKER**.

### D. GAP DECISION

For every **MISSING** item:

- is it necessary?
- is it reusable?
- should Intergrax implement it?
- what architectural owner should own it?

**Stop for approval** before implementation when a new reusable platform capability is required.

### E. PROOF BUILD

Only then proceed:

- implementation;
- tests;
- real run;
- evidence;
- report;
- publication.

---

## User-facing explanation requirement

Explain each stage first in **plain user / problem language**. Technical detail comes afterward.

A session **must always** make clear:

- where it currently is in the scenario roadmap;
- what is being decided now;
- why that decision matters;
- whether the scenario is becoming stronger or weaker;
- whether the final public Proof Library objective is still preserved.

Avoid unnecessary implementation jargon during scenario design (Stages 1–4).

---

## Technical Proof Library lifecycle

After Stages 1–4 are complete, every Scenario Proof session **MUST** follow this strict technical sequence. Do not skip phases.

```text
4. PROOF DESIGN
5. IMPLEMENTATION
6. TARGETED TESTING
7. REAL PROOF EXECUTION
8. EVIDENCE / REPORT VERIFICATION
9. PUBLICATION / LIBRARY ACCEPTANCE
10. CLOSE AND MOVE TO NEXT SCENARIO
```

| Phase | Action |
|-------|--------|
| **4. Proof design** | Positive and adversarial paths; real boundaries; evaluator contract; artifact plan |
| **5. Implementation** | Smallest **production-capable application component** plus the smallest proof layer needed to falsify and evidence the claim; reuse platform mechanisms |
| **6. Targeted testing** | Deterministic unit/integration gates for proof-owned logic |
| **7. Real proof execution** | Actual run with real boundaries (not dry-run substitute) |
| **8. Evidence / report verification** | Typed evidence validates; report manually audited |
| **9. Publication / library acceptance** | All acceptance gates pass (see § Public Library acceptance gate) |
| **10. Close** | Finalize scenario package documentation; move to next scenario |

---

## Self-contained executable proof package

> A Proof Library entry is a **self-contained executable proof package** inside the cloned Intergrax repository. For **SCENARIO**, the package **MAY** contain application component, provider implementations/config, proof entrypoint, and evaluator/evidence/report code — but **application code MUST remain separable from proof-only code**. A package directory does **not** imply all code is proof harness.

### Self-contained does NOT mean

- a separate repository
- a copied Intergrax runtime
- copied Critic / RAG / Observability / etc.
- an independent application framework

### Self-contained DOES mean

> The proof can be configured, started, validated, executed, evaluated, reported, and cleaned up **independently** of other proofs and undocumented repository state.

**Mandatory acceptance invariant:**

```text
A proof is not accepted into the Proof Library unless it can be executed
independently from its own package using documented configuration, without
relying on another proof's infrastructure or undocumented local state.
```

The **cloned Intergrax repository** is the supported execution boundary. Do **not** require that copying only the proof directory outside the repository must work.

---

## Package structure

### Scenario proofs — two-stage canonical shape

Scenario packages follow the **two-stage lifecycle** (§ Canonical Scenario Lifecycle). Do not use a flat root-level `evaluator.py` / `scenario.py` layout for new Scenario proofs.

#### Design stage

Created **only** by `create_scenario_proof.py`:

```text
platform_proofs/scenarios/<scenario_slug>/

├── README.md              # public gateway (~3–5 min read)
├── SCENARIO_SPEC.md       # deep canonical contract (A/B/C/D/E) + lifecycle frontmatter
└── assets/                # optional — after Scenario Quality Gate
```

Design stage **MUST NOT** include implementation artifacts (`proof.json`, `run_proof.py`, `application/`, `proof/`, `fixtures/`, `.env.example`, root-level evaluator/evidence modules, or other forbidden design-stage files).

#### Accepted implementation stage

After `ACCEPTED FOR IMPLEMENTATION`, initialize **only** via `init_scenario_implementation.py` (§ Phase 4 — Initialize implementation scaffold). The initializer emits the platform-native layout (`application/`, `proof/`, `fixtures/`, `proof.json`, `run_proof.py`, `.env.example`).

Post-initialization, authors may add scenario-specific components when genuinely required:

```text
platform_proofs/scenarios/<scenario_slug>/

├── README.md
├── SCENARIO_SPEC.md
├── application/                 # production-capable application core
├── proof/                       # evaluator, evidence projection
├── fixtures/                    # controlled data when required
├── proof.json
├── run_proof.py
├── .env.example
├── docker-compose.yml           # only if proof owns containerized infra
├── sql/                         # only if needed
└── output/                      # canonical published artifacts
```

> Authors **MUST NOT** manually recreate the implementation skeleton — use the initializer.

### Conformance proofs

Existing domain-oriented paths may remain (e.g. `platform_proofs/<domain>/<proof_slug>/`). Scenario proofs use `platform_proofs/scenarios/<scenario_slug>/`. Conformance packages may keep a flatter proof-owned layout; Scenario proofs use the platform-native two-layer structure above.

### Rules

- Do **not** create empty placeholder files or directories beyond what the canonical commands generate.
- Create only components the proof actually needs.
- `run_proof.py` is the **canonical proof-owned entrypoint**.
- `proof.json` is the **machine-readable package contract**.
- Source and runtime components remain inside the package unless genuine reusable platform code belongs in Intergrax.
- Platform mechanisms are **imported / reused** — never copied.

---

## Descriptor v3

Every proof declares a static `proof.json` with schema:

```text
intergrax.platform_proof_descriptor.v3
```

**Canonical schema source:** `scripts/proof/intergrax_platform_proof_descriptor.py` and [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) § D2. Do not duplicate full schema here.

### Required for all proofs

| Field | Role |
|-------|------|
| `library_class` | `SCENARIO` or `CONFORMANCE` |
| `domains_exercised` | Non-empty list of Intergrax domains actually exercised (no primary, owner, or ranking) |
| `proof_kind` | Proof-specific kind slug |
| `mechanisms_exercised` | Mechanisms the proof exercises |

**Principle:** The proof does not belong to a domain. It exercises domains.

### Additional for SCENARIO

| Field | Role |
|-------|------|
| `problem_category` | Problem taxonomy |
| `problem_summary` | Short problem statement |
| `failure_mode_summary` | Failure being tested |

**No descriptor v1 or v2.** No loose metadata bags. No compatibility shims. No aliases. No fallback. No legacy proof execution path. No fake placeholder domain values (`MULTI_DOMAIN`, `CROSS_DOMAIN`, `SCENARIO`, etc.).

Discovery is automatic from `proof.json` under `platform_proofs/` — no central manifest registration for descriptor-backed proofs.

### Conformance framing

```text
PROPERTY / CONTRACT UNDER TEST
→ MECHANISMS
→ DOMAINS EXERCISED
```

### Scenario framing

```text
REAL PROBLEM
→ REQUIRED GUARANTEES
→ CAPABILITY FIT
→ MECHANISMS
→ DOMAINS EXERCISED
```

Scenario **documentation and design** are **multi-domain by default**: the problem chooses required guarantees; guarantees choose mechanisms and participating domains during Intergrax Fit (§ C). When the executable package ships, declare the truthful `domains_exercised` list — one domain or several, with no ownership semantics.

---

## Standalone execution and suite execution

Every proof must support **two execution paths** using the **same underlying implementation**.

### A. Standalone

For a person exploring one proof.

**Canonical invocation context:** repository root. Scenario README commands must be valid from repository root unless an exceptional proof explicitly documents otherwise and has architectural justification.

Canonical style:

```bash
uv run python platform_proofs/scenarios/<scenario_slug>/run_proof.py
```

For existing Conformance paths, substitute the actual package path (e.g. `platform_proofs/<domain>/<proof_slug>/run_proof.py`).

Standalone execution must **not** require understanding proof-suite internals.

### B. Suite

For CI, regression, and multiple proofs. Reuse existing canonical `scripts/proof/` discovery and runner infrastructure.

```bash
uv run python scripts/proof/run-intergrax-proof-suite.py --profile full
```

**Never build a second runner.**

**Invariant:**

```text
Standalone run and suite run execute the same proof logic.
```

No separate “demo implementation.”

---

## Standard lifecycle commands

Every proof README must document only the phases relevant to it.

**Canonical lifecycle:**

```text
CONFIGURE
→ PREFLIGHT
→ INFRA UP
→ DATA / FIXTURE SETUP
→ VALIDATE
→ RUN
→ EVALUATE
→ REPORT
→ CLEANUP
```

For proofs with infrastructure, expected UX should be simple and explicit. All commands below assume **repository root** as the working directory. For containerized proof infrastructure, use an explicit Compose path — do not document ambiguous bare `docker compose up -d` as the canonical example.

```bash
cp platform_proofs/scenarios/<scenario_slug>/.env.example \
  platform_proofs/scenarios/<scenario_slug>/.env
docker compose \
  -f platform_proofs/scenarios/<scenario_slug>/docker-compose.yml \
  up -d
uv run python \
  platform_proofs/scenarios/<scenario_slug>/run_proof.py \
  --validate-only
uv run python \
  platform_proofs/scenarios/<scenario_slug>/run_proof.py
docker compose \
  -f platform_proofs/scenarios/<scenario_slug>/docker-compose.yml \
  down -v
```

Do **not** require Docker for proofs that do not need it.

---

## `--validate-only`

For proofs with meaningful environmental or data prerequisites, provide a **non-LLM** validation path:

```bash
uv run python .../run_proof.py --validate-only
```

Validate what can be validated **without** consuming provider inference:

- infrastructure reachable
- schema / data materialized
- fixtures valid
- deterministic invariants
- configuration completeness where possible

Failure must be **explicit and actionable**. Prefer:

```text
BLOCKED_ENVIRONMENT
PostgreSQL unavailable.
Run: docker compose -f platform_proofs/scenarios/<scenario_slug>/docker-compose.yml up -d
```

over raw connection stack traces as the primary UX.

Not every proof needs identical validation internals — but proofs with infra/data prerequisites should expose this path.

---

## Configuration contract

### Reuse platform configuration

Standard Intergrax configuration and provider names **MUST** be reused. Do not invent proof-specific aliases for existing platform settings.

| Bad | Good |
|-----|------|
| `MY_PROOF_OPENAI_KEY` | `OPENAI_API_KEY` (per provider) |
| `SQL_PROOF_MODEL` | `INTERGRAX_LLM_MODEL` |

See [PLATFORM_CONFIGURATION.md](../docs/project/technical/guides/PLATFORM_CONFIGURATION.md) for canonical names.

### Proof-local configuration

Proof-specific infrastructure may use proof-local variables, e.g. `PROOF_POSTGRES_PORT` or the existing `INTERGRAX_PP_*` convention where established. Names must be scoped sufficiently to avoid collisions when multiple proofs run concurrently.

---

## `.env` contract

Every proof requiring configuration **SHOULD** provide a committed `.env.example` in the proof package:

- contains **no secret values**
- documents every proof-specific setting
- safe / defaultable values receive defaults in documentation
- required secrets remain blank
- `.env` itself is **never** committed (local / gitignored)

### Precedence

Intergrax reads configuration from the **process environment**. The platform library does **not** load `.env` by itself — see [PLATFORM_CONFIGURATION.md](../docs/project/technical/guides/PLATFORM_CONFIGURATION.md).

The Proof Library defines **one deterministic** configuration contract for proof packages:

```text
process environment
> nearest .env found by walking upward from the proof package directory
> safe defaults
```

Where:

- process environment has highest priority
- `.env` must **never** overwrite an already-set process variable
- only **one** `.env` file is loaded — the nearest file on the path from the proof package up to (and including) repository root
- search never continues above repository root; `$HOME` and filesystem paths outside the repository are never scanned
- `.env.example` is committed when configuration is required
- `.env` is local / gitignored
- secrets never enter CLI args, `proof.json`, evidence, report, logs, or unsafe observability

Every configured Scenario Proof has `.env.example`. At runtime the canonical proof environment loader (`scripts/proof/intergrax_proof_environment.py`):

1. starts at the proof package directory;
2. searches upward for `.env`;
3. stops at repository root;
4. loads only the nearest `.env`;
5. never overwrites process environment.

Repository-root `.env` can therefore act as shared configuration for every proof that does not provide a nearer `.env`.

Proof implementations must call the shared loader — do **not** implement a proof-local dotenv / config loader.

---

## Secrets

Strict rules — secrets must **never**:

- appear in `proof.json`
- be committed in `.env`
- be passed through CLI arguments
- be persisted in evidence
- be rendered in report
- be emitted unredacted through observability
- be put in Docker image layers

Safe provenance is acceptable:

```text
provider=openai
model=<model>
credential_source=environment
```

Never record credential values.

---

## Proof-owned infrastructure

A proof **MAY** own Docker Compose infrastructure (PostgreSQL, Redis, Qdrant, MinIO, mock external service, etc.).

| Rule | Requirement |
|------|-------------|
| Ownership | Infrastructure belongs to the proof package |
| Isolation | Must not depend on another proof's Compose project |
| Volumes | Do not reuse another proof's volumes |
| Dev stack | Do not depend on general Intergrax dev stack unless scenario explicitly proves that integration and it is documented |
| Images | Pin important external image versions |
| Health | Add healthchecks where meaningful |
| Lifecycle | Document startup and teardown; cleanup must be explicit |
| Data | Setup must be deterministic where the claim depends on data |

---

## Docker Compose concurrency

Proofs must be safe for a growing library and parallel author sessions.

### No fixed global container names

Do **not** use `container_name:` unless a proven external integration absolutely requires one. Let Compose namespace resources.

### Ports

Avoid undocumented hardcoded globally scarce ports. Preferred model:

```yaml
ports:
  - "${PROOF_POSTGRES_PORT:-5435}:5432"
```

The proof README and `.env.example` must expose that variable. If a future canonical dynamic-port allocator is implemented, this guide may evolve — do not invent one now.

### Volumes and networks

Use proof-local Compose resources. No external shared volume by default.

---

## Parallel execution requirement

A proof author must consider whether two different proofs can run concurrently.

**Acceptance posture:**

- no global fixed container names
- configurable host ports
- proof-scoped volumes
- proof-scoped environment variables where local infra requires them
- no shared mutable fixture state
- no assumption that another proof is stopped

If true parallel execution is impossible because of an external provider or service constraint, document the limitation explicitly in the proof README.

---

## Dataset and fixtures

When a proof relies on controlled data:

- deterministic source / generator preferred
- fixed version / seed where applicable
- verify materialized data before model execution
- model must **not** receive hidden expected answers or ground-truth implementation
- **expected-answer prompting is forbidden**
- fixture may simulate environment only when the boundary under proof remains real

Ground truth belongs to evaluator / evidence logic — **not** the model prompt.

---

## Falsification requirements

Every **SCENARIO** must define:

- positive path where relevant
- meaningful negative / adversarial paths
- exact conditions that cause FAIL
- bounded execution
- model-independent invariants wherever possible

**PASS cannot mean:** “The answer looks reasonable.”

**Prefer:**

- structural checks
- deterministic invariant checks
- typed critic / evaluator verdicts
- evidence dependency validation
- explicit unsupported-claim detection

LLM-as-judge may augment but must **not** become the sole source of truth when deterministic evidence exists.

---

## Critic / second agent

A second agent / Critic **MUST NOT** be introduced merely to make a scenario appear multi-agent.

Use Critic when **independent verification is part of the required guarantee**. It verifies observable artifacts:

- candidate output
- evidence
- tool traces
- trajectory
- `InvestigationProof`
- structured claims

Do **not** collect private chain-of-thought as proof evidence. Preferred concept: **observable reasoning trajectory** — not hidden internal reasoning.

Reuse Intergrax Critic & Verification Layer when applicable. Do not create proof-local critic architecture when a platform mechanism exists.

---

## Observability

Every **SCENARIO** must satisfy § Mandatory observability, explainability, and diagnostics and pass the **Application Observability Test**. The Scenario **MUST NOT** be a black box.

Reuse Intergrax observability / HOS where applicable — especially `RuntimeState.trace_events`, `ObservabilityEmitter`, typed `TraceEvent`, typed `DiagnosticPayload`, and `ToolCallTrace`. The proof report should expose useful operational evidence such as:

- execution identity
- attempts
- tool invocations
- evidence dependencies
- critic verdicts
- revisions / retries
- terminal reason
- relevant timings
- provenance

Do **not** create a private competing observability bus or store. Logs are **not** canonical execution evidence. Never expose chain-of-thought or secrets.

---

## Evidence contract

Reuse existing contracts — no competing model:

- `PlatformProofEvidence`
- `SuiteReceipt`
- artifact verification (PP-SUITE-4)

Evidence must distinguish:

- what the model could see
- hidden ground truth
- real vs fixture boundaries
- execution evidence
- evaluator verdict
- limitations
- excluded claims

**Exit code alone is never sufficient** when `evidence_required=true`.

For descriptor-backed proofs with `evidence_required=true`, write `evidence.json` to the runner-provided `INTERGRAX_PROOF_ARTIFACT_DIR` when executed via suite.

---

Do **not** create a private competing observability bus or store. Logs are **not** canonical execution evidence. Never expose chain-of-thought or secrets. The Proof layer **MUST NOT** invent execution explanations absent from canonical structured artifacts.

---

## Report contract

Every accepted public **SCENARIO** must generate a rich human-readable `report.html`. Reuse the canonical renderer / report standard ([PLATFORM_PROOF_REPORT_STANDARD.md](../docs/project/proofs/PLATFORM_PROOF_REPORT_STANDARD.md)).

**Source-of-truth rule (mandatory):** report prose about execution **MUST** be derived from machine-readable structured artifacts (`PlatformProofEvidence`, runtime trace export, evidence graph) — **not** from expected outcomes or post-hoc narrative invention. No report builder may invent a tool call that was not executed, a model decision that was not emitted, rationale absent from execution output, evidence not present in canonical evidence, or a challenge not present in lifecycle data.

SCENARIO reports **MUST** support (via generic renderer and/or evidence-backed domain extensions): decision/investigation timeline; explicit operator-facing rationales; tools/external actions and evidence; claim lifecycle; critic/governance; diagnostics; final decision provenance linked to verified evidence, actions, rationales, challenges, and terminal gate. Do not require empty decorative sections for Conformance proofs.

The report must let a skeptical engineer understand, **without reading source first**:

1. the real problem
2. why it matters
3. failure being tested
4. claim
5. architecture / mechanisms used
6. real vs simulated boundaries
7. what information the model had
8. adversarial conditions
9. execution story / timeline
10. evidence graph / dependencies where relevant
11. critic / verification results where relevant
12. final verdict
13. why PASS / FAIL occurred
14. limitations
15. excluded claims
16. exact reproduction instructions
17. provenance / environment

The report derives from typed evidence — it is **not** a second source of truth. Do not make the report a marketing page.

---

## Artifacts

| Location | Role |
|----------|------|
| `.artifacts/` | Transient run state (gitignored) |
| `<proof-package>/output/` | Canonical publishable output (Git-trackable) |

Stable filenames where declared:

```text
evidence.json
proof-result.json
report.html
```

Canonical output must be safe to commit and link from public documentation. Never commit provider secrets or raw unsafe traces.

`.artifacts/` remains transient suite state; canonical `output/` is commit-ready public proof evidence.

---

## Scenario documentation model (README gateway + SCENARIO_SPEC)

Every Scenario Proof uses a **two-file documentation model**:

| File | Role | Audience |
| --- | --- | --- |
| **`README.md`** | Public gateway (~3–5 min) | Skeptical visitor, product reader |
| **`SCENARIO_SPEC.md`** | Deep canonical contract | Technical reviewer, implementer |

> **The problem owns the Scenario. Platform domains participate according to the guarantees required by that problem.**

Do **not** maintain identical detailed PASS/FAIL lists, adversarial conditions, ground-truth details, or verifier contracts in both files. README summarizes; SCENARIO_SPEC is normative.

### README.md — public gateway

Required presentation order:

```text
HERO / IDENTITY (title, public question, lifecycle status)
↓
ABSTRACT
↓
AT A GLANCE
↓
VISUAL PROOF STORY
↓
THE PROBLEM
↓
THE RISK
↓
THE NAIVE FAILURE / TRAP
↓
ADVERSARIAL CHALLENGE (summary)
↓
WHAT THE PROOF CLAIMS
↓
PASS / FAIL (summary table)
↓
OUTCOMES (RESOLVED / UNRESOLVED)
↓
LATEST VERIFIED RUN        # post-implementation only
↓
RUN / REPORT / EVIDENCE / SOURCE
↓
LIMITATIONS (summary)
↓
GO DEEPER → link to SCENARIO_SPEC.md
```

README must **not** contain the full A/B/C/D/E deep contract.

### SCENARIO_SPEC.md — deep canon

Canonical deep contract for scenario design and implementation:

```text
A. SCENARIO
B. SOLUTION
C. INTERGRAX FIT
D. GAP DECISION
E. PROOF BUILD
```

§ C is **not** a single-domain assignment. Expected future analysis:

```text
required guarantee
→ Intergrax mechanism
→ exact owner/component
→ participating domain(s)
→ AVAILABLE / AVAILABLE BUT NEEDS WIRING / MISSING
```

Cross-links: README links to SCENARIO_SPEC.md; SCENARIO_SPEC links back to README.md (relative paths).

### Abstract contract

Every Scenario README must include **`## Abstract`** immediately after title/public question/status and before **At a glance**.

Target: one short paragraph (~4–8 sentences). A reader who spends only a few seconds should understand:

1. what happened;
2. who has the problem;
3. why it matters;
4. what the naive AI answer is likely to get wrong;
5. what the Scenario is trying to demonstrate.

No platform internals, Intergrax component names, implementation detail, marketing slogans, or claims beyond current lifecycle state. Readable by a non-Intergrax reader. Summarizes — does not duplicate the full Problem section.

### Lifecycle status wording

Use precise lifecycle language — never bare `ACCEPTED` in a way that could be confused with proof PASS.

| Status | Meaning |
| --- | --- |
| `DESIGN / NOT YET ACCEPTED` | Scaffold or design in progress; Scenario Quality Gate not passed |
| `ACCEPTED FOR IMPLEMENTATION` | Scenario concept accepted; no executable proof, evidence, or report yet |
| `IMPLEMENTATION INITIALIZED` | Gated skeleton generated; domain implementation in progress |
| Post-implementation | Show factual execution state in **Latest verified run** (verdict, SHA, timestamps, invariants) |

At **DESIGN** stage do **not** show PASS badges, executable status, or sample runtime numbers.

### Mandatory across all Scenario Proofs

- identity / public question;
- lifecycle status (truthful);
- **Abstract** (problem-story summary);
- real problem (public summary + § A detail in SCENARIO_SPEC);
- consequence / risk;
- WOW and Skeptic Challenge (§ A in SCENARIO_SPEC);
- adversarial conditions (summary in README; normative detail in SCENARIO_SPEC);
- claim;
- PASS / FAIL (summary in README; normative detail in § B);
- excluded claims / limitations (summary in README; normative detail in SCENARIO_SPEC);
- A/B/C/D/E lifecycle sections in **SCENARIO_SPEC.md**;
- **At a glance** table (filled during qualification);
- at least one explanatory visual flow once the scenario is mature enough (after Scenario Quality Gate);
- **Go deeper** link from README to SCENARIO_SPEC.md.

### Conditional (only when relevant)

Do **not** force irrelevant concepts on every scenario:

- ground truth isolation;
- verifier independence;
- competing hypotheses;
- temporal admissibility;
- side-effect safety;
- HITL;
- recovery;
- governance.

Use **Conditional authoring prompts** in the SCENARIO_SPEC scaffold when a scenario may need them.

### At a glance contract

Near the top, every mature Scenario README exposes:

| Field | Purpose |
| --- | --- |
| Problem | Workflow or operational pain |
| Observed impact | Measurable or observable harm |
| Trap | Naive failure / correlation trap |
| Decision risk | What goes wrong if diagnosis is wrong |
| Scenario outcome | RESOLVED / UNRESOLVED semantics |
| Status | Current lifecycle wording |
| Proof class | SCENARIO |

Scaffold creates placeholders; authors must fill during qualification.

### Visual assets

- Store scenario-owned assets under `platform_proofs/scenarios/<slug>/assets/`.
- Prefer self-contained **light/dark SVG** pairs per [DOCUMENTATION_DESIGN_SYSTEM.md](../docs/project/technical/guides/DOCUMENTATION_DESIGN_SYSTEM.md).
- Use `<picture>` with relative paths; wrap in `<a href="...-light.svg">` for full-size navigation on GitHub.
- **Do not** auto-generate meaningless SVGs in the scaffold — use an authoring HTML comment placeholder until after the quality gate.
- **Do not** create decorative artwork, fake dashboards, or screenshots of nonexistent execution.

Target: one strong proof-story diagram; one supporting diagram when it materially improves understanding; clear tables and callouts.

### Post-run sections

**Latest verified run** — populated only after real execution and report acceptance:

- verdict; proof version; Intergrax SHA; model/provider; run timestamp; RESOLVED/UNRESOLVED; key invariant results.
- CTA links: View report · View evidence · View source · Run locally.

At design stage: omit or show a clearly disabled **Not yet available** note. No fake badges or numbers.

### README contract for executable SCENARIO packages

After implementation, each README must also document phases relevant to execution (do not lead with platform internals). Normative semantics remain in SCENARIO_SPEC.md.

```text
1. Real problem (public layer + § A in SCENARIO_SPEC)
2. Claim / PASS / FAIL (summary in README; § B in SCENARIO_SPEC)
3. Intergrax mechanisms exercised (§ C in SCENARIO_SPEC)
4. Real / fixture boundaries (§ B in SCENARIO_SPEC when relevant)
5. Quick start / configuration
6. Validate
7. Run
8. View report / artifacts
9. Cleanup
10. Limitations / excluded claims (summary + link to SCENARIO_SPEC)
11. Latest verified run
12. Run / report / evidence / source links
```

A visitor should understand the problem before learning which Intergrax layers solve it.

---

## Public Library acceptance gate

A Scenario Proof is **not accepted** as a Proof Library entry until **all** general gates **and** the Scenario-specific gates below are true.

### General gates

- [ ] real problem clearly defined
- [ ] meaningful failure risk
- [ ] falsifiable claim
- [ ] Intergrax mechanisms mapped
- [ ] capability gaps resolved honestly
- [ ] package independently runnable within cloned repo
- [ ] configuration documented
- [ ] infrastructure isolated
- [ ] deterministic / preflight validation where relevant
- [ ] real proof executed
- [ ] required evidence validates
- [ ] evaluator passes
- [ ] rich report generated and manually audited
- [ ] reproduction tested
- [ ] limitations / excluded claims present
- [ ] no secrets
- [ ] no undocumented dependencies
- [ ] canonical output ready for publication

### Scenario-specific mandatory gates (YES/NO)

Any critical **NO** → Scenario **cannot** be **ACCEPTED**.

| # | Gate | Required |
|---|------|----------|
| 1 | Useful application remains without proof/evaluator/report? | YES |
| 2 | Canonical proof uses deployment-capable application path? | YES |
| 3 | No FakeLLM/scripted model in canonical path? | YES |
| 4 | No Mock/MagicMock/testing_support in application path? | YES |
| 5 | Real model/provider used when AI behavior is material? | YES |
| 6 | Controlled data enters through normal provider/tool contracts? | YES |
| 7 | Application performs bounded autonomous workflow? | YES |
| 8 | Proof harness does not own business reasoning? | YES |
| 9 | Generic capabilities come from Intergrax rather than local clones? | YES |
| 10 | Application core can swap controlled provider for production provider without redesign? | YES |
| 11 | Proof layer only falsifies/evaluates/evidences? | YES |
| 12 | System invariants — not fake deterministic output — provide proof stability? | YES |
| 13 | Existing real-boundary requirements are satisfied? | YES |
| 14 | Excluded production-validation claims are explicit? | YES |

If any fail:

```text
NOT ACCEPTED INTO PROOF LIBRARY
```

even if code exists.

Update the scenario package (`README.md`, `SCENARIO_SPEC.md`, and descriptor when implemented) as the canonical source of truth. Update [PROOFS.md](../docs/project/proofs/PROOFS.md) **only** when accepted public evidence / claim boundaries change.

---

## Workflow for parallel proof-author sessions

Every new independent session receives:

```text
Read and follow:
platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md

Scenario:
<USER-SUPPLIED REAL PROBLEM DESCRIPTION>
```

The session **MUST** then:

1. resolve current HEAD (`git fetch origin development`)
2. read this guide
3. confirm **one session = one Scenario Proof** ownership
4. work through Stages 1–4 using the mandatory conversation format (§ Mandatory session conversation format)
5. inspect only architecture / components relevant to the scenario
6. pass Scenario Quality Gate and Skeptic Challenge before implementation
7. **STOP** for architecture decision if a reusable mechanism is **MISSING** (STOP ≠ abandon scenario)
8. only after approval — Stage 5 / technical lifecycle (implement → test → execute → evidence → report)
9. publish only after acceptance gates

Do not assume decisions from another scenario unless encoded in canonical repo documentation. **Current repo wins.**

---

## Shared-development rules

Mandatory rules for proof sessions:

| Rule | Requirement |
|------|-------------|
| Branch | `development` |
| HEAD | resolve before every material task |
| Concurrency | parallel sessions expected |
| Branching | no branches / worktrees unless operator changes policy |
| Destructive git | no reset, rebase, stash, clean, amend, force |
| Unrelated work | never overwrite concurrent changes |
| Commits | only scoped changes; return SHA |
| Conflicts | if scoped concurrent conflict occurs — **STOP** |

---

## Engineering restrictions

- reuse before create
- typed contracts
- `extra="forbid"` where descriptor / evidence contracts use it
- no `dict[str, Any]` in new proof contracts
- no `getattr` / `setattr` / loose reflection
- no speculative abstraction
- no second runner
- no second observability stack
- no second receipt
- no expected-answer prompt leakage
- no fake proof-local replacement for platform mechanism
- no FakeLLM, scripted model, or test double in canonical Scenario application path
- no proof harness owning business workflow while claiming application autonomy
- no deterministic fake model output used to substitute real AI boundary for stability
- no descriptor v1, compatibility shims, aliases, fallback, or legacy proof execution path
- no mass migrations for symmetry

---

## Session STOP conditions

A proof-author session **MUST STOP** and report instead of improvising when:

- required reusable mechanism is **MISSING** (see § Stage 4 — Missing capability decision; STOP ≠ abandon scenario)
- existing architecture ownership is ambiguous
- proof would need to copy product logic
- real boundary cannot be exercised
- ground truth would need to be leaked to the model
- evaluator cannot define meaningful falsification
- package cannot be made independently reproducible
- scope begins requiring platform redesign not approved by user
- concurrent edits conflict in scoped files
- canonical proof-package configuration / environment loading required by the scenario is not available as reusable proof infrastructure — **STOP** and report the gap; do not implement a proof-local dotenv / config loader
- Scenario Quality Gate or Skeptic Challenge cannot be satisfied after honest strengthening attempts — reject or redesign; do not proceed to proof build

---

### Design-stage and implementation commands (quick reference)

Full lifecycle: § Canonical Scenario Lifecycle.

| Step | Command | Lifecycle |
|------|---------|-----------|
| 1. Create design package | `uv run python scripts/proof/create_scenario_proof.py --slug <slug> --title "<title>"` | `DESIGN / NOT YET ACCEPTED` |
| 2. Design + quality gate | Fill README + SCENARIO_SPEC (A–E); pass Scenario Quality Gate | → `ACCEPTED FOR IMPLEMENTATION` |
| 3. Init implementation | `uv run python scripts/proof/init_scenario_implementation.py --slug <slug>` | → `IMPLEMENTATION INITIALIZED` |
| 4. Build + prove | Implement `application/` + `proof/`; run `run_proof.py` | → executable → evidence → library acceptance |

Do not manually invent scenario directory shapes, skip the quality gate, or run the initializer before acceptance frontmatter is set. Complete explanatory visuals under `assets/` **before** implementation when the scenario is mature enough. Post-run README sections populate only after execution.

---

## Current reference proof

No executable Scenario or Conformance platform proof is designated as the canonical reference example yet. The first Scenario Proof (`ai_incident_investigation`) is **ACCEPTED FOR IMPLEMENTATION** — design qualification passed; implementation and executable evidence have not started. Public presentation reference: [`scenarios/ai_incident_investigation/README.md`](scenarios/ai_incident_investigation/README.md). Deep contract: [`scenarios/ai_incident_investigation/SCENARIO_SPEC.md`](scenarios/ai_incident_investigation/SCENARIO_SPEC.md).

**Observability contract migration (DOC-PROOF-OBS-1):** the global non-black-box observability standard applies prospectively. Existing in-progress Scenario packages that predate full observability implementation (including `ai_incident_investigation`) are **not** retroactively rejected at design stage, but **MUST** satisfy the new standard before executable acceptance (APP-2A and later). No immediate rewrite of design-stage documentation is required solely because migration is pending; public claims must remain truthful.

Scenario package source of truth: [`scenarios/ai_incident_investigation/`](scenarios/ai_incident_investigation/).

---

## Anti-patterns

| Anti-pattern | Why it fails |
|--------------|--------------|
| Feature chosen before problem | Demo, not falsification |
| Multiple scenarios in one session | Violates one-session ownership; fragments quality gates |
| Feature stuffing for WOW | Complexity without credible guarantee |
| LangGraph-equivalent scenario without strengthening | Fails Skeptic Challenge |
| Fake replacing mechanism under proof | Cannot claim that boundary proved |
| FakeLLM / scripted model in canonical Scenario application path | Invalidates production-capable Scenario; use real model boundary when AI behavior is material |
| Proof harness owns workflow; application is thin wrapper | Fails Application Survival Test and bounded autonomy |
| Separate fake/scripted proof path vs deployment application path | Violates canonical Scenario execution path rule |
| Proof-only diagnosis/routing/hardcoded outcome in canonical path | Business logic must live in application core |
| Mock/MagicMock/testing_support in Scenario application core | **PROHIBITED** in canonical path |
| Stability from fake deterministic model output | Use platform invariants instead |
| Product proof masquerading as platform proof | Violates ownership rule |
| Proof buried only in `applications/` | Product-owned; not platform proof |
| Duplicate proof runner or manifest | Fragments execution truth |
| Undocumented environment assumptions | BLOCKED or false PASS |
| Claim broader than evidence | Public governance violation |
| No negative scenario | Not falsification — demo only |
| PASS based only on prose | Not machine-checkable |
| PASS based only on exit code when `evidence_required=true` | Suite verifies typed `evidence.json` |
| Chain-of-thought collection as evidence | Not platform invariant; explicitly forbidden for Scenario observability |
| Report invents execution narrative absent from structured artifacts | Violates source-of-truth rule; BLOCKER |
| Proof-specific logger as sole execution trace source | Fails Application Observability Test |
| Material decision without bounded rationale/objective | Fails explainability pillar |
| Operational failure narrated as epistemic UNRESOLVED | Violates diagnostic terminal semantics |
| Reimplementing platform components inside proof | Proves clone, not platform |
| Proof-local critic / observability / receipt stack | Fragments platform truth |
| Expected-answer prompt leakage | Invalidates falsification |

---

## Related documents

| Document | Role |
|----------|------|
| [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) | Governance — classification, falsification, evidence |
| [README.md](README.md) | Proof Library gateway |
| [PLATFORM_CONFIGURATION.md](../docs/project/technical/guides/PLATFORM_CONFIGURATION.md) | Canonical env / provider names |
| [PROOFS.md](../docs/project/proofs/PROOFS.md) | Public proof dashboard |
| [PUBLIC_PROOF_AND_CLAIMS_MODEL.md](../docs/project/maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md) | Public wording governance |
