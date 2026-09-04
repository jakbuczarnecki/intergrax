# Decision System

**Intergrax Decision System** is the platform capability that leads a decision from proposal through optional deliberation, verification, revision, optional adjudication, resolution, and finalization to an **authoritative lifecycle outcome** - a **semantic capability hosted inside canonical Execution**, not a second runtime.

The Decision System answers **„jaki jest autorytatywny wynik decyzji?”** - classification, recommendation, selection, plan, approval, finding, or evidence-backed conclusion. It is **not** an „ulepszony Critic”, **not** Council Runtime, and **not** a parallel execution engine.

> [!IMPORTANT]
> **Maturity boundary (frozen target vs current production):**
>
> - **Architecture:** **TARGET CANON - FROZEN** (this document and paired [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md)).
> - **Implementation:** Canonical Decision System runtime **implemented and active**.
> - **CURRENT decision authority = Decision System.** Critic runtime **retired**.
> - **Production qualification of full Decision System still pending DS-E2E** — no whole-system production-qualified claim.

**Primary audience:** Principal / Staff engineers, harness integrators, and Tier-2/3 authors configuring decision strategies, verification posture, and adjudication flows.

---

## Why it matters

Without a single platform Decision System:

- verification, deliberation, revision, and human authority collapse into ad-hoc critic loops,
- Council becomes a second runtime with its own scheduler and retry,
- candidate outputs are mistaken for authoritative decisions,
- version history is mutated instead of appended,
- policy authorization is confused with decision correctness,
- parallel proposals race to last-write-wins finalization,
- crash resume can mint duplicate authoritative outcomes or expand budgets,
- diagnostics and decision-making compete for the same ownership.

The Decision System provides **typed lifecycle semantics, version lineage, compositional verification, extensible strategies, and audit surfaces** so applications compose domain decisions safely while the **Execution System** remains the canonical host for execution lifecycle, identity, and strategy routing.

**Decision Lifecycle is hosted by canonical Execution. Domain/application owns what is being decided.**

---

## At a glance

| Concern | Summary |
| -------- | -------- |
| **Core question** | What is the authoritative decision outcome for this scope? |
| **Execution host** | **Execution System** - canonical execution lifecycle, identity, and strategy routing |
| **Decision capability** | Semantic lifecycle inside hosting Execution - **no** DecisionRuntime |
| **Lifecycle** | Proposal → optional Deliberation → Verification → Revision → optional Adjudication → Resolution → Finalization |
| **Decision Resolution** | `ACCEPTED` · `REJECTED` · **`UNRESOLVED`** - merytoryczny wynik lifecycle; oddzielny od termination wykonania |
| **Strategy** | Pluggable `DecisionStrategy` - Single Model, Council, Rule-Based, Hybrid, future registered strategies |
| **Artifact** | Typed `Decision Artifact` family - not universal `payload: dict[str, Any]` |
| **Candidate vs authoritative** | Candidates are proposals; **ACCEPTED** binds a specific **Decision Version**; terminal **REJECTED** / **UNRESOLVED** persist an authoritative **resolution record** without a fake accepted version |
| **Verification** | Compositional **Verification Pipeline** - see [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) |
| **Deliberation** | Optional strategy capability - see [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md) |
| **UNRESOLVED** | First-class auditable outcome when material is insufficient or conflict is irresolvable |
| **Decision ≠ Authorization ≠ Execution** | Three separate platform responsibilities - see [below](#decision--authorization--execution) |
| **Version binding** | Every verification result, challenge, approval, adjudication, and authorization record binds **Decision ID + Decision Version + scope + tenant + execution identity** |
| **Concurrency** | Parallel proposal branches preserve lineage; no duplicate authoritative decisions per scope |
| **Crash / resume** | Canonical hosting Execution checkpoint/persistence - **no** Decision checkpoint engine |
| **Retry boundaries** | Technical retry (Execution / Reliability) · decision revision (Decision Lifecycle) · deliberation rounds (Decision Strategy) - never one generic loop |
| **HITL** | Invokes platform HITL - does not implement Human Engine |
| **Policy** | Cross-cutting authorization - Decision System does not own Runtime Policy Engine |
| **Diagnostics** | May feed investigation - does not own Decision System |
| **Observability** | Full decision audit trail - no private chain-of-thought |
| **Maturity** | **A4 target / I0 / P0 / E0** for Decision System - see [Current maturity](#current-maturity) |

---

## Flagship architecture visual

<a href="assets/fullsize/decision-system-flagship.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-system-flagship-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-system-flagship-light.svg">
  <img
    alt="Conceptual diagram: Application flows through Execution into optional Decision Lifecycle with Strategy, Verification, Revision, and Adjudication to Authoritative Decision, bounded by Policy, HITL, Execution, Observability, and Diagnostics."
    src="assets/decision-system-flagship-light.svg"
  >
</picture>
</a>

> **Decision Lifecycle is hosted by canonical Execution. Decision correctness ≠ permission to execute ≠ execution itself.**

```text
Application intent
      ↓
Execution (canonical host)
      ↓
optional Decision Lifecycle
├── Decision Strategy (Council / Single / Rule / Hybrid)
├── Verification Pipeline
├── Revision (bounded)
└── optional Adjudication
      ↓
Authoritative lifecycle outcome (accepted decision or resolution record)
      ↓
Policy / HITL may gate consequential execution
      ↓
Execution System routes work
├── INFERENCE
├── AGENTIC
└── ORCHESTRATION → Nexus (when orchestration is required)
```

---

## Responsibility model

| Concern | Owner |
| ------- | ----- |
| Decision lifecycle semantics | Decision System |
| Hosting execution lifecycle | Execution System |
| Strategy routing | Execution System |
| Orchestration scheduling | Nexus |
| Decision verification | Verification Pipeline |
| Decision resolution/finalization semantics | Decision System |
| Durable execution/checkpoint host | Execution System / canonical execution persistence boundary |
| Authorization | Policy / Governed Execution |
| HITL authority | HITL |
| Audit evidence | Observability |

| Domain | Owns | Does not own |
| ------ | ---- | ------------ |
| **Decision System** | Lifecycle semantics, candidate/authoritative semantics, version lineage, resolution (incl. UNRESOLVED), strategy orchestration contract | Global retry, authorization, side effects, diagnostics classification, private CoT, execution hosting |
| **Verification Pipeline** | Check correctness of a **Decision Version** - stages, challenges, fail-closed rules | Finalize authoritative decision, mutate versions, policy, HITL, global retry |
| **Decision Strategy** | Deliberation rounds, parallel proposals, disagreement artifacts, synthesis candidates | Separate runtime, scheduler, checkpoint engine, authorization |
| **Execution System** | Execution lifecycle, identity, root execution establishment, canonical boundary, strategy routing, hosting Decision capability | Domain rubric content, decision semantics |
| **Nexus** | ORCHESTRATION execution strategy - scheduling child Executions, dependency/readiness/fan-out/merge, orchestration control flow | Universal host of Decision System, Decision persistence by definition, mandatory path for every Decision |
| **Governance / Policy** | Execution authorization for consequential actions | Whether a decision artifact is correct |
| **HITL** | Human approver / adjudicator / escalation records | Decision lifecycle orchestration |
| **Reliability** | Technical retry on provider/tool failure | Semantic revision loops |
| **Observability** | Audit evidence for decision events | Decision rubric or strategy content |
| **Diagnostics** | Detect/classify platform problems | Own decision lifecycle |

```text
Harness  → lifecycle orchestration, verification composition, evidence hooks
Domain   → artifact types, rubrics, strategy selection, correctness criteria
```

### Public invariants

```text
Decision Lifecycle hosted by canonical Execution - never a second DecisionRuntime.
```

```text
Decision capability MUST NOT require ORCHESTRATION strategy.
```

```text
A Decision Lifecycle may be hosted by an Execution routed through
INFERENCE, AGENTIC, or ORCHESTRATION as required by its work.
```

```text
Decision correctness ≠ permission to execute ≠ execution itself.
```

```text
Candidate Decision ≠ Authoritative Decision - history is never overwritten.
```

```text
UNRESOLVED is a valid, auditable resolution outcome.
```

```text
Decision Resolution ≠ Lifecycle / Execution Termination.
```

```text
Execution = FAILED does not imply Decision Resolution = REJECTED.
```

```text
Verification checks. Verification does not authorize, execute, or finalize alone.
```

```text
Approval for v1 does NOT authorize v2.
```

---

## Decision Lifecycle

The lifecycle is a **state machine model** hosted by canonical Execution - semantic stages execute within the hosting Execution boundary. Work required by a Decision Strategy is routed through the Execution System (INFERENCE, AGENTIC, or ORCHESTRATION).

<a href="assets/fullsize/decision-lifecycle.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-lifecycle-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-lifecycle-light.svg">
  <img
    alt="Decision Lifecycle flowchart: Proposal, optional Deliberation, Verification, Revision, optional Adjudication, Resolution with ACCEPTED REJECTED or UNRESOLVED, and Finalization to authoritative resolution record or accepted decision version."
    src="assets/decision-lifecycle-light.svg"
  >
</picture>
</a>

| Stage | Purpose |
| ----- | ------- |
| **Proposal** | Mint initial **Candidate Decision** with typed **Decision Artifact** |
| **Deliberation** | Optional - strategy produces one or more candidates (e.g. Council) |
| **Verification** | Compositional pipeline evaluates a specific **Decision Version** |
| **Revision** | Explicit process mints **new Decision Version** when verification challenges |
| **Adjudication** | Optional - resolve competing proposals, verifier conflict, deadlocked Council, or human adjudication |
| **Resolution** | `ACCEPTED` · `REJECTED` · **`UNRESOLVED`** - bounded, auditable **Decision Resolution** |
| **Finalization** | Persist authoritative **lifecycle outcome** - accepted decision version or terminal resolution record |

Council is **only** a Decision Strategy implementation - not a mandatory stage.

---

## Decision Strategy ≠ Execution Strategy

**DecisionStrategy** and **ExecutionStrategy** are distinct axes.

| Axis | Question | Examples |
| ---- | -------- | -------- |
| **DecisionStrategy** | How do we reach a decision? | Single Model, Rule-Based, Hybrid, Council |
| **ExecutionStrategy** | How is a concrete unit of work executed? | INFERENCE, AGENTIC, ORCHESTRATION |

Council may use ORCHESTRATION but is not synonymous with Nexus as a system. Single Model and Rule-Based may operate without Nexus.

### Example - simple decision (no Nexus)

```text
Execution
↓
Decision Lifecycle
↓
Single Model
↓
Verification
↓
ACCEPTED
```

### Example - rule decision (no Nexus)

```text
Execution
↓
Decision Lifecycle
↓
Rule-Based Strategy
↓
REJECTED
```

### Example - council (Nexus when orchestration required)

```text
Execution
↓
Decision Lifecycle
↓
Council
↓
parallel analyses required
↓
Execution System
↓
ORCHESTRATION
↓
Nexus
↓
child Executions
```

Nexus appears because orchestration work is required - not because every Decision mandates it.

### Decision capability optionality

Decision System is **optional per flow**. When no authoritative decision is required, ordinary Execution completes without entering Decision Lifecycle.

- **Absence** means Decision Lifecycle is **not entered** - not a `NoDecisionStrategy` / null-strategy workaround.
- **No global on/off flag** - optionality is per flow, not a `DECISION_SYSTEM_ENABLED` switch.
- **Orthogonal to ExecutionStrategy** - INFERENCE, AGENTIC, and ORCHESTRATION each work without Decision capability; neither axis implies the other.
- **Optional host seam** - Execution runtime may reference optional Decision capability hooks (DS-EXEC-01); ordinary flows remain fully valid without selecting or entering Decision Lifecycle.

Proof gate: DS-EXEC-00 (`tests/unit/runtime/execution/test_decision_optionality.py`).

### Execution-hosted Decision Lifecycle (DS-EXEC-01)

Canonical Execution owns **hosting scope** for an optional Decision Lifecycle capability. Decision-aware code explicitly invokes the scoped host; Execution does **not** automatically start Decision Lifecycle before delegate routing.

```text
ExecutionRuntime
      ↓ optional scoped host binding
DecisionLifecycleHost
      ↓
canonical DecisionLifecycle contracts (decision_lifecycle.py)
```

| Invariant | Meaning |
| --------- | ------- |
| **Host presence ≠ Decision selected** | Configuring a host does not imply a flow needs a decision |
| **Host presence ≠ lifecycle entered** | No `DecisionIdentity` or `DecisionLifecycleState` is created until decision-aware code calls the host |
| **Decision-aware flow explicitly invokes host** | `require_active_decision_lifecycle_host()` inside governed delegate work |

| Layer | Owns |
| ----- | ---- |
| **ExecutionRuntime** | Lifecycle hosting scope (optional scoped host bind/reset around canonical boundary) |
| **DecisionLifecycleHost** | Typed access to canonical lifecycle operations (`start`, `transition`) |
| **`decision_lifecycle.py`** | State machine semantics and legal transitions |
| **StrategyExecutionRouter** | Physical `ExecutionStrategy` routing (INFERENCE · AGENTIC · ORCHESTRATION) |
| **ExecutionBoundary** | Context / admission / delegate coordination - no Decision semantics |

Proof gate: DS-EXEC-01 (`tests/unit/runtime/execution/test_decision_lifecycle_host.py`).

### Execution-hosted Decision checkpoint persistence (DS-EXEC-02)

Canonical Execution owns **hosting scope** for an optional Decision checkpoint persistence capability. Decision-aware code explicitly saves and loads canonical `DecisionCheckpointState` snapshots; Execution does **not** automatically checkpoint, restore, or resume lifecycle work.

```text
ExecutionRuntime
 ├── optional DecisionLifecycleHost
 └── optional DecisionCheckpointPersistence
      ↓ execution-scoped binding
decision-aware lifecycle code
      ↓
save_decision_checkpoint / load_decision_checkpoint
      ↓
existing DecisionCheckpointState contracts
```

| Invariant | Meaning |
| --------- | ------- |
| **Persistence presence ≠ lifecycle entered** | Configuring persistence does not create lifecycle state |
| **Persistence presence ≠ automatic checkpoint** | No save/load on `ExecutionRuntime.execute()` unless decision-aware code invokes helpers |
| **Decision-aware flow explicitly invokes persistence** | `require_active_decision_checkpoint_persistence()` inside governed delegate work |

| Layer | Owns |
| ----- | ---- |
| **ExecutionRuntime** | Persistence hosting scope (optional scoped bind/reset around canonical boundary) |
| **DecisionCheckpointPersistence** | Execution-facing durability port (`load`, `save`) keyed by `DecisionFinalizationKey` |
| **`decision_checkpoint.py`** | Checkpoint semantics and validation |
| **DecisionLifecycleHost** | Lifecycle operations only (`start`, `transition`) - no persistence ownership |

Execution hosts persistence access. Decision contracts own checkpoint semantics. No automatic save/load/resume.

Proof gate: DS-EXEC-02 (`tests/unit/runtime/execution/test_decision_checkpoint_runtime_integration.py`).

### Execution-hosted Decision work submission (DS-NEXUS-01)

Decision Strategy implementations may require physical work (INFERENCE, AGENTIC, or ORCHESTRATION) without knowing internal execution engines. Decision-aware delegate code obtains an optional execution-scoped `ExecutionWorkPort` and submits canonical `ExecutionRequest` values - including `ExecutionCapability.ORCHESTRATION` when orchestration is required.

```text
ExecutionRuntime
      ↓ optional execution-scoped work port binding
Decision-aware delegate
      ↓ require_active_execution_work_port()
ExecutionWorkPort
      ↓ ChildExecutionRunner (child ExecutionId under active parent)
StrategyExecutionRouter
      ├── INFERENCE
      ├── AGENTIC
      └── ORCHESTRATION
            ↓ private implementation
          Nexus
```

| Invariant | Meaning |
| --------- | ------- |
| **Nexus is private** | Decision contracts and decision-aware helpers must not import `intergrax.runtime.nexus` or reference Nexus types |
| **Work port presence ≠ orchestration required** | Ordinary Execution flows remain valid without configuring a work port |
| **Decision does not route strategies** | `StrategyExecutionRouter`, `OrchestrationExecutor`, and Nexus are Execution-owned composition concerns |
| **Child lineage preserved** | Work port submits child Executions under the active parent via `ChildExecutionRunner` |
| **Missing backend fails closed** | ORCHESTRATION requests without a configured backend raise canonical Execution errors - no Decision-specific fallback |

Nexus appears only when Execution strategy routing selects ORCHESTRATION for submitted work - not because Decision System exists.

Proof gate: DS-NEXUS-01 (`tests/unit/runtime/execution/test_decision_execution_work.py`).

### Orchestration checkpoint/recovery participation (DS-NEXUS-02)

When Decision-aware code requests `ExecutionCapability.ORCHESTRATION`, physical orchestration work checkpoints and resumes through canonical Execution recovery (`RuntimeCheckpoint`, `ExecutionTreeSnapshot`, `prepare_task_for_checkpoint_resume`). Decision semantic checkpoint (`DecisionCheckpointState`) remains Decision-owned and is not mutated by physical resume.

| Invariant | Meaning |
| --------- | ------- |
| **Separate checkpoints** | `DecisionCheckpointState` does not embed execution-tree snapshots; `RuntimeCheckpoint` does not embed Decision lifecycle state |
| **No Decision recovery ownership** | Decision-facing helpers do not call `prepare_task_for_checkpoint_resume` or related recovery planners |
| **Physical resume ≠ semantic transition** | Restored Decision stage/version unchanged unless explicit lifecycle transition is invoked afterward |
| **Hosting lineage preserved** | `DecisionIdentity.execution` records the hosting Execution lineage at capture time |

Proof gate: DS-NEXUS-02 (`tests/unit/runtime/execution/test_decision_orchestration_recovery.py`).

---

## Decision Resolution

**Decision Resolution** answers: *jaki jest merytoryczny wynik procesu decyzyjnego dla tego scope?*

| Outcome | Meaning |
| ------- | ------- |
| **`ACCEPTED`** | A specific **Decision Version** satisfied required lifecycle gates and is the accepted decision for the scope |
| **`REJECTED`** | Lifecycle executed correctly, but **no** proposed version was accepted as the right decision |
| **`UNRESOLVED`** | The system lacks sufficient basis for a responsible resolution - not a synthetic pass/fail |

Proposal history, disagreement artifacts, and verification lineage **remain preserved** for all three outcomes.

### Decision Resolution ≠ Lifecycle / Execution Termination

**Decision Resolution** and **lifecycle / execution termination** are **independent axes**:

| Axis | Question |
| ---- | -------- |
| **Decision Resolution** | What is the substantive decision outcome? |
| **Lifecycle / Execution Termination** | How did the hosting execution end? (e.g. completed, failed, cancelled, timed out, budget stop, provider outage) |

```text
Decision Resolution = UNRESOLVED
Execution = COMPLETED
```

is a **valid** result - the system ran correctly and responsibly refused an artificial resolution.

```text
Execution = FAILED
```

does **not** automatically imply:

```text
Decision Resolution = REJECTED
```

Infrastructure failure, cancellation, timeout, and budget stop are **execution/lifecycle termination** events - not substitutes for merytoryczne `REJECTED` or `UNRESOLVED`.

---

## Finalization

**Finalization** persists the terminal **authoritative lifecycle outcome** for a decision scope.

| Decision Resolution | Finalization artifact |
| ------------------- | --------------------- |
| **`ACCEPTED`** | **Authoritative Accepted Decision** - binds the accepted **Decision Version** and its typed artifact |
| **`REJECTED`** | **Authoritative Resolution Record** - terminal lifecycle outcome with `REJECTED`; **no** accepted Decision Version is minted |
| **`UNRESOLVED`** | **Authoritative Resolution Record** - terminal lifecycle outcome with `UNRESOLVED`; **no** accepted Decision Version is minted |

There is **no** `fake decision` workaround. Candidate versions and proposal history remain in auditable lineage after finalization.

For a given decision scope, **at most one** terminal authoritative lifecycle outcome may exist - either one **Authoritative Accepted Decision** or one terminal **Authoritative Resolution Record**.

Pure finalize guard semantics define authoritative conflict/idempotency rules in contracts. Durable atomic enforcement belongs to the hosting Execution persistence boundary (DS-CORE-06+).

---

## Decision Artifacts

A **Decision Artifact** is the typed, contract-bound payload a decision carries. Examples:

| Artifact kind | Illustrative use |
| ------------- | ---------------- |
| Classification | incident severity, intent label |
| Recommendation | strategic option ranking |
| Selection | chosen tool, plan branch, hypothesis |
| Plan | structured action proposal |
| Approval | sign-off artifact (distinct from Execution Authorization) |
| Finding | audit or review finding |
| Evidence-backed conclusion | structured conclusion with evidence refs |

**Evidence Claims** remain a valuable, reusable artifact family for evidence-backed decisions ([`PROOF_RECEIPTS.md`](PROOF_RECEIPTS.md) · evidence architecture). Not every decision is an `EvidenceClaimSet`.

Extensibility is **typed and contractual** - registered artifact kinds and schema contracts, not `payload: dict[str, Any]`.

---

## Candidate vs Authoritative Decision

| Concept | Meaning |
| ------- | ------- |
| **Candidate Decision** | A proposed decision version - may fail verification or remain non-final |
| **Authoritative Accepted Decision** | The specific **Decision Version** that satisfied required lifecycle gates - only when Decision Resolution is **`ACCEPTED`** |
| **Authoritative Resolution Record** | Terminal lifecycle outcome for **`REJECTED`** or **`UNRESOLVED`** - authoritative without an accepted Decision Version |
| **Decision Version** | Immutable identity in lineage - `v1 → challenge → v2 → verification → v3 authoritative` |

v1 and v2 remain in auditable lineage after v3 is authoritative. **Never mutate** a prior version in place.

---

## Decision ≠ Authorization ≠ Execution

<a href="assets/fullsize/decision-authorization-execution.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-authorization-execution-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-authorization-execution-light.svg">
  <img
    alt="Three-column diagram separating Authoritative Decision (what the system concluded), Execution Authorization (whether action may proceed), and Execution (what the hosting Execution actually did), with Decision System, Policy, and Execution ownership boxes."
    src="assets/decision-authorization-execution-light.svg"
  >
</picture>
</a>

| Responsibility | Question |
| -------------- | -------- |
| **Authoritative Accepted Decision / Resolution Record** | What did the system finally conclude, recommend, find - or explicitly refuse to resolve? |
| **Execution Authorization** | May this specific action execute in this authority/policy context? |
| **Execution** | What did the hosting Execution actually execute? |

A correct **Authoritative Decision** may still be **blocked**, **deferred**, or **require human approval** before side effects. Policy evaluates at configured execution points - not solely as one post-decision gate ([`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md)).

The Decision System does **not** own Governance, Runtime Policy Engine, Execution Authority, or HITL infrastructure.

---

## Version binding / security

Every:

- Verification Result,
- Challenge,
- Human approval,
- Adjudication Result,
- Policy / authorization record,

**must** bind to exact:

- **Decision ID**
- **Decision Version**
- decision scope
- tenant
- execution identity (`TaskId` / `RunId` / `AttemptId` / **TARGET** `ExecutionId`)

Approval for `v1` does **not** pass to `v2`. Authorization for `v1` does **not** authorize `v2`. Loose context dicts are **not** authority identity.

<a href="assets/fullsize/decision-version-lineage.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-version-lineage-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-version-lineage-light.svg">
  <img
    alt="Version lineage diagram: v1 candidate through challenge and v2 revised to verification and v3 authoritative, with parallel v2A and v2B branches showing no last-write-wins."
    src="assets/decision-version-lineage-light.svg"
  >
</picture>
</a>

---

## Execution / orchestration boundary

**Hard rule:** Decision Lifecycle is hosted by canonical Execution - **no** DecisionRuntime.

The Decision Lifecycle:

- is **not** a separate runtime,
- has **no** own scheduler / retry / checkpoint / budget / execution identity,
- uses the hosting Execution System for execution mechanics and strategy routing.

**Never:** introduce a second DecisionRuntime.

Nexus participates **only** when the hosting Execution selects ORCHESTRATION strategy for work required by the Decision Strategy (e.g. parallel Council analyses, child Executions).

---

## Policy boundary

Verification quality **is not** authorization. The Runtime Policy Engine remains a cross-cutting platform responsibility ([`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md)). Policy may allow, deny, or require human decision for actions flowing from a valid **Authoritative Decision**.

---

## HITL boundary

The legacy **L2 Human Critic** concept is **removed** from the target model. Humans may act as:

- approver,
- adjudicator,
- domain authority,
- policy-required authority,
- escalation target.

The Decision System **invokes** the existing HITL mechanism ([`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md)) - it does **not** implement a Human Engine.

<a href="assets/fullsize/decision-hitl-version-binding.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-hitl-version-binding-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-hitl-version-binding-light.svg">
  <img
    alt="HITL and policy bind execution authorization to exact Decision Version; approval for v1 is invalid after revision mints v2."
    src="assets/decision-hitl-version-binding-light.svg"
  >
</picture>
</a>

---

## Diagnostics boundary

| System | Role |
| ------ | ---- |
| **Diagnostics** | Detect / classify platform operation problems |
| **Decision System** | Lead decision process from proposal to authoritative outcome |

Diagnostics may pass data into investigation / decision flows. Diagnostics is **not** owner of Decision System. Decision System is **not** owner of Diagnostics ([`OBSERVABILITY.md`](OBSERVABILITY.md) DIAG semantics).

---

## Observability / audit

The Decision System must support full reconstruction of:

- decision request,
- selected strategy,
- participant identities / profiles,
- candidate IDs,
- decision versions,
- disagreement artifacts,
- evidence refs,
- verification stages / results,
- challenges,
- revision requests,
- revisions,
- adjudication,
- budget stop,
- escalation,
- human authority record,
- resolution (ACCEPTED / REJECTED / UNRESOLVED),
- authoritative accepted decision ID (when ACCEPTED),
- authoritative resolution record ID (when REJECTED / UNRESOLVED),
- authorization relation to bound version.

**Do not** persist private chain-of-thought.

<a href="assets/fullsize/decision-observability-audit-reconstruction.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-observability-audit-reconstruction-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-observability-audit-reconstruction-light.svg">
  <img
    alt="Audit reconstruction chain from request through proposal, verification, challenge, revision, resolution, human and policy records to execution - excluding private chain-of-thought."
    src="assets/decision-observability-audit-reconstruction-light.svg"
  >
</picture>
</a>

---

## Budgets

Decision Strategy, Verification, Revision, and Adjudication share the **hosting Execution budget**. Deliberation continuation, revision loops, and verification stages must respect configured ceilings. Resume **cannot** increase a previously granted budget ceiling.

---

## Insufficient material and UNRESOLVED

The system must **not** be forced into synthetic `ACCEPTED` / `REJECTED` when:

- evidence is insufficient,
- verifiers conflict irreconcilably,
- parallel branches remain legitimately competing.

**`UNRESOLVED`** is a first-class, auditable **Decision Resolution** outcome - distinct from execution failure.

---

## Concurrency

Parallel proposals are supported:

```text
       → v2A
v1
       → v2B
```

Both branches preserve history. **No last-write-wins** for finalization.

For a given decision scope, **at most one** terminal authoritative lifecycle outcome may exist. Competing valid branches require **adjudication**, **preserved conflict artifact**, or **`UNRESOLVED`** resolution.

---

## Crash / resume

Decision state must be recoverable via the **canonical hosting Execution checkpoint / persistence boundary** - **no** Decision checkpoint engine.

If the hosting Execution uses ORCHESTRATION, Nexus may participate in orchestration checkpointing, but Decision durability **MUST NOT** depend on Nexus being the selected execution strategy.

<a href="assets/fullsize/decision-crash-concurrency.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-crash-concurrency-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-crash-concurrency-light.svg">
  <img
    alt="Crash recovery diagram: canonical Execution checkpoint recovers decision state and version lineage with finalize guard against duplicate authoritative decisions and budget ceiling on resume."
    src="assets/decision-crash-concurrency-light.svg"
  >
</picture>
</a>

| Requirement | Rule |
| ----------- | ---- |
| Durability | Lifecycle state, version lineage, finalize guard state, and revision budget snapshot persisted through canonical hosting Execution checkpoint/persistence boundary |
| Atomic finalization | `DecisionFinalizationPersistence.commit_authoritative_outcome` performs check+commit in one backend transaction — no load/save race |
| Resume | Continue from persisted stage - not full deliberation restart without cause |
| Crash safety | Durable authoritative outcome lookup converges terminal state even when checkpoint lags finalization commit |
| Budget | `DecisionRevisionCheckpointState` is authoritative on resume; runtime policy mismatch fails closed |
| Qualification | Local SQLite/subprocess proof ≠ DS-E2E-06/07 distributed production qualification |

**Durable wire security (DS-REC-INV-01..05):** Decision checkpoint and authoritative outcome records use explicit versioned UTF-8 JSON wire codecs (`intergrax/runtime/execution/decision_durable_wire_codec.py`) with `schema_version` + `record_type` envelopes. Runtime executable object deserialization (e.g. `pickle.loads`) is **forbidden** at the Decision authority boundary; legacy pickle blobs fail closed. Artifact payload reconstruction requires an explicit typed `DecisionArtifactPayloadCodec`; unknown kinds fail closed. SQLite adapters remain storage-only and invoke the shared codec seam.

Proof gates: DS-REC-01 (`tests/unit/runtime/execution/test_decision_finalization_persistence.py`) · DS-REC-02/03 (`tests/unit/runtime/execution/test_decision_durable_recovery.py`) · wire codec (`tests/unit/runtime/execution/test_decision_durable_wire_codec.py`).

---

## Retry boundaries

| Kind | Trigger | Owner |
| ---- | ------- | ----- |
| **Technical retry** | Model / tool / provider failure | Execution System / Reliability |
| **Decision revision** | Semantically insufficient decision content | Decision Lifecycle |
| **Deliberation continuation** | Another Council / strategy round | Decision Strategy |

Do **not** merge these into one generic retry loop.

---

## Plugin / extension posture

- **Decision Strategy** - registered strategies behind stable strategy contract; lifecycle unaware of Council internals.
- **Verification stages** - compositional plugins with typed stage contracts ([`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md)).
- **Decision Artifact kinds** - typed registration - not reflection over loose dicts.

Platform plugin architecture applies at extension boundaries ([`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md)).

Third-party packages contribute through setuptools entry-point groups
``intergrax.decision_strategies``, ``intergrax.decision_verification_stages``, and
``intergrax.decision_artifact_kinds``. Platform discovery and admission run first;
Decision domain composition (`intergrax/runtime/decision_plugin_composition.py`) validates
targets and composes immutable registries. Installation alone does not activate plugins —
explicit ``discover_entry_points=True`` composition is required.

When ``require_manifest_capability_binding=True``, absence of positive Platform Plugin
manifest capability evidence is an admission failure (fail-closed). Plugins without
verifiable manifest binding are not loaded, instantiated, or registered.

---

## Cross-scenario validation

| Scenario | Decision System path |
| -------- | -------------------- |
| **Incident Investigation** | hypotheses → evidence → verification → revision → supported / **UNRESOLVED** conclusion |
| **Strategic Decision** | parallel proposals → disagreement → synthesis → verification → authoritative recommendation |
| **Contract Review** | findings → deterministic + semantic verification → human authority when required |
| **Cyber Incident** | multiple hypotheses → evidence conflicts → verification / adjudication → **UNRESOLVED** possible |
| **Regulated Action** | decision quality accepted → policy requires human → execution authorization → Nexus side effect |

No scenario requires a special-case exception to the frozen fundamentals above.

### Scenario walkthrough - AI Incident Investigation

<a href="assets/fullsize/decision-scenario-incident-investigation.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-scenario-incident-investigation-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-scenario-incident-investigation-light.svg">
  <img
    alt="Incident investigation flow: signals through hypotheses and evidence to candidate conclusion, independent verification, challenge or revision, ending in ACCEPTED or UNRESOLVED."
    src="assets/decision-scenario-incident-investigation-light.svg"
  >
</picture>
</a>

### Scenario walkthrough - Regulated Action

<a href="assets/fullsize/decision-scenario-regulated-action.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decision-scenario-regulated-action-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/decision-scenario-regulated-action-light.svg">
  <img
    alt="Regulated action flow: candidate decision through verification and ACCEPTED resolution, then policy and HITL gates before execution authorization and Nexus side effect."
    src="assets/decision-scenario-regulated-action-light.svg"
  >
</picture>
</a>

```mermaid
stateDiagram-v2
    [*] --> Proposal
    Proposal --> Verification
    Verification --> Resolution: pass
    Verification --> Revision: challenge
    Revision --> Verification
    Resolution --> Finalization: ACCEPTED / REJECTED / UNRESOLVED
    Finalization --> [*]
```

---

## Relationship to Intergrax

| Neighbor | Relationship |
| -------- | ------------- |
| [**Unified Execution Architecture**](UNIFIED_EXECUTION_ARCHITECTURE.md) | Canonical host - lifecycle, identity, strategy routing |
| [**Nexus Execution Flow**](NEXUS_EXECUTION_FLOW.md) | ORCHESTRATION backend when selected by Execution strategy routing |
| [**Unified Execution Runtime**](UNIFIED_EXECUTION_RUNTIME.md) | Profiles, budgets, checkpoint ports |
| [**Decision Verification**](DECISION_VERIFICATION.md) | Compositional verification pipeline |
| [**Decision Deliberation**](DECISION_DELIBERATION.md) | Strategy / Council / deliberation |
| [**Governed Execution**](GOVERNED_EXECUTION.md) | Execution authorization - separate from decision correctness |
| [**Reliability / HITL**](RELIABILITY_FAILURE_AND_HITL.md) | Technical retry; canonical HITL invocation |
| [**Observability**](OBSERVABILITY.md) | Decision audit evidence |
| [**CRITIC_VERIFICATION**](CRITIC_VERIFICATION.md) | **HISTORICAL** migration snapshot — Critic runtime retired (DS-MIG-04) |

---

## Current maturity

Aligned with [`MATURITY_TAXONOMY.md`](../technical/guides/MATURITY_TAXONOMY.md):

| Axis | Level | Rationale |
| ---- | ----- | --------- |
| **Architecture (A)** | **A4** | Frozen target canon established; boundaries to Execution, Policy, HITL, Diagnostics explicit |
| **Implementation (I)** | **I3** | Core lifecycle · revision · verification · governance · execution integration implemented |
| **Production (P)** | **P1** | Decision System is production decision authority; whole-system qualification pending DS-E2E |
| **Evidence (E)** | **E0** | No Decision System Docker E2E qualification completed |

---

## Evidence / proof

| Class | Artifacts |
| ----- | --------- |
| **Architecture** | This hub · [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) · [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md) |
| **Implementation plan** | [`maintainers/plans/DECISION_SYSTEM.md`](../maintainers/plans/DECISION_SYSTEM.md) |
| **Historical Critic snapshot** | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |
| **Public proof** | Not claimed - pending DS-E2E Docker qualification phase |

### Production qualification boundary

The Decision System is **not** production-qualified after:

- unit tests,
- integration tests,
- mocked E2E.

**Production qualification** requires completion of the real **Docker E2E qualification phase** ([`maintainers/plans/DECISION_SYSTEM.md`](../maintainers/plans/DECISION_SYSTEM.md) - Phase DS-E2E).

---

## Go deeper

| Depth | Route |
| ----- | ----- |
| **Extended engineering model** | [`satellites/DECISION_SYSTEM_extended_depth.md`](satellites/DECISION_SYSTEM_extended_depth.md) - identity, versioning, lifecycle, authority, concurrency, recovery, platform boundaries |
| Verification pipeline | [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) |
| Deliberation / Council | [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md) |
| Implementation plan | [`maintainers/plans/DECISION_SYSTEM.md`](../maintainers/plans/DECISION_SYSTEM.md) |
| Historical Critic snapshot | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) |
| Governance | [`GOVERNED_EXECUTION.md`](GOVERNED_EXECUTION.md) |
| Reliability / HITL | [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) |
| Maturity taxonomy | [`MATURITY_TAXONOMY.md`](../technical/guides/MATURITY_TAXONOMY.md) |

---

## Engineering canon

### Cursor read scope (token budget)

**Default:** this hub read-scope block + at-a-glance + one cited diagram section.

- **Implement Decision System:** read this file + [`maintainers/plans/DECISION_SYSTEM.md`](../maintainers/plans/DECISION_SYSTEM.md) hub only.
- **Architecture satellite:** [`satellites/DECISION_SYSTEM_extended_depth.md`](satellites/DECISION_SYSTEM_extended_depth.md) on demand - one per session unless RESUME cites more.
- **Verification slice:** add [`DECISION_VERIFICATION.md`](DECISION_VERIFICATION.md) + [`maintainers/plans/DECISION_VERIFICATION.md`](../maintainers/plans/DECISION_VERIFICATION.md).
- **Deliberation slice:** add [`DECISION_DELIBERATION.md`](DECISION_DELIBERATION.md) + [`maintainers/plans/DECISION_DELIBERATION.md`](../maintainers/plans/DECISION_DELIBERATION.md).
- **Skip** full [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) unless auditing historical migration disposition.
