# Autonomous Work / Virtual Workforce — Architecture & Gap Audit

**Status:** Strategic architecture audit — **NOT** a canonical domain specification and **NOT** an implementation plan  
**Audit date:** 2026-09-02  
**Repository:** `jakbuczarnecki/intergrax`  
**Branch:** `development`  
**Audit baseline HEAD before write:** `7913f5e06a1f76af8120b183acd73ef9994e249c`  
**Decision posture:** architecture-first; no implementation is authorized by this document

---

## 1. Executive conclusion

Intergrax is already unusually well positioned to evolve from a task-oriented agent platform into a platform capable of supporting **long-lived governed autonomous workers**.

The correct architectural move is **not** to create a new `VirtualEmployeeAgent` with more tools, a longer loop, and broader memory. That would collapse responsibilities already deliberately separated across Intergrax and would recreate private execution, policy, observability, recovery, and code-generation logic inside an agent.

The recommended direction is instead:

> **Introduce an Autonomous Work / Virtual Workforce operating layer that composes existing Intergrax domains into a persistent unit of responsibility.**

The worker owns a durable responsibility, goals and work posture. Existing Intergrax mechanisms continue to own execution, cognition, policy enforcement, memory, collaboration, code synthesis, isolation, evidence, diagnostics and hosting.

The strongest strategic finding is that the hard platform primitives are already substantially present:

- agent runtime and orchestration,
- long-running execution and resume semantics,
- durable budget state,
- governance and canonical HITL,
- meaningful-side-effect policy,
- observability and replay/evidence,
- Collaborative Work identity, membership and delegation,
- sandbox execution,
- a dedicated Ephemeral Code Craft subsystem,
- critic / verification mechanisms,
- Tier-3 application hosting.

The largest remaining gap is therefore **semantic and operational**, not primarily cognitive:

- no first-class worker definition,
- no persistent responsibility / goal model,
- no worker lifecycle distinct from task lifecycle,
- no worker-centric control plane,
- no worker-centric observability projection,
- no canonical adaptive-obstacle recovery controller,
- insufficient production isolation guarantees for autonomous generated-code execution,
- incomplete control-plane governance for high-risk worker mutations.

### Strategic verdict

| Question | Verdict |
|---|---|
| Is the market direction worth pursuing? | **YES — strategically relevant** |
| Does Intergrax need a rewrite? | **NO** |
| Should this be implemented as a special agent? | **NO** |
| Should this be only a Tier-3 application? | **NO** |
| Is a new platform-level semantic layer justified? | **LIKELY YES, subject to architecture review** |
| Can existing domains be reused? | **EXTENSIVELY** |
| Is CodeCraft already relevant to unexpected-problem resolution? | **YES — directly** |
| Is CodeCraft production-safe for arbitrary autonomous code today? | **NO** |
| Is governance strong enough conceptually? | **YES, but control-plane and coverage gaps remain** |
| Is a compelling proof feasible without inventing a second runtime? | **YES** |

---

## 2. Scope of this audit

This audit answers five questions:

1. **What should “virtual employee / autonomous worker” mean in Intergrax?**
2. **Which Intergrax capabilities already solve parts of the problem?**
3. **Which existing mechanisms should be reused, extended, or explicitly not duplicated?**
4. **What is genuinely missing?**
5. **What architecture should be reviewed before any implementation begins?**

This audit deliberately does **not**:

- create a new canonical domain pair,
- rename existing domains,
- authorize code changes,
- duplicate `Task`, `Run`, `Execution`, `Principal`, `AgentDefinition`, `CodeCraft`, Sandbox, HOS, HITL or policy semantics,
- claim production readiness,
- claim that “AI employee” equals human replacement.

---

## 3. Market interpretation — what is actually changing

The important market transition is not simply “agents are becoming better”. It is a change in the **unit of abstraction**.

Traditional agent product model:

```text
human gives task
    ↓
agent reasons
    ↓
agent uses tools
    ↓
agent returns result
    ↓
run ends
```

Emerging autonomous-work model:

```text
organization assigns responsibility
          ↓
long-lived worker identity
          ↓
continuous event/work intake
          ↓
prioritization and planning
          ↓
multiple governed executions over time
          ↓
progress against goals / KPI / SLA
          ↓
recovery from obstacles
          ↓
human intervention only when policy/risk requires it
```

This distinction matters because the second model requires infrastructure that ordinary agent demos can avoid:

- stable identity and authority,
- persistence beyond a single run,
- work queues and event subscriptions,
- budgets over time,
- explicit responsibility and success criteria,
- governance of external side effects,
- interruption / pause / resume,
- diagnostics,
- evidence and replay,
- collaboration and delegation,
- safe capability acquisition,
- operator control and fleet views.

The market signal is therefore considered **substantive**, while the marketing claim “autonomous agents replace employees” should not be used as an architecture premise.

Recommended strategic vocabulary:

- externally / product positioning: **Virtual Workforce**, **Digital Workers**, **Autonomous Work**;
- internally / canonical architecture candidate: prefer a neutral term such as **Autonomous Work** or **Worker Runtime** until naming is reviewed.

---

## 4. Core architectural distinction: Agent != Worker

A worker must not become a renamed agent.

### Agent

An agent is a cognitive/execution participant that can reason, choose actions, use tools, delegate, and participate in an execution.

### Worker

A worker is a **persistent operational principal / unit of responsibility** that may create or invoke many executions and may use one or more agents over time.

Target distinction:

```text
Worker responsibility
      │
      ├── goal A
      │    ├── Execution 1 → Agent A
      │    ├── Execution 2 → Agent B
      │    └── Execution 3 → CodeCraft recovery
      │
      ├── goal B
      │    └── Execution 4 → workflow / direct tool path
      │
      └── continuous event intake
```

### Required invariant

> **Worker != AgentDefinition != AgentRun != Nexus Task != WorkItem != HostedApplication.**

Each existing concept retains its current owner and semantics.

---

## 5. Recommended target model

The recommended architecture is a three-level composition:

```text
┌─────────────────────────────────────────────┐
│ VIRTUAL WORKFORCE APPLICATION / CONTROL UI │
│ fleet, goals, KPI, approvals, risk, cost   │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│ AUTONOMOUS WORK / WORKER OPERATING LAYER   │
│                                             │
│ WorkerDefinition                            │
│ WorkerInstance                              │
│ Responsibility / Goal                       │
│ Work intake / subscriptions                 │
│ Worker lifecycle                            │
│ Goal progress                               │
│ Recovery Controller                         │
│ Capability acquisition decision             │
│ Worker-level budgets / policy refs           │
└──────────────────────┬──────────────────────┘
                       │
           EXISTING INTERGRAX DOMAINS
                       │
   ┌──────────┬────────┼─────────┬──────────┐
   ▼          ▼        ▼         ▼          ▼
Execution   Agents  Governance  Memory  Collaborative Work
Runtime
   │                                │
   ├──────── Tools / Integrations ──┤
   ├──────── CodeCraft ─────────────┤
   ├──────── Sandbox ───────────────┤
   ├──────── CVL / Critic ──────────┤
   └──────── Observability / DIAG ──┘
```

### Responsibility split

**Virtual Workforce application owns:**

- human-facing configuration,
- fleet management UI,
- worker dashboards,
- goals / KPI views,
- approval inbox,
- operator controls.

**Autonomous Work layer owns:**

- worker semantics,
- worker lifecycle,
- responsibility and goal bindings,
- work intake policy at worker level,
- worker-level progress state,
- recovery orchestration semantics,
- capability-acquisition decision semantics,
- worker → execution correlation.

**Existing domains continue to own:**

- Execution Runtime — actual execution lifecycle,
- Agents — cognition and agent behavior,
- Governance — authority and policy decisions,
- HITL — human approval semantics,
- Collaborative Work — principal/workspace/delegation/work-plane semantics,
- Memory/UCL/Context — memory and context composition,
- CodeCraft — generated-code lifecycle,
- Sandbox — execution isolation substrate,
- CVL/Critic — verification contracts,
- Observability — canonical execution evidence,
- DIAG — interpretation / problem detection,
- Application Hosting — process/runtime hosting.

---

## 6. EXISTING → REUSE → EXTEND → NEW matrix

### 6.1 Summary matrix

| Concern | Existing Intergrax capability | Action | Why |
|---|---|---|---|
| Cognitive agent | Agent Engine / UAEP / Nexus paths | **REUSE** | Worker should invoke cognition, not replace it |
| Task execution | Unified execution / Nexus execution flow | **REUSE** | Worker creates/owns work intent; execution remains runtime-owned |
| Long-running tasks | long-running options, pause/checkpoint/resume paths | **REUSE + EXTEND integration** | Worker lifetime must exceed individual run lifetime |
| Durable execution budget | durable execution budget ledger | **REUSE** | No second budget system |
| Tools | ToolRuntime / tool catalog | **REUSE** | Known capability remains preferred path |
| Dynamic code | CodeCraft | **REUSE + HARDEN** | Already owns generate→gate→exec→test→verify |
| Sandbox | Sandbox Runtime / hosted sandbox host | **REUSE + HARDEN** | No second sandbox |
| Verification | CVL / Critic | **REUSE** | No second generated-code verifier |
| Governance | Governed Execution / PolicyEngine / RuntimePolicyEngine | **REUSE + EXTEND COVERAGE** | Worker must not own private policy logic |
| Human approval | canonical HITL | **REUSE** | No worker-specific approval runtime |
| External side effects | meaningful-side-effect policy paths | **REUSE + EXTEND COVERAGE** | Required for autonomous actions |
| Collaboration identity | Collaborative Work Principal / Membership | **REUSE** | Worker should map to or compose with canonical Principal semantics |
| Delegation | Collaborative Work delegation | **REUSE** | Authority must remain non-amplifying |
| Shared work | MP WorkItem / Assignment direction | **EXTEND existing roadmap** | Do not turn Nexus Task into business WorkItem |
| Memory | Memory + Context Engineering | **REUSE + PROFILE** | Worker needs durable context without new memory store semantics |
| Evidence | HOS / RuntimeEvent / journal / replay | **REUSE + PROJECT** | Worker dashboard should derive from canonical evidence |
| Diagnostics | DIAG | **REUSE + EXTEND classifiers** | Recovery needs deterministic problem classification |
| Hosted process | Application Hosting | **REUSE** | Worker semantic lifetime is not process lifetime |
| Worker identity | none as first-class semantic object | **NEW** | Needed for durable operational identity |
| Responsibility | none canonical | **NEW** | Core distinction from task agent |
| Worker Goal | no worker-scoped goal entity | **NEW** | Needed for continuous outcome orientation |
| Worker lifecycle | no distinct lifecycle | **NEW** | ACTIVE/IDLE/RECOVERING/etc. cannot be Task state aliases |
| Worker fleet control | none canonical | **NEW** | Operator needs activate/pause/quarantine/stop |
| Worker-centric projection | no canonical projection | **NEW projection, reuse evidence** | Do not create second event store |
| Recovery Controller | no canonical obstacle→strategy selector | **NEW** | Prevent every failure from becoming retry or CodeCraft |
| Capability acquisition policy | fragmented by capability mechanism | **NEW composition semantics** | Decide when/how a worker may acquire missing capability |

---

## 7. Existing capability audit

## 7.1 Execution Runtime / long-running execution

Intergrax already contains long-running task preparation, checkpoint/resume concepts, host queue execution wiring and durable execution budget state.

Relevant evidence includes:

- `intergrax/applications/_shared/task_intake.py` — long-running configuration and checkpoint-on-pause behavior,
- `intergrax/runtime/task/nexus_worker_execution.py` — worker execution path integrating Nexus and durable budget state,
- `intergrax/runtime/execution/budget/persistence.py` — `DurableExecutionBudgetLedger`,
- `docs/project/architecture/GOVERNED_EXECUTION.md`,
- `docs/project/architecture/NEXUS_EXECUTION_FLOW.md`.

### Assessment

**Strong reuse candidate.**

The new worker model must **not** create its own execution state machine for task attempts. Worker lifecycle must sit above the execution plane.

Required relation:

```text
WorkerInstance
  ├─ can remain ACTIVE while no Task exists
  ├─ can own/receive multiple WorkItems over time
  ├─ may create many Tasks/Runs
  └─ does not replace Task/Run/Attempt semantics
```

---

## 7.2 Governed Execution

Governed Execution is one of the strongest foundations for autonomous workers.

Current architecture establishes:

- execution-centric policy,
- explicit effective authority,
- policy outcomes including ALLOW / DENY / REQUIRE_HUMAN,
- canonical HITL,
- tool authorization and declarative policy,
- meaningful-side-effect evaluation on demonstrated paths,
- post-run governance on configured paths,
- authority inheritance where children may narrow but must not expand parent authority.

### Worker extension

Proposed authority chain:

```text
Organization / tenant authority
        ↓
Worker effective authority
        ↓
Execution effective authority
        ↓
child Execution / Agent
        ↓
Tool / CodeCraft operation
```

### New invariant

> **Capability may grow; authority must not self-expand.**

A worker may learn how to parse a new format, synthesize a transformation, or adapt to a changed API. It must never be able to acquire additional credentials, policy scopes, workspace membership, or production authority merely because those rights would help complete its goal.

### Critical gap

`CONTROL_PLANE_MUTATION` is documented as a governance **GAP** in the current architecture. Autonomous workers increase the importance of this gap because worker activation, policy/budget changes, capability promotion, quarantine, fleet changes and runtime configuration are themselves privileged control-plane mutations.

**Priority:** P0 before broad autonomous operation.

---

## 7.3 Collaborative Work / Multiplayer foundation

Collaborative Work already provides a strong organizational semantics foundation:

- `Principal`,
- explicit `WorkspaceMembership`,
- authority grants,
- scoped non-amplifying `Delegation`,
- effective-authority composition,
- future `WorkItem`, `Assignment`, `WorkArtifact`, `Decision`, `ContextView`, Activity semantics.

Important existing invariants include:

- `Principal != AgentDefinition != AgentRun != RequestIdentity`,
- `WorkItem != Nexus Task`,
- delegation cannot amplify authority,
- membership is explicit,
- missing mandatory authority/policy evaluation fails closed.

### Assessment

This should be **reused directly**, not recreated inside a worker feature.

A digital worker should be represented as, or have an explicit binding to, a canonical collaborative `Principal` of kind `AGENT` or future reviewed worker-specific principal classification **only if the existing principal taxonomy proves insufficient**.

Do not introduce `VirtualWorkerPrincipal` merely for branding.

### Gap

MP-2+ shared work concepts remain future work. A mature Virtual Workforce depends heavily on `WorkItem`, `Assignment`, artifacts and decisions because long-lived workers need a business work plane separate from execution tasks.

---

## 7.4 Observability / evidence

Intergrax already owns a canonical execution evidence spine:

```text
Execution Runtime
   ↓
RuntimeEvent / HOS
   ↓
persistence
   ↓
Unified Run Journal / projections / replay / export
```

Current architecture explicitly rejects vendor telemetry as the execution source of truth.

### Assessment

This is the correct foundation for workforce observability.

Do **not** create a separate `WorkerEventBus` that becomes a second source of truth for runtime execution.

Instead introduce a worker-oriented projection:

```text
WorkerInstance
   ↓
Responsibility / Goal
   ↓
WorkItem / Assignment
   ↓
Task / Run / Attempt / Execution
   ↓
RuntimeEvent
```

Worker-domain lifecycle facts that are not execution facts may use an appropriate platform signal or future domain event family, but they must not fabricate Task/Run/Execution identity.

### Operator dashboard target

A worker dashboard should answer:

- Is the worker active, idle, blocked, degraded, recovering, paused, quarantined?
- Which responsibility and goals are currently assigned?
- What work is in progress?
- What work was completed autonomously?
- What required human intervention?
- What was denied by policy?
- Which recoveries were attempted?
- Which capabilities were generated?
- How much did it cost?
- What SLA/KPI did it achieve?
- Can every meaningful action be traced to canonical evidence?

---

## 7.5 CodeCraft — strategic fit

CodeCraft is the most direct existing answer to the requirement that an autonomous worker should be able to overcome previously unplanned capability gaps.

Current canonical architecture already defines the purpose:

> governed ephemeral code generation when catalog tools are insufficient.

The existing lifecycle is conceptually:

```text
Agent goal
   ↓
CodeCraftOrchestrator
   ↓
Generate
   ↓
Static Gate
   ↓
Policy / HITL
   ↓
Sandbox
   ↓
Execute + Test
   ↓
CVL / Critic verdict
   ↓
bounded retry OR CraftResult
   ↓
dispose
```

Current CodeCraft also already provides:

- task/craft-scoped ephemeral tools,
- bounded iterations,
- total execution-time budget,
- `disabled`, `dry_run`, `assist_only`, `supervised`, `autonomous` modes,
- static checks,
- governed execution integration,
- sandbox substrate reuse,
- tests and verification,
- result promotion semantics.

### Strategic conclusion

**Do not build a new coding environment for Virtual Workforce.**

Use CodeCraft as the canonical generated-code subsystem and harden it where necessary.

### Critical current limitations

The canonical CodeCraft document explicitly warns that the subsystem is **not** universally production-qualified for arbitrary generated-code execution. Current limitations include:

- authority defects recorded by the existing audit,
- session identity binding concerns,
- HITL self-assertion concern,
- `local` isolation is a development substrate rather than a hard security boundary,
- `container` does not yet guarantee a distinct OCI boundary,
- partial network-egress enforcement,
- cloud/container resolver behavior may downgrade to local when hosted isolation cannot be resolved,
- no claim of hostile-code / sandbox-escape qualification.

These become **P0** for autonomous worker use.

---

## 7.6 Sandbox

Intergrax already has:

- `runtime/sandbox`,
- Sandbox session lifecycle,
- sandbox tool and skill bundles,
- hosted sandbox contracts and wiring,
- CodeCraft sandbox resolver.

### Assessment

Reuse the sandbox domain. Do not create `WorkerSandbox`.

### Required production posture for autonomous CodeCraft

For worker-triggered generated code, requested isolation must be enforceable as a hard requirement:

```text
requested cloud/container isolation
+ substrate unavailable
= DENY / controlled failure
```

Not:

```text
requested cloud/container isolation
+ substrate unavailable
= silently execute locally
```

Minimum production-grade isolation requirements should include, subject to security design review:

- ephemeral filesystem boundary,
- explicit mount policy,
- network egress allowlist / deny-by-default option,
- CPU quota,
- memory quota,
- wall-clock timeout,
- process/fork quota,
- scoped credentials,
- secret brokering rather than broad environment inheritance,
- dependency/package policy,
- artifact ingress/egress controls,
- deterministic cleanup,
- audit evidence of actual substrate and enforced controls,
- anti-downgrade guarantee.

---

## 7.7 Memory and Context Engineering

Workers require continuity, but worker memory must not become a new memory subsystem.

Reuse:

- task/session memory where appropriate,
- durable long-term / organizational memory mechanisms where configured,
- Context Engineering and UCL composition,
- principal/workspace scoping from Collaborative Work.

### New need

A worker profile may define what kinds of context it is allowed to recall and retain across work episodes.

Potential configuration concepts:

```text
WorkerMemoryProfile
  responsibility_memory_scope
  allowed_context_sources
  retention_policy_ref
  cross_workitem_recall
  organization_memory_access
  sensitive_context_policy_ref
```

These should **reference** existing memory/context capabilities, not implement storage.

---

## 7.8 Application Hosting

A virtual worker may be always available, but semantic worker lifetime must not equal process lifetime.

Correct separation:

```text
Hosted application process
    may restart

WorkerInstance
    remains durable

Task/Run
    may pause/resume/retry
```

Application Hosting should host the application/control surface and runtime composition; it must not become the source of truth for worker identity or business responsibility.

---

## 8. What is genuinely NEW

## 8.1 WorkerDefinition

A durable definition of an operational worker.

Candidate conceptual fields — **not a frozen schema**:

```text
WorkerDefinition
  worker_definition_id
  display_name
  role
  responsibility_refs
  default_goal_policy
  principal_binding
  workspace_scope
  authority_profile_ref
  governance_profile_ref
  budget_profile_ref
  memory_profile_ref
  capability_profile_ref
  codecraft_profile_ref
  risk_profile_ref
  schedule_profile_ref
  escalation_policy_ref
  collaboration_profile_ref
  observability_profile_ref
```

The schema must prefer references to existing domain-owned profiles rather than embedding duplicate configurations.

---

## 8.2 WorkerInstance

A durable operational instance created from a definition.

Conceptual identity:

```text
WorkerDefinition
      ↓ instantiate
WorkerInstance
      ↓
assigned responsibilities / goals / workspace
      ↓
0..N WorkItems over time
      ↓
0..N Executions over time
```

A WorkerInstance is not an execution attempt.

---

## 8.3 Responsibility

The key abstraction separating worker from task agent.

Examples:

- “Process all incoming purchase orders according to policy.”
- “Maintain supplier integration health and ensure order delivery workflows continue operating.”
- “Investigate qualifying incidents and keep the incident record current until closure.”

A responsibility defines a persistent area of ownership rather than one prompt.

It may produce many goals and work items.

---

## 8.4 WorkerGoal

A durable outcome target with measurable progress.

Potential semantics:

```text
WorkerGoal
  goal_id
  responsibility_id
  objective
  success_criteria
  metric_refs
  SLA/SLO refs
  deadline / cadence
  priority
  status
  progress projection
```

Important distinction:

> Goal is not a model prompt and not a Task description.

A goal may generate multiple work items and executions.

---

## 8.5 Worker lifecycle

Worker lifecycle must be distinct from Task/Run lifecycle.

Candidate states:

```text
PROVISIONING
ACTIVE
IDLE
WORKING
WAITING_EXTERNAL
WAITING_FOR_HUMAN
RECOVERING
DEGRADED
PAUSED
QUARANTINED
STOPPED
```

These are **candidate semantics**, not implementation enums.

Examples:

- worker `ACTIVE` + no open work → `IDLE`,
- worker `ACTIVE` + one execution `PAUSED` does not necessarily mean worker is globally paused,
- worker `QUARANTINED` blocks creation/continuation of privileged work according to policy,
- process restart must not recreate a new worker identity.

---

## 8.6 Worker Control Plane

A fleet requires explicit operator controls.

Candidate operations:

- register / instantiate,
- activate,
- pause,
- resume,
- stop,
- quarantine,
- release quarantine,
- assign responsibility,
- assign/reassign work,
- change budget profile,
- change policy profile,
- change allowed capability profile,
- inspect current state,
- inspect recoveries,
- revoke worker authority,
- terminate active work under controlled semantics.

### Mandatory invariant

Every security-sensitive mutation of worker operational state is a **control-plane mutation** and must be governed, evidenced and scoped.

This is directly blocked by the existing platform-wide control-plane governance gap and therefore requires P0 architecture work.

---

## 9. Adaptive obstacle recovery

This is the core capability that turns an “agent that fails” into an autonomous worker that can continue pursuing a goal safely.

Do not implement this as “on error, call CodeCraft”.

A canonical Recovery Controller must first classify the obstacle.

Target decision flow:

```text
work attempt fails / progress stalls
            ↓
DIAG / deterministic evidence
            ↓
Obstacle Classification
            ↓
┌───────────────────────────────────────────┐
│ transient fault       → retry/backoff     │
│ dependency unavailable → wait/reschedule  │
│ rate limit             → throttle/wait    │
│ credential revoked     → escalate         │
│ policy DENY            → stop / no bypass │
│ ambiguous business act → human decision   │
│ known alternate tool   → replan           │
│ schema/API drift       → adaptive recovery│
│ missing capability     → acquisition path │
│ suspicious input       → quarantine       │
└───────────────────────────────────────────┘
```

### Key rule

> Recovery changes strategy or capability. It does not weaken policy.

---

## 10. Adaptive Capability Recovery

Recommended name for the sub-flow that handles safe acquisition of a missing capability.

```text
Goal blocked
   ↓
missing capability confirmed
   ↓
search existing Tool / Skill / Integration / known capability
   ↓
existing suitable capability?
   ├─ yes → use governed existing capability
   └─ no
       ↓
capability acquisition allowed by worker profile?
       ├─ no → escalate / fail safely
       └─ yes
           ↓
classify risk tier
           ↓
CodeCraft / docs / browser / schema inspection
           ↓
generate candidate helper / adapter
           ↓
static gate
           ↓
policy / required approval
           ↓
hard sandbox
           ↓
tests
           ↓
critic / CVL verification
           ↓
ephemeral use / shadow / canary / promotion policy
           ↓
resume original work
```

This preserves CodeCraft ownership of synthesis while adding the missing worker-level decision about **whether capability acquisition is the right recovery strategy**.

---

## 11. Capability autonomy tiers

A worker should not have a binary “CodeCraft enabled” flag as its complete risk model.

Candidate conceptual tiers:

| Tier | Example | Autonomous posture |
|---|---|---|
| **A0 — Known Capability** | existing catalog tool | execute under normal policy |
| **A1 — Ephemeral Safe** | parser / transform / local structured-data helper | generate, verify, use ephemerally under hardened sandbox |
| **A2 — Scoped Adaptive** | temporary API adapter with restricted egress and scoped secret | generate + test + use only inside narrow scope if policy permits |
| **A3 — Production Change** | update durable connector / workflow code | generate/test/shadow/canary; promotion requires explicit policy and often human approval |
| **A4 — Authority Change** | acquire new credential, broaden DB scope, disable policy | **never self-authorized** |

Normative direction:

> **SELF-EXTENDING CAPABILITY: conditionally allowed.**  
> **SELF-EXTENDING AUTHORITY: forbidden.**

---

## 12. Scenario 1 — unknown order attachment

Responsibility:

> Process incoming customer orders according to company policy.

Unexpected input:

```text
order_1237.xyz
```

Target flow:

```text
email event
  ↓
worker accepts work
  ↓
known parser path fails: UnsupportedFormat
  ↓
DIAG evidence
  ↓
Recovery Controller
  ↓
classify: missing parser capability
  ↓
existing parser search → none
  ↓
A1 capability acquisition allowed
  ↓
CodeCraft generates parser
  ↓
static gate
  ↓
hardened sandbox
  ↓
run parser against attachment
  ↓
validate output against Order schema
  ↓
CVL / deterministic tests pass
  ↓
ephemeral capability usable in current scope
  ↓
original order workflow resumes
  ↓
order completed
  ↓
worker dashboard records recovery and evidence
```

What must **not** happen:

- raw local Python execution outside CodeCraft/Sandbox,
- automatic permission expansion,
- persistence into global ToolRegistry without explicit promotion semantics,
- silent bypass of malicious-input scanning,
- loss of traceability between failure, recovery and resumed work.

---

## 13. Scenario 2 — vendor API drift

Responsibility:

> Keep supplier order synchronization operating within SLA.

Failure:

```text
existing vendor endpoint returns HTTP 410 / incompatible schema
```

Target flow:

```text
integration failure
   ↓
DIAG: likely API/contract drift
   ↓
Recovery Controller
   ↓
worker may inspect approved vendor documentation
   ↓
detect new API contract
   ↓
CodeCraft generates candidate adapter
   ↓
static / security gates
   ↓
restricted network sandbox
   ↓
contract tests
   ↓
test credentials only
   ↓
verification
   ↓
shadow execution
   ↓
canary if policy allows
   ↓
A3 promotion path
   ↓
human approval where required
   ↓
production integration update
   ↓
original synchronization resumes
```

The worker may autonomously **discover, implement and prove** a fix while still being prohibited from independently granting itself production authority.

---

## 14. Governance requirements for worker autonomy

## 14.1 Worker-level effective authority

Worker authority must be explicit and durable, derived from canonical identity/workspace/organization semantics.

No implicit authority from:

- display role,
- worker name,
- assigned goal text,
- tenant/workspace identifier alone,
- model statement,
- generated code,
- human approval unrelated to the governed action.

## 14.2 No authority expansion through recovery

```text
Worker authority = X
       ↓
Recovery may choose different strategy
       ↓
resulting execution authority <= X
```

## 14.3 Secrets

Generated code should receive only purpose-scoped credentials through a brokered mechanism. Full parent environment inheritance is an unacceptable target posture for autonomous production CodeCraft.

## 14.4 Network

Network access must be explicit by capability/risk profile. Documentation access and production API access are separate permissions.

## 14.5 Human approval

Reuse canonical HITL.

Worker-specific UI may surface approvals, but must not create worker-local approval semantics.

## 14.6 Kill switch / quarantine

A worker must support a governed emergency containment path. Quarantine should prevent new privileged execution and define controlled handling of active work.

---

## 15. Observability and workforce dashboard

A worker dashboard must be an **operational projection**, not an LLM-generated status narrative.

Canonical facts should derive from worker state plus Intergrax evidence.

Minimum dashboard model:

```text
Worker: Order Operations Worker
Status: WORKING
Risk posture: NORMAL
Current responsibility: Process incoming orders

Goal completion: 97.8%
Orders processed today: 842
Autonomous completions: 817
Human interventions: 11
Recovery attempts: 8
Successful recoveries: 7
Policy denials: 6
Generated capabilities: 2
Cost today: ...
SLA compliance: ...

Current work:
  order #12973
    parse attachment
    obstacle: unknown format
    recovery: CodeCraft
      generation
      static gate PASS
      sandbox PASS
      verification PASS
    resumed
```

Required operator views:

- fleet overview,
- worker detail,
- goals/KPI,
- open work,
- recovery history,
- policy/approval history,
- generated capabilities,
- costs/budgets,
- collaboration/delegation,
- evidence drill-down,
- current risk state,
- quarantine/kill controls.

---

## 16. Readiness assessment

Percentages are architectural estimates for prioritization only; they are not test coverage or maturity certification.

| Area | Approx. readiness for Autonomous Work | Reason |
|---|---:|---|
| Agents / cognition | 85% | strong existing runtime; worker is not a replacement |
| Tools / capabilities | 90% | mature catalog/runtime foundation |
| Orchestration / execution | 85% | strong execution primitives and Nexus |
| Long-running execution | 75–80% | important pieces exist; worker-level persistence still missing |
| Memory / context | 80% | strong foundation; needs worker-scoped composition |
| Governance foundations | 85% | correct architecture, but incomplete coverage/control-plane gap |
| Observability / evidence | 85–90% | strong canonical spine; worker projection missing |
| HITL | 85% | canonical path already exists |
| Identity / authority | ~80% | strong building blocks; worker binding must be designed |
| Collaborative Work / Multiplayer | 60–65% | MP-1 strong; MP-2+ crucial future semantics |
| CodeCraft orchestration | 80–85% | directly aligned with missing-capability recovery |
| Production generated-code isolation | 45–55% | current sandbox/isolation warnings are material |
| Adaptive obstacle recovery | 30–40% | mechanisms exist but no canonical controller |
| Worker lifecycle | 10–20% | largely new semantics |
| Responsibility / goals / KPI | ~10% | first-class model missing |
| Fleet / worker control plane | ~10% | new platform/application capability required |
| Virtual Workforce product | ~15% | product composition not yet built |

### Interpretation

The platform is **not 80% complete as a product**.

A more accurate statement is:

> A large majority of the technically difficult lower-level primitives appear to exist or have direct equivalents, while the worker semantic layer, worker control plane, adaptive recovery composition and production isolation hardening remain substantial work.

---

## 17. Priority gaps

## P0 — must be resolved before production-style autonomous workers

### P0.1 CodeCraft authority defects

Close existing critical identity / authority / HITL defects recorded by the CodeCraft audit.

### P0.2 Isolation anti-downgrade

If a profile requires container/cloud isolation, failure to resolve that substrate must fail closed.

### P0.3 Real production isolation boundary

Provide and qualify an actual strong container/cloud execution boundary for autonomous generated code.

### P0.4 Network / secret enforcement

Move from partial/profile-only intent toward enforceable egress and scoped credential semantics.

### P0.5 Control-plane governance

Close enough of `CONTROL_PLANE_MUTATION` governance to safely manage worker activation, quarantine, budgets, policy changes and capability promotion.

### P0.6 Evidence correlation

Ensure worker/recovery/craft actions correlate back to canonical execution/evidence identities without private trace authority.

---

## P1 — required for a credible worker platform

### P1.1 Worker semantic contracts

Define WorkerDefinition, WorkerInstance, Responsibility and WorkerGoal.

### P1.2 Worker lifecycle

Define persistent lifecycle independent of Task/Run.

### P1.3 Worker → Principal / Workspace / Authority binding

Reuse Collaborative Work and governance without inventing parallel identity.

### P1.4 Worker work intake

Define event subscriptions, assignments and work acceptance semantics.

### P1.5 Worker-level budget semantics

Compose existing run/execution budgets into longer-lived worker limits and accounting windows.

### P1.6 Recovery Controller

Canonical obstacle classification and recovery strategy selection.

### P1.7 Capability acquisition policy

Define when a missing capability may trigger CodeCraft and at what autonomy tier.

### P1.8 Worker observability projection

Build deterministic views from worker state + HOS/DIAG evidence.

### P1.9 Worker control plane

Activate/pause/stop/quarantine/inspect under governance.

---

## P2 — needed for differentiated Virtual Workforce product

- worker-to-worker collaboration and delegation over MP-2+ work semantics,
- artifact/decision collaboration,
- goal optimization over long horizons,
- dynamic work prioritization,
- fleet-level capacity and scheduling,
- reusable capability promotion pipelines,
- shadow/canary policies for generated integration changes,
- workforce simulation / chaos arena,
- UI/UX for organization-level supervision,
- workforce cost/performance analytics.

---

## 18. Architecture invariants proposed for review

These are proposed audit conclusions, not yet canonical platform invariants.

### AW-INV-01 — Worker is not Agent

`WorkerInstance != AgentDefinition != AgentRun`.

### AW-INV-02 — Worker is not execution

`WorkerInstance != Task != Run != Attempt != Execution`.

### AW-INV-03 — Responsibility outlives execution

A responsibility may span zero or many executions and survive process/runtime restarts.

### AW-INV-04 — Work plane stays separate from execution plane

Business `WorkItem`/Assignment semantics remain Collaborative Work-owned; Nexus Task remains execution-owned.

### AW-INV-05 — Capability growth does not imply authority growth

Worker recovery may synthesize capabilities but never self-expand effective authority.

### AW-INV-06 — Code generation has one canonical path

Autonomous generated executable code uses CodeCraft + approved sandbox substrate; no agent-private subprocess/code loop.

### AW-INV-07 — Isolation requirement is non-downgradable

A worker requiring hardened isolation must fail closed if that substrate cannot be proven.

### AW-INV-08 — Governance remains execution-centric

Worker operations compose with Governed Execution; no `WorkerPolicyEngine` competes with platform policy.

### AW-INV-09 — HITL remains canonical

Worker UI may request/show approvals, but decision semantics remain canonical HITL.

### AW-INV-10 — Evidence has one runtime truth

Worker dashboards derive execution truth from HOS/RuntimeEvent evidence; no second execution history.

### AW-INV-11 — Recovery cannot bypass DENY

Policy denial is not an obstacle that the worker is allowed to “solve”.

### AW-INV-12 — Control-plane mutations are governed

Activation, quarantine, policy changes, budget changes and capability promotion require explicit authorization/evidence.

### AW-INV-13 — Worker lifetime is not host-process lifetime

Restarting a host must not create a new worker identity or lose responsibility state.

### AW-INV-14 — Reuse before create

Existing Tool/Skill/Integration capabilities are preferred over CodeCraft. Capability synthesis is a recovery path, not a default planning shortcut.

### AW-INV-15 — Ephemeral before durable promotion

Generated capability is task/craft scoped by default; durable promotion is a separate governed lifecycle.

---

## 19. Anti-patterns explicitly rejected

### Anti-pattern A — `VirtualEmployeeAgent(BaseAgent)`

Rejected because it would mix worker identity, goals, persistence, execution, governance and recovery into the cognitive layer.

### Anti-pattern B — infinite agent loop

```text
while True:
    think()
    act()
```

Rejected because continuous autonomy requires durable lifecycle, scheduling, budgets, pause/resume, recovery and governance — not an immortal model loop.

### Anti-pattern C — second sandbox

Rejected. Harden existing Sandbox Runtime and CodeCraft substrate resolution.

### Anti-pattern D — second policy engine

Rejected. Worker-specific rules must compose through existing governance mechanisms.

### Anti-pattern E — second observability stack

Rejected. Worker UI is a projection over canonical evidence plus worker-domain state.

### Anti-pattern F — Task as worker

Rejected. A worker may exist with no active task and may execute thousands of tasks over its lifetime.

### Anti-pattern G — generated code automatically becomes durable tool

Rejected. CodeCraft already separates ephemeral capability from durable promotion semantics.

### Anti-pattern H — “self-healing” means retry everything

Rejected. Recovery requires deterministic classification and risk-aware strategy selection.

---

## 20. Recommended first proof — Autonomous Order Operations Worker

### Responsibility

> Continuously process incoming customer orders according to company policy and maintain service continuity within the defined SLA.

### Why this proof

It exercises nearly every differentiating capability in one comprehensible business scenario:

- event-driven intake,
- persistent responsibility,
- many work items,
- email/files,
- data extraction,
- API integrations,
- policy/authority,
- meaningful side effects,
- long-running execution,
- human approval,
- abnormal conditions,
- CodeCraft recovery,
- sandboxing,
- observability,
- business KPI.

### Failure/chaos corpus

Inject at least:

- normal email order,
- PDF order,
- spreadsheet order,
- unknown document format,
- corrupted attachment,
- prompt-injection content in document,
- missing customer data,
- duplicate order,
- contradictory order values,
- transient API timeout,
- rate limit,
- vendor API schema drift,
- HTTP 410 / endpoint migration,
- revoked credential,
- supplier outage,
- policy-prohibited action,
- malicious generated-code temptation,
- sandbox unavailable,
- required human business decision.

### Arena metrics

| Metric | Purpose |
|---|---|
| Goal completion rate | did the worker achieve the business outcome? |
| Autonomous completion rate | how much work needed no human intervention? |
| Human intervention rate | operational burden on humans |
| Recovery attempt rate | how often environment challenged the worker |
| Recovery success rate | ability to overcome obstacles |
| Mean recovery time | speed of adaptation |
| Policy violation rate | target must be zero |
| Unauthorized side effects | target zero |
| Unauthorized network egress | target zero |
| Isolation downgrade count | target zero |
| False escalation rate | unnecessary human burden |
| Missed escalation rate | dangerous over-autonomy |
| Generated capability pass rate | CodeCraft usefulness |
| Generated capability rollback rate | adaptive quality |
| Trace/evidence completeness | auditability |
| Cost per completed order | economics |
| SLA adherence | business value |
| Duplicate/replayed side effects | reliability/idempotency |

---

## 21. Proposed strategic roadmap

This is a **review roadmap**, not an implementation authorization.

### Stage 0 — Architecture & benchmark

**Current stage.**

- market validation,
- repository as-built audit,
- architecture mapping,
- gap identification,
- explicit reuse boundaries,
- risk identification.

Exit gate: independent review accepts/corrects this document.

### Stage 1 — Autonomous Work semantics

Freeze the semantic model:

- WorkerDefinition,
- WorkerInstance,
- Responsibility,
- WorkerGoal,
- lifecycle,
- relation to Principal, WorkItem and Execution.

No new runtime until ownership boundaries are accepted.

### Stage 2 — Persistent Worker Runtime composition

Implement only the missing orchestration above existing execution primitives:

- durable worker state,
- event/work intake,
- work dispatch to canonical execution,
- persistence/restart behavior,
- worker-level budget accounting.

### Stage 3 — Workforce governance / control plane

- worker authority binding,
- governed fleet mutations,
- pause/stop/quarantine,
- policy/budget changes,
- approval integration,
- emergency containment.

### Stage 4 — Worker observability

- deterministic worker projection,
- fleet dashboard,
- KPI and goals,
- recovery visibility,
- evidence drill-down,
- cost/risk views.

### Stage 5 — Safe Adaptive Capability

- close CodeCraft authority defects,
- strict anti-downgrade,
- real strong isolation boundary,
- network/secret enforcement,
- hostile-input and sandbox escape test strategy,
- capability autonomy tiers.

### Stage 6 — Adaptive Recovery

- obstacle taxonomy,
- recovery decision semantics,
- DIAG integration,
- capability acquisition path,
- resume original work after recovery.

### Stage 7 — Collaborative Workforce

- MP-2+ WorkItem / Assignment integration,
- worker-human-worker delegation,
- artifacts,
- decisions,
- activity/provenance.

### Stage 8 — Autonomous Order Operations proof

- end-to-end scenario,
- chaos corpus,
- arena metrics,
- visible operator dashboard,
- evidence pack.

### Stage 9 — Production hardening

- security,
- chaos/reliability,
- scale,
- credential isolation,
- network isolation,
- recovery abuse cases,
- policy bypass attempts,
- disaster/restart semantics.

### Stage 10 — Virtual Workforce product

- worker builder/configuration,
- fleet management,
- role/responsibility templates,
- goals/KPI,
- approvals,
- organizational deployment experience.

---

## 22. Required architecture review questions before implementation

The independent review should explicitly answer:

1. Is a new `AUTONOMOUS_WORK` domain justified, or can the missing semantics live cleanly within an existing domain without violating ownership?
2. Should `WorkerDefinition` and `WorkerInstance` be platform contracts or application-level contracts with a thinner platform primitive?
3. How exactly does Worker bind to Collaborative Work `Principal` without creating a competing identity system?
4. Is Responsibility a standalone entity or a typed class of goal/work ownership?
5. Which domain owns WorkerGoal truth and KPI projection?
6. How do worker-level budgets compose with Run/Execution budgets without duplicating ledgers?
7. What is the exact worker lifecycle and which transitions require governance?
8. What is the boundary between Recovery Controller and DIAG?
9. What is the boundary between Recovery Controller and Reliability retry semantics?
10. What is the boundary between capability discovery and CodeCraft?
11. Which CodeCraft autonomy tiers are acceptable for production?
12. What exact sandbox substrate qualifies for A1/A2/A3?
13. How is egress policy proven at runtime?
14. How are secrets scoped and brokered into generated-code sandboxes?
15. What evidence is required before an ephemeral helper can be promoted durably?
16. Which control-plane gaps must close before worker fleet mutation is exposed?
17. How does worker-centric observability avoid becoming a competing execution source of truth?
18. Which MP-2+ Collaborative Work phases become dependencies versus optional enhancements?
19. How are human approvals scoped so they cannot accidentally expand unrelated authority?
20. What is the minimum proof that demonstrates real autonomous work rather than a scripted demo?

---

## 23. As-built evidence map

Primary repository evidence used by this audit:

| Area | Canonical / relevant source |
|---|---|
| CodeCraft architecture | `docs/project/architecture/CODE_CRAFT.md` |
| CodeCraft ADR | `docs/project/technical/adr/entries/2026-06-10/ADR-CODECRAFT-001.md` |
| CodeCraft runtime | `intergrax/runtime/codecraft/` |
| CodeCraft tools | `intergrax/tools/providers/codecraft/` |
| Sandbox runtime | `intergrax/runtime/sandbox/` |
| Sandbox tools | `intergrax/tools/providers/sandbox/` |
| Hosted sandbox contract | `intergrax/integrations/contracts/sandbox_host.py` |
| Governance architecture | `docs/project/architecture/GOVERNED_EXECUTION.md` |
| Runtime policy | `intergrax/runtime/policy/` |
| HITL/interrupt path | `intergrax/runtime/interrupts/` and governed execution docs |
| Collaborative Work | `docs/project/architecture/COLLABORATIVE_WORK.md` |
| Multiplayer capability | `docs/project/capabilities/architecture/MULTIPLAYER_AI.md` |
| Observability | `docs/project/architecture/OBSERVABILITY.md` |
| Execution map/readiness | `docs/project/architecture/UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md`, `UNIFIED_EXECUTION_IMPLEMENTATION_READINESS.md` |
| Long-running task intake | `intergrax/applications/_shared/task_intake.py` |
| Durable budget ledger | `intergrax/runtime/execution/budget/persistence.py` |
| Nexus worker execution | `intergrax/runtime/task/nexus_worker_execution.py` |
| Existing worker-like Tier-3 proof | `applications/governed_contractor_application/` |

---

## 24. External market/reference signals

External material is used only to validate that the problem class is commercially relevant; it does not define Intergrax architecture.

Relevant signal categories observed during the audit:

- Microsoft Agent 365 / agent control-plane direction: registry, governance, observability, identity and fleet-style management.
- Salesforce Agentforce / digital labor positioning: durable business-role framing beyond one-shot task agents.
- Anthropic Claude Code sandbox/autonomy work: stronger isolation as an enabler of greater autonomous execution.
- Gartner caution around treating autonomous agents as literal labor replacement without identity/governance discipline.

Architecture decisions in this document are grounded in the Intergrax repository, not copied from those products.

---

## 25. Final architecture recommendation

The recommended target is:

```text
ORGANIZATION / HUMAN
        │
        ▼
VIRTUAL WORKFORCE APPLICATION
        │
        ▼
AUTONOMOUS WORK SEMANTIC LAYER
WorkerDefinition / WorkerInstance
Responsibility / Goal / lifecycle
Recovery Controller
        │
        ├──────── Collaborative Work
        ├──────── Governance / HITL
        ├──────── Memory / Context
        └──────── worker→execution composition
                     │
                     ▼
             EXECUTION RUNTIME
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
      Agents       Tools        CodeCraft
                                  │
                              Static Gate
                                  │
                              Governance
                                  │
                               Sandbox
                                  │
                           Test / CVL / Critic
                                  │
                           ephemeral result
                                  │
                          resume original work

ALL EXECUTION FACTS
        ↓
OBSERVABILITY / DIAG
        ↓
WORKER-CENTRIC PROJECTIONS
        ↓
OPERATOR DASHBOARD
```

### Core product thesis

Intergrax should aim to support systems where an organization assigns a **durable responsibility**, not merely a prompt; the platform then safely coordinates the work required to pursue that responsibility over time.

### Core technical differentiator

The most valuable differentiator is not “an agent can write code”.

It is:

> **A governed autonomous worker can encounter an unforeseen capability gap, diagnose it, acquire or synthesize a narrowly scoped capability through the canonical CodeCraft path, prove it in hardened isolation, remain inside its original authority, and resume the original business goal with complete evidence.**

That composes several existing Intergrax strengths into a materially stronger system story:

- persistent autonomous work,
- governance,
- evidence,
- collaboration,
- safe adaptive capability,
- human oversight without human micromanagement.

---

## 26. Next gate

**Do not begin implementation from this document directly.**

Required next action:

> Perform an independent architecture audit of this document against the current `development` branch, challenge every `EXISTING / REUSE / EXTEND / NEW` classification, identify duplication risks and missed platform primitives, and either accept or correct the proposed target architecture.

Only after that review should Intergrax decide whether to create a canonical Autonomous Work domain/ADR/plan and decompose the accepted architecture into implementation tasks.
