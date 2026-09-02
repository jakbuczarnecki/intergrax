# Review — Autonomous Work / Virtual Workforce Architecture & Gap Audit

**Status:** Independent review of `AUTONOMOUS_WORK_VIRTUAL_WORKFORCE_ARCHITECTURE_GAP_AUDIT.md`  
**Review date:** 2026-09-02  
**Branch:** `development`  
**Reviewed baseline:** `84169e87cade1eb7960577718dce7f054036a950`

## 1. Review verdict

The audited document is **substantively correct and complete enough to serve as the strategic architecture baseline**, but it should not yet be converted directly into a canonical domain specification or implementation plan.

All major requirements established for the initiative are present:

- market/trend validation and the shift from one-shot agent tasks to durable responsibility,
- audit of Intergrax agents, applications, harness/runtime and platform layers,
- explicit `Agent != Worker` separation,
- persistent worker identity/lifecycle/responsibility/goals,
- governance and non-amplifying authority,
- observability, traceability and dashboard/control-plane expectations,
- Collaborative Work / multiplayer reuse,
- long-running execution, pause/resume and durable state reuse,
- dynamic coding through the existing CodeCraft domain rather than a second coding runtime,
- hardened sandbox requirements,
- unexpected-obstacle recovery,
- unknown attachment and vendor API drift examples,
- capability autonomy tiers,
- proof scenario, chaos corpus and business/technical metrics,
- staged strategic roadmap.

The central architecture choice is accepted: **Virtual Workforce should be a platform-level Autonomous Work semantic/operating layer plus a human-facing application/control surface, composed over existing Intergrax domains — not a special super-agent and not an infinite model loop.**

## 2. Corrections / clarifications required before canonicalization

### R1 — Worker must not become a second identity/authority owner

The source document sometimes describes Worker as a "persistent operational principal" and speaks about "Worker effective authority". This is directionally understandable but risks duplicating Collaborative Work.

Correct target rule:

```text
WorkerInstance
    ↓ binds to
Collaborative Work Principal
    ↓
Membership + Authority Grant + Delegation + applicable policies
    ↓
resolved effective authority
    ↓
Execution / Agent / Tool / CodeCraft
```

The Autonomous Work domain may own the **binding** and worker operational semantics, but must not own a competing principal, membership, delegation or authority source of truth.

Add invariant for canonical design:

> **Worker identity is operational identity; authorization identity and effective authority are resolved through canonical Collaborative Work / Governance semantics.**

### R2 — "Always running" must mean durable + event/schedule driven, not continuously invoking an LLM

The source correctly rejects `while True: think(); act()`, but the positive wake-up model needs to be explicit.

A worker should normally be dormant/idle and wake because of one of the following:

- event/subscription (new email, queue item, webhook, file, business event),
- schedule/timer,
- assigned WorkItem,
- resumed external dependency,
- approved HITL continuation,
- goal/SLA evaluation deadline,
- recovery/reschedule timer.

Target model:

```text
Durable WorkerInstance
       ↓
IDLE / WAITING
       ↓ trigger
Wake-up / work intake
       ↓
create governed work/execution
       ↓
complete / wait / recover
       ↓
return to durable worker state
```

This is the correct meaning of a worker that "works all the time": **the responsibility persists all the time; compute/model execution happens only when work or a goal condition requires it.**

### R3 — Add proactive goal evaluation, not only reactive work intake

The initiative requires more than reacting to incoming events. A worker must also be able to inspect whether its durable goals are being met.

The canonical design should therefore distinguish:

- **reactive intake** — respond to an email/event/assignment,
- **scheduled responsibility checks** — e.g. every hour check whether unprocessed orders remain,
- **goal/SLA monitoring** — detect that a target is drifting even if no explicit failure event arrived,
- **proactive work creation** — create a new WorkItem/Execution when a permitted action is necessary to restore goal progress.

This must still be bounded by governance and budgets; a goal must never become implicit unlimited authority.

### R4 — Capability acquisition is broader than "write code"

The source already searches Tool/Skill/Integration first, but the canonical recovery model should explicitly define ordered acquisition options:

```text
1. reuse existing Tool / Skill / Integration
2. re-plan using a different known capability
3. inspect approved documentation/schema
4. configure/instantiate an already approved capability if policy allows
5. synthesize an ephemeral helper through CodeCraft
6. request governed durable capability promotion if necessary
7. escalate when the required capability or authority is unavailable
```

Enabling/installing/changing an integration is a control-plane mutation and must never be treated as ordinary agent reasoning.

### R5 — Durable promotion lifecycle needs its own explicit gate

The source correctly states `ephemeral before durable promotion` and discusses shadow/canary, but canonicalization should define the separation more strongly:

```text
CraftResult / ephemeral helper
        ↓
proof package
- code/artifact
- tests
- static/security evidence
- sandbox substrate evidence
- input/output contract
- dependency manifest
- network/secret requirements
- provenance
        ↓
promotion policy
        ↓
shadow
        ↓
canary
        ↓
HITL if required
        ↓
versioned durable capability/integration release
        ↓
rollback path
```

A worker may discover and prepare the fix autonomously while durable production mutation remains governed independently.

### R6 — Explicitly separate problem recovery from business adaptation

Not every obstacle is a technical failure. The Recovery Controller should distinguish at least:

- infrastructure/transient failure,
- capability gap,
- environment/API/schema drift,
- suspicious/malicious input,
- missing authority/credential,
- business ambiguity/conflict,
- goal/SLA drift,
- policy denial.

Technical recovery may be autonomous. Business ambiguity often requires a Decision/HITL path. Policy denial is terminal for that action and is never a recoverable obstacle.

### R7 — Worker budgets need time-window semantics

The document correctly says not to duplicate execution budgets. Canonical design should add that worker-level budgets are **aggregations/limits over time**, for example:

- cost/day or cost/month,
- model tokens/day,
- CodeCraft attempts/day,
- external side effects/hour,
- concurrent work limit,
- human-approval burden threshold,
- recovery budget per WorkItem.

The execution ledger remains canonical for execution spending; worker views aggregate/enforce longer-lived limits rather than creating a competing accounting truth.

### R8 — Worker observability requires two related truths, not one conflated event stream

Execution facts remain HOS/RuntimeEvent truth. Worker-domain facts such as `WorkerActivated`, responsibility assignment or quarantine are legitimate durable domain/control-plane facts but must not masquerade as executions.

Target separation:

```text
Worker/domain/control-plane facts
             +
canonical execution evidence
             ↓
worker-centric read model/dashboard
```

This preserves one execution truth while allowing the worker lifecycle itself to be auditable.

## 3. Coverage checklist against the initiative

| Requirement | Review result |
|---|---|
| Validate whether the trend is valuable | **COVERED** |
| Audit existing Intergrax agents/apps/harness/platform | **COVERED** |
| Decide layer vs application vs agents | **COVERED — layer + application accepted** |
| Long-lived autonomous responsibility | **COVERED; wake-up/proactive semantics need R2/R3** |
| Governance | **COVERED strongly** |
| Observability / traceability / dashboards | **COVERED; R8 clarification** |
| Multiplayer / human-worker-worker collaboration | **COVERED** |
| Unexpected problem solving | **COVERED strongly** |
| Dynamic code generation/execution | **COVERED through CodeCraft** |
| Sandboxing | **COVERED strongly** |
| Unknown file format example | **COVERED** |
| Changed vendor API example | **COVERED** |
| Self-generated capability without authority expansion | **COVERED strongly** |
| Production promotion/control | **PARTIAL; strengthen via R5** |
| Continuous event intake | **COVERED; operational wake-up model needs R2** |
| Proactive goal pursuit independent of incoming task | **PARTIAL; add R3** |
| Budget/cost controls across worker lifetime | **PARTIAL; add R7** |
| Human control / pause / quarantine / kill | **COVERED** |
| Proof scenario and measurable benchmark | **COVERED strongly** |

## 4. Final reviewed target architecture

```text
ORGANIZATION / HUMAN
       ↓
VIRTUAL WORKFORCE CONTROL APPLICATION
- define workers/responsibilities/goals
- fleet dashboard
- approvals
- pause/quarantine/kill
       ↓
AUTONOMOUS WORK SEMANTIC LAYER
- WorkerDefinition / WorkerInstance
- Responsibility / WorkerGoal
- durable lifecycle
- wake-up / event / schedule / goal evaluation
- work intake and prioritization
- worker→Principal binding
- worker→execution correlation
- Recovery Controller
- capability acquisition policy
       ↓
COLLABORATIVE WORK
Principal / Membership / Delegation / WorkItem / Assignment
       ↓
GOVERNANCE
resolved authority / policy / HITL / control-plane authorization
       ↓
EXECUTION RUNTIME
Task / Run / Attempt / Execution / budgets / pause-resume
       ↓
AGENTS + TOOLS + INTEGRATIONS
       ↓ when capability missing
CODECRAFT
Generate → Gate → Govern → Hardened Sandbox → Test → Verify
       ↓
ephemeral capability
       ↓
resume original work
       ↓ if durable production capability is needed
separate governed promotion → shadow → canary → approval → versioned release/rollback

Worker/domain facts + execution evidence
       ↓
OBSERVABILITY / DIAG / deterministic projections
       ↓
WORKFORCE DASHBOARD
```

## 5. Final verdict

**ACCEPT WITH CLARIFICATIONS.**

The original audit did not miss any of the major concepts agreed for the initiative. Its architecture direction is sound. The eight review items above are mostly boundary hardening rather than a change of direction.

The two most important additions before canonical architecture work are:

1. define the worker as a **durable event/schedule/goal-driven responsibility holder**, not an always-running LLM process;
2. make `WorkerInstance → Principal → resolved authority` explicit so the new domain cannot accidentally become a second identity/governance system.

After these corrections are incorporated into the future canonical architecture/ADR, the initiative is ready to move from Stage 0 audit into Stage 1 semantic design.