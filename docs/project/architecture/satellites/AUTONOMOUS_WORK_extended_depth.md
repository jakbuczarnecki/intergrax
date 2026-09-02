# AUTONOMOUS_WORK — extended architecture depth

**Parent hub:** [`AUTONOMOUS_WORK.md`](../AUTONOMOUS_WORK.md)

This satellite holds **extended technical architecture** for Autonomous Work. It does not redefine domain ownership established in the hub. Conceptual contracts here are target architecture; exact Python schemas are frozen in AW-1.

**Do not read this entire file in one session.** Load only the section named by the active `AW-*` row or Cursor read-scope rule in the parent hub.

---

## Cursor read scope (token budget)

| Work type | Read |
| --- | --- |
| **Default** | Parent hub only |
| **Recovery / adaptive capability** | Hub §Adaptive obstacle recovery + this file §Recovery Controller, §Capability acquisition, §A0–A4, §CodeCraft recovery + [`CODE_CRAFT.md`](../CODE_CRAFT.md) relevant slice |
| **Lifecycle** | This file §Worker lifecycle only |
| **Observability** | This file §Observability + [`OBSERVABILITY.md`](../OBSERVABILITY.md) relevant slice |
| **Domain contracts (AW-1)** | This file §Detailed domain model only |
| **Control plane (AW-9)** | This file §Control plane only |
| **Enterprise scenarios / proof** | This file §Enterprise examples only |
| **Long-horizon continuity** | This file §Long-Horizon Work Continuity only + [MEMORY.md](../MEMORY.md) front ownership + [CONTEXT_ENGINEERING.md](../CONTEXT_ENGINEERING.md) front ownership + [UNIFIED_CONTEXT_LIFECYCLE.md](../UNIFIED_CONTEXT_LIFECYCLE.md) front ownership |

---

## Detailed domain model

Autonomous Work introduces durable worker semantics above the collaborative work plane and governed execution plane. The following entities are **conceptual contracts** — not final Python schemas.

### Entity relationships

```text
WorkerDefinition
      ↓ instantiate (versioned)
WorkerInstance
      ↓ binds
WorkerPrincipalBinding
      ↓ scoped by
WorkerWorkBinding (workspace / work-plane scope)
      ↓ owns
Responsibility (0..N)
      ↓ may define
WorkerGoal (0..N per responsibility)
      ↓ references
WorkerBudgetProfile
WorkerCapabilityProfile
WorkerRecoveryContext (per obstacle episode)
WorkContinuityState (durable orientation snapshot)
      ↓ creates / correlates
WorkItem / Assignment (Collaborative Work)
      ↓ dispatches
Task / Run / Attempt / Execution (Unified Execution Runtime)
```

### WorkerDefinition

Reusable definition of a worker role. Immutable at the logical level once published; changes produce a new definition revision.

```text
WorkerDefinition
  worker_definition_id          # stable platform ID
  display_name
  role                          # descriptive business role label (not authority)
  version / revision            # optimistic concurrency + audit semantics
  responsibility_template_refs
  default_goal_policy_ref
  principal_binding_policy_ref
  workspace_scope_ref
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

A definition **references** domain-owned profiles. It does not embed duplicate policy, memory, sandbox, authority or execution configuration.

### WorkerInstance

Durable instantiated worker. Survives individual task completion, run failure, host restart and idle periods.

```text
WorkerInstance
  worker_instance_id
  worker_definition_id + definition_revision
  lifecycle_state                 # see §Worker lifecycle
  principal_binding_ref
  workspace_context_ref
  active_responsibility_refs
  active_goal_refs
  budget_profile_ref              # may override definition default
  capability_profile_ref
  recovery_context_ref            # current or last obstacle episode
  created_at / updated_at / revision
```

`WorkerInstance` identity is **not** tied to host process PID, agent session or execution ID.

### Responsibility

Persistent area of business ownership. May exist with zero running tasks and may spawn many goals and work items over time.

```text
Responsibility
  responsibility_id
  worker_instance_id
  objective                       # durable business statement
  scope_ref                       # workspace / business scope
  policy_constraints_ref
  status                          # active / suspended / completed
  assigned_at / revision
```

Examples:

- Process incoming customer orders according to company policy.
- Maintain supplier integration continuity inside SLA.
- Investigate qualifying incidents until closure.

### WorkerGoal

Durable measurable outcome — **not** an LLM prompt.

```text
WorkerGoal
  goal_id
  responsibility_id
  objective
  success_criteria
  metric_refs
  SLA/SLO refs
  deadline_or_cadence
  priority
  status
  progress_projection_ref
  evaluation_cadence_ref          # proactive check schedule
```

A goal may generate multiple WorkItems and Executions. Goal evaluation is bounded — see §Reactive and proactive work.

### WorkerPrincipalBinding

Explicit binding between a worker instance and a canonical Collaborative Principal.

```text
WorkerPrincipalBinding
  binding_id
  worker_instance_id
  principal_id
  workspace_membership_ref
  effective_authority_snapshot_ref  # point-in-time for correlation
  status                            # active / revoked / stale
  bound_at / revision
```

Responsibilities and goals **never grant authority**. A display role such as `Order Operations Worker` is descriptive only.

### WorkerWorkBinding

Scopes which work-plane objects a worker may accept, create or prioritize.

```text
WorkerWorkBinding
  binding_id
  worker_instance_id
  workspace_id
  work_queue_refs
  subscription_refs               # event / webhook / queue
  assignment_policy_ref
```

### Profile references

| Profile | Purpose | Owning domain |
| --- | --- | --- |
| `governance_profile_ref` | Policy/HITL posture for worker-triggered work | Governed Execution |
| `budget_profile_ref` | Worker-level accounting windows | Autonomous Work (composes execution budgets) |
| `memory_profile_ref` | Permitted continuity scope | Memory / Context Engineering |
| `capability_profile_ref` | Known tools/skills/integrations inventory | Tools / Skills / Integrations |
| `codecraft_profile_ref` | Generated-code recovery posture | Code Craft |
| `risk_profile_ref` | Isolation tier, egress, adversarial handling | Code Craft / Sandbox |
| `schedule_profile_ref` | Proactive evaluation cadence | Autonomous Work |
| `escalation_policy_ref` | Human escalation paths | Reliability / HITL |
| `observability_profile_ref` | Correlation and projection settings | Observability |

### Version / revision semantics

- `WorkerDefinition` changes produce a new **definition revision**; existing instances may remain on prior revision until migrated.
- `WorkerInstance`, `Responsibility`, `WorkerGoal` use **optimistic revision** for concurrent updates.
- Lifecycle transitions are **append-only facts** correlated to evidence — not silent overwrites.
- Host restart rehydrates persisted state; it does not create a new worker identity.

---

## Worker lifecycle

Canonical semantic states. Concrete enum shape is frozen in AW-1; transition service in AW-2.

### State inventory

```text
PROVISIONING      # instance being created / binding validated
ACTIVE            # operationally available (not necessarily reasoning)
IDLE              # healthy, no current work in progress
WORKING           # actively processing work or dispatching execution
WAITING_EXTERNAL  # blocked on external dependency
WAITING_FOR_HUMAN # blocked on human approval / decision
RECOVERING        # obstacle recovery in progress
DEGRADED          # operating under reduced capability or elevated risk
PAUSED            # operator-governed suspension of new work
QUARANTINED       # security/policy containment
STOPPED           # terminal; no further autonomous work
```

### Significant transitions

| From | Trigger | To | Meaning |
| --- | --- | --- | --- |
| — | instantiate | `PROVISIONING` | Worker record created; bindings validated |
| `PROVISIONING` | bindings valid | `ACTIVE` | Worker ready for wake-up |
| `PROVISIONING` | binding/validation fail | `STOPPED` | Cannot operate; fail closed |
| `ACTIVE` | no current work | `IDLE` | Normal healthy rest state |
| `IDLE` / `ACTIVE` | wake-up + work accepted | `WORKING` | Processing or dispatching |
| `WORKING` | execution dispatched, awaiting result | `WORKING` or `IDLE` | Worker may return IDLE while execution runs |
| `WORKING` / `IDLE` | external dependency timeout | `WAITING_EXTERNAL` | Scheduled retry or dependency watch |
| `*` | HITL / approval required | `WAITING_FOR_HUMAN` | Cognition paused; human path active |
| `*` | obstacle classified recoverable | `RECOVERING` | Recovery strategy executing |
| `RECOVERING` | recovery success | `WORKING` or `IDLE` | Resume original work context |
| `RECOVERING` | recovery exhausted / partial | `DEGRADED` or escalate | Bounded failure; may need human |
| `*` | operator pause | `PAUSED` | No new work; in-flight containment per policy |
| `*` | security/policy containment | `QUARANTINED` | All privileged work stopped |
| `PAUSED` | operator resume | `ACTIVE` / `IDLE` | Governed reactivation |
| `QUARANTINED` | operator release after review | `PAUSED` or `ACTIVE` | Governed release only |
| `*` | operator stop / terminal policy | `STOPPED` | Permanent cessation |

Not every combination is valid. Invalid transitions fail deterministically (AW-2).

### Semantic distinctions

**Worker pause vs execution pause**

- An execution may be paused (HITL, rate limit) while the worker remains `WORKING` or `IDLE`.
- `PAUSED` at worker level suspends **new** work intake and proactive evaluation; in-flight executions follow containment policy.

**Quarantine vs pause**

- `PAUSED` is an operational suspension — often reversible by authorized operator.
- `QUARANTINED` is a **security/policy containment** state — suspicious input, policy violation suspicion, or adversarial recovery attempt. Requires governed release.

**Host restart**

- Persisted `WorkerInstance` state rehydrates with same `worker_instance_id`.
- Lifecycle resumes from last persisted state; no new identity.
- In-flight executions follow execution-plane recovery semantics independently.

**External dependency waiting**

- Worker enters `WAITING_EXTERNAL` with scheduled wake-up or subscription.
- No LLM activity during wait unless policy permits bounded re-evaluation.

**Human approval waiting**

- Worker enters `WAITING_FOR_HUMAN` with evidence-linked escalation.
- Original work context preserved for continuation.

**Recovery**

- `RECOVERING` is explicit — obstacle classified, strategy selected, bounded attempts tracked.
- Recovery success returns to original WorkItem/goal context with evidence chain.

**Stopped terminal state**

- `STOPPED` is terminal for autonomous operation.
- Reactivation requires explicit operator action (new instance or governed restart), not silent auto-resume.

### Lifecycle vs execution lifecycle

```text
Worker lifecycle:     PROVISIONING → ACTIVE ↔ IDLE ↔ WORKING → … → STOPPED
Execution lifecycle:  Task → Run → Attempt → Execution → (complete/fail)
```

Worker lifecycle sits **above** Task/Run/Execution lifecycle. A worker may be `IDLE` while executions complete asynchronously.

---

## Wake-up and scheduling semantics

A Virtual Worker is persistent, but cognition is **event-driven and bounded**.

> **Persistent responsibility does not mean persistent compute.**

Forbidden default design:

```text
while true:
    think_with_llm()
    act()
```

### Wake-up sources

| Source | Description | Typical transition |
| --- | --- | --- |
| **External event** | Webhook, file arrival, email, domain signal | `IDLE` → `WORKING` |
| **Queue message** | Work queue delivery | `IDLE` → `WORKING` |
| **New WorkItem** | Assignment or created work in collaborative plane | `IDLE` → `WORKING` |
| **Schedule / timer** | Cron-like or interval trigger | `IDLE` → `WORKING` |
| **Goal evaluation** | Proactive SLA/KPI check cadence | `IDLE` → `WORKING` |
| **SLA checkpoint** | Approaching deadline on tracked goal | `IDLE` → `WORKING` |
| **External dependency recovery** | Previously unavailable dependency now available | `WAITING_EXTERNAL` → `WORKING` |
| **HITL continuation** | Human decision received | `WAITING_FOR_HUMAN` → `WORKING` |
| **Operator action** | Manual wake, priority override | `PAUSED` → `ACTIVE` (governed) |
| **Recovery timer** | Scheduled retry after backoff | `RECOVERING` / `WAITING_EXTERNAL` → `WORKING` |

### Canonical wake-up flow

```text
WorkerInstance ACTIVE/IDLE
        ↓ wake-up trigger (one of above)
work acceptance / goal evaluation
        ↓
create or select WorkItem
        ↓
canonical Governed Execution
        ↓
result / wait / recovery
        ↓
persist worker state
        ↓
IDLE or next bounded work
```

Wake-up handling must be **idempotent** — duplicate events must not create duplicate side effects (AW-4).

---

## Reactive and proactive work

Autonomous Work supports two work-origin classes. Both produce WorkItems and Executions; they differ in **what triggers** the worker.

### Reactive work

External work arrives; worker accepts and dispatches.

```text
incoming work (email / webhook / queue / assignment)
        ↓
worker accepts work
        ↓
WorkItem
        ↓
Governed Execution
        ↓
result / obstacle
        ↓
worker remains responsible
```

The worker does not need to "discover" work — the work plane delivers it.

### Proactive work

Worker evaluates responsibility or goal and detects required action **without** an external task prompt.

```text
goal / responsibility
        ↓
scheduled evaluation (cadence-bound)
        ↓
worker detects required action
        ↓
creates or prioritizes WorkItem
        ↓
Governed Execution
        ↓
result / obstacle
        ↓
worker remains responsible
```

### Example — SLA 30 minutes

```text
Goal: 99% of orders completed within 30 minutes
        ↓
scheduled goal evaluation (e.g. every 5 minutes)
        ↓
three orders at 27 minutes without completion
        ↓
worker creates/prioritizes escalation WorkItems
        ↓
governed Executions per order
        ↓
evidence + SLA projection update
```

Proactive evaluation is bounded by:

- configured cadence (not continuous polling),
- worker budget (cost, concurrency, evaluation frequency cap),
- effective authority (cannot authorize actions it lacks),
- policy (cannot bypass DENY).

This is the clearest illustration of **Worker vs Agent**:

- An **Agent** executes reasoning inside a single governed run.
- A **Virtual Worker** owns the goal across time, schedules evaluation, creates work, and remains responsible after each execution ends.

---

## Recovery Controller

Autonomous Work owns **worker-level decision semantics** for responding to obstacles. It does not own every underlying recovery mechanism (retry, HITL, CodeCraft, Sandbox).

### Canonical flow

```text
execution fails / progress stalls
        ↓
canonical evidence + diagnostics (DIAG)
        ↓
Obstacle Classification
        ↓
strategy selection
        ↓
bounded recovery attempt OR escalation
        ↓
resume original WorkItem/goal OR terminal failure
```

> **Recovery changes strategy or capability. It does not weaken policy.**

### Obstacle taxonomy

| Obstacle | Default strategy | Autonomous? | Human needed? | Forbidden behavior |
| --- | --- | --- | --- | --- |
| **Transient fault** | reliability retry / backoff | Yes (bounded) | Rarely | Infinite retry |
| **Dependency unavailable** | wait / reschedule / `WAITING_EXTERNAL` | Yes (bounded) | If prolonged | Fabricate dependency |
| **Rate limit** | throttle / wait / reschedule | Yes | Rarely | Bypass rate limiter |
| **Credential revoked/missing** | escalate; fail closed | No | **Yes** | Synthesize credential |
| **Policy DENY** | stop; record evidence | No | Per policy | Treat as recoverable gap |
| **Business ambiguity** | canonical HITL path | No | **Yes** | Guess business decision |
| **Known alternate capability** | replan with approved capability | Yes (if in profile) | Rarely | Use unapproved capability |
| **API drift** | inspect docs → adaptive recovery | Conditional | Often for A3 | Silent production mutation |
| **Schema drift** | inspect schema → adapter candidate | Conditional | Often for A3 | Skip validation |
| **Missing capability** | capability acquisition ladder | Conditional | Depends on tier | Jump to CodeCraft first |
| **Suspicious/malicious input** | quarantine / security path | No | Often | Execute untrusted content |
| **Goal/SLA drift** | proactive re-prioritization | Yes (bounded) | If escalation policy | Ignore SLA |

### Classification precedes synthesis

`on_error → CodeCraft` is **forbidden architecture**. Every obstacle must pass classification before any capability acquisition or synthesis.

### Missing capability vs missing authority

| Gap type | Worker may attempt | Worker may not |
| --- | --- | --- |
| **Missing capability** | Search ladder → configure → CodeCraft → ephemeral use | Auto-promote to production |
| **Missing authority** | Escalate to human / operator | Self-expand credentials, scope, membership, policy |

---

## Capability acquisition

Missing capability recovery is broader than CodeCraft. The worker searches existing inventory before any synthesis.

### Full acquisition ladder

```text
1. Existing Tool?
      ↓ no
2. Existing Skill?
      ↓ no
3. Existing Integration?
      ↓ no
4. Alternate approved workflow/capability?
      ↓ no
5. Approved docs/schema inspection?
      ↓
6. Configure/instantiate existing approved capability?
      ↓ no
7. CodeCraft ephemeral synthesis (if profile permits)?
      ↓ success
8. Governed durable promotion (separate gate)?
      ↓ unavailable
9. Escalation — capability or authority unavailable
```

CodeCraft is the **canonical generated-code path** — not the default first response. See [`CODE_CRAFT.md`](../CODE_CRAFT.md).

### Missing capability

The worker lacks a tool, parser, adapter or workflow to proceed. Recovery may:

- find an existing approved capability,
- configure an existing capability within authority,
- synthesize ephemeral code under A1/A2 constraints.

### Missing authority

The worker lacks permission for the required action. Recovery may **only**:

- escalate to human operator,
- request governed authority change through Collaborative Work (A4 — never self-authorized).

The worker **must not** solve missing authority by synthesizing credentials, broadening database scope, disabling policy or expanding egress.

---

## A0–A4 capability autonomy tiers

Risk-aware capability growth. Each tier defines autonomous posture, isolation, governance and persistence rules.

### Summary matrix

| Tier | Example | Autonomous? | Isolation | Governance | HITL | Durable? |
| --- | --- | --- | --- | --- | --- | --- |
| **A0 — Known Capability** | existing approved tool | Yes | Per tool profile | Standard execution | Per policy | Already durable |
| **A1 — Ephemeral Safe** | parser / transform / local helper | Conditional | Hardened sandbox required | Static gate + policy | If profile requires | **No** — ephemeral only |
| **A2 — Scoped Adaptive** | temporary API adapter, restricted egress + scoped secret | Conditional | Hardened + egress enforced | Narrow scope policy | Often required | Ephemeral; promotion separate |
| **A3 — Production Change** | durable connector / workflow update | No (governed) | Strong isolation for test | Promotion pipeline | Commonly required | **Yes** — via governed promotion |
| **A4 — Authority Change** | new credential, broader DB scope, disable policy | **Never** | N/A | Control-plane mutation | **Always** | Per authority domain |

### A0 — Known Capability

- **Example:** invoke `orders.parse_pdf` tool already in capability profile.
- **Autonomous:** yes, within normal governed execution.
- **Isolation:** per tool/skill profile.
- **Governance:** standard policy evaluation.
- **HITL:** per existing policy rules.
- **Persistence:** capability already exists in platform inventory.

### A1 — Ephemeral Safe

- **Example:** generate a one-off parser for unknown attachment format.
- **Autonomous:** yes only if CodeCraft profile permits and hardened sandbox available.
- **Isolation:** hardened sandbox mandatory; anti-downgrade enforced.
- **Governance:** static gate + policy; no production side effects.
- **HITL:** per risk profile; injection/malicious input → quarantine.
- **Persistence:** **ephemeral only** — discarded after use on original work item.

### A2 — Scoped Adaptive

- **Example:** temporary HTTP adapter for drifted vendor API with restricted egress and scoped secret.
- **Autonomous:** conditional — narrow scope, enforceable egress, purpose-scoped secrets.
- **Isolation:** hardened sandbox + network policy.
- **Governance:** scoped secret brokering; contract tests required.
- **HITL:** often required before external calls.
- **Persistence:** ephemeral by default; may feed A3 promotion evidence.

### A3 — Production Change

- **Example:** durable vendor connector after confirmed API drift.
- **Autonomous:** **no** — governed promotion pipeline.
- **Isolation:** strong isolation for test/shadow/canary phases.
- **Governance:** proof package, contract/regression tests, promotion decision.
- **HITL:** commonly required.
- **Persistence:** **yes** — versioned durable publication with rollback.

### A4 — Authority Change

- **Example:** new CRM write permission, production database scope expansion.
- **Autonomous:** **never self-authorized**.
- **Isolation:** N/A — this is an authority domain concern.
- **Governance:** Collaborative Work / Governed Execution control-plane mutation.
- **HITL:** **always** required.
- **Persistence:** per authority domain; worker cannot trigger silently.

> **A4 = never self-authorized**

Normative direction:

> **SELF-EXTENDING CAPABILITY: conditionally allowed.**  
> **SELF-EXTENDING AUTHORITY: forbidden.**

---

## CodeCraft and Sandbox recovery

When capability acquisition reaches step 7, generated code follows the **canonical CodeCraft path**. Autonomous Work does not create alternate synthesis routes.

### Recovery flow

```text
Obstacle
  → capability gap (classified)
  → CodeCraft allowed? (profile + A-tier check)
  → generate
  → static gate
  → policy / HITL when required
  → hardened Sandbox
  → execute / test
  → CVL / Critic verification
  → bounded result
  → ephemeral use on original work
  → resume original WorkItem / goal
```

### Explicit prohibitions

- **No** raw local Python bypass outside approved sandbox.
- **No** private agent subprocess execution path.
- **No** alternate `WorkerCodeCraft` subsystem.
- **No** automatic global promotion to ToolRegistry.

### Production blockers (current)

These remain release blockers for production-style autonomous workers:

| Blocker | Requirement |
| --- | --- |
| Isolation anti-downgrade | Required hardened tier cannot fall back to local execution |
| Egress enforcement | Network restrictions provable and evidenced |
| Scoped secrets | Purpose-scoped secret brokering; no trace leakage |
| Hostile-code qualification | Adversarial/sandbox-escape test corpus |
| Proven execution substrate | Container/cloud isolation qualified for generated code |

If worker risk profile requires hardened isolation and substrate cannot be proven → **fail closed**.

Full CodeCraft canon: [`CODE_CRAFT.md`](../CODE_CRAFT.md) · [`satellites/CODE_CRAFT_extended_depth.md`](CODE_CRAFT_extended_depth.md)

---

## Durable capability promotion

Ephemeral recovery success does **not** imply production promotion.

> **Ephemeral success != production promotion.**

### Promotion lifecycle

```text
CraftResult
  → proof package assembly
  → static / security validation
  → isolated tests
  → contract / regression tests
  → shadow execution (where applicable)
  → canary (where applicable)
  → promotion decision
  → optional HITL
  → versioned durable release
  → rollback path retained
```

### Proof package contents

| Artifact | Purpose |
| --- | --- |
| Code / artifact | Generated capability source |
| Tests | Unit, integration, contract |
| Static / security result | L0 gate, security scan evidence |
| Sandbox substrate evidence | Isolation tier, egress proof |
| Input / output contract | Schema, typed boundaries |
| Dependencies | Declared and approved |
| Network requirements | Hosts, ports, protocols |
| Secret requirements | Scoped secret declarations |
| Provenance | `craft_id`, worker context, obstacle classification |

A3 promotion is a **control-plane mutation** — explicit authorization and evidence required. No silent global ToolRegistry persistence.

---

## Budgets

Worker lifetime exceeds any single Execution. Worker-level accounting windows aggregate and constrain existing execution budgets.

> Worker budget **aggregates / constrains** existing execution budgets.  
> Execution ledger remains source of truth for individual run expenditure.

### Worker-level time-window semantics

| Budget dimension | Example window | Purpose |
| --- | --- | --- |
| **Cost / day** | $50/day per worker | Cap total child execution spend |
| **Cost / month** | $1000/month per worker | Fleet cost governance |
| **Model tokens / day** | 2M tokens/day | Cognition cost cap |
| **CodeCraft attempts / day** | 10 attempts | Limit synthesis abuse |
| **Recovery attempts / WorkItem** | 3 attempts | Bound recovery loops |
| **External side effects / hour** | 100 API calls/hour | Rate-limit production impact |
| **Concurrent WorkItems** | 5 simultaneous | Concurrency cap |
| **Human approval burden** | 20 HITL requests/day | Escalation budget |
| **Proactive goal-check cadence** | min 5 min interval | Prevent evaluation storms |

### Composition model

```text
WorkerBudgetProfile (policy + windows)
      ↓ constrains
0..N Execution budgets (per WorkItem / Execution)
      ↓ enforced by
existing durable execution budget mechanisms
```

### Rules

- No second execution-budget engine or competing ledger.
- Worker budget survives host restart.
- Budget exhaustion → governed stop or escalation, not silent override.
- Proactive evaluation consumes budget like any other wake-up.

---

## Observability

Two families of facts must not be conflated.

### Worker / domain / control-plane facts

Owned or contributed by Autonomous Work (durable state + correlation):

| Fact (illustrative) | When |
| --- | --- |
| `WorkerActivated` | Instance becomes ACTIVE |
| `WorkerPaused` | Operator pause |
| `WorkerQuarantined` | Security/policy containment |
| `ResponsibilityAssigned` | New responsibility bound |
| `GoalChanged` | Goal created/updated/completed |
| `RecoveryStarted` | Obstacle classified; strategy selected |
| `RecoveryCompleted` | Recovery success or terminal failure |
| `WorkItemCorrelated` | Worker links to work-plane object |
| `BudgetThresholdReached` | Worker budget window exceeded |

Exact event names and schemas are frozen in AW-8.

### Execution evidence

Owned by HOS / RuntimeEvent (Observability domain):

- Task/Run/Attempt lifecycle events,
- tool invocations, policy decisions,
- CodeCraft session events,
- DIAG classifications.

Autonomous Work **correlates** worker context into execution evidence; it does not create a second execution event source.

### Correlation chain

```text
Worker
  → Responsibility
  → Goal
  → WorkItem
  → Task
  → Run
  → Attempt
  → Execution
  → RuntimeEvent
```

### Dashboard semantics

A worker fleet dashboard is a **projection** from persisted worker state + correlated execution evidence — not a second history system.

Minimum operator questions the projection must answer:

- active / idle / working / recovering / waiting / degraded / paused / quarantined?
- assigned responsibilities and goals?
- current/open work?
- autonomous completion rate?
- human intervention rate?
- recovery attempts and success?
- policy denials?
- generated capabilities (ephemeral vs promoted)?
- budget/cost?
- SLA/KPI progress?
- full evidence drill-down?

Canon cross-ref: [`OBSERVABILITY.md`](../OBSERVABILITY.md)

---

## Control plane

Enterprise Autonomous Work requires explicit fleet operations. All sensitive mutations are **governed and evidenced**.

### Operations

| Operation | Description | Governance |
| --- | --- | --- |
| **Register / instantiate** | Create WorkerDefinition catalog entry; spawn WorkerInstance | Authenticated; tenant-scoped |
| **Activate** | Move from PROVISIONING to ACTIVE | Policy check on bindings |
| **Pause** | Suspend new work intake | Operator authorization |
| **Resume** | Restore from PAUSED | Operator authorization |
| **Stop** | Terminal cessation | Operator authorization |
| **Quarantine** | Security/policy containment | Security operator |
| **Release** | Exit quarantine after review | Security operator + evidence |
| **Assign responsibility** | Bind responsibility to worker | Governed mutation |
| **Assign work** | Manual work assignment / priority override | Governed mutation |
| **Change budget profile** | Update worker budget windows | Governed mutation |
| **Change policy profile** | Update governance posture | Governed mutation; may require HITL |
| **Change capability profile** | Update known capability inventory | Governed mutation |
| **Revoke authority binding** | Disconnect Principal binding | Fail-closed; stop privileged work |
| **Terminate active work** | Contain in-flight executions | Governed; evidence required |

### Sensitive mutation rule

Any control-plane mutation that changes worker configuration, authority binding, capability promotion or security posture:

1. requires explicit authentication and authorization,
2. produces evidence (audit trail),
3. fails closed on ambiguity.

Promotion (A3) and authority change (A4) are control-plane mutations — never autonomous.

---

## Enterprise examples

Three reference scenarios illustrating Worker semantics, recovery and proactive work. See hub §Reference enterprise scenario for the flagship worker overview.

### Example 1 — unknown attachment

```text
order arrives (email event)
  → worker accepts work (reactive)
  → WorkItem: process order
  → governed Execution
  → attachment format unrecognized
  → DIAG: missing capability
  → Recovery Controller: capability gap
  → acquisition ladder: no existing parser
  → CodeCraft A1 (ephemeral safe)
  → static gate
  → hardened Sandbox
  → validate against Order schema
  → CVL verification
  → ephemeral parser used on attachment
  → resume original order WorkItem
  → worker returns IDLE
```

**Prompt injection / malicious file boundary:** if attachment content is suspicious (injection patterns, executable payload), obstacle class shifts to **suspicious/malicious input** → quarantine path, not blind CodeCraft execution.

### Example 2 — vendor API drift

```text
order sync Execution
  → HTTP 410 / schema mismatch
  → DIAG: API drift
  → Recovery Controller
  → approved docs/schema inspection (ladder step 5)
  → CodeCraft adapter candidate (A2)
  → restricted network + scoped secret
  → contract tests in sandbox
  → ephemeral adapter resolves immediate work
  → shadow execution on sample traffic
  → canary on subset
  → governed A3 promotion decision
  → optional HITL
  → versioned durable connector published
  → resume synchronization WorkItem
```

Ephemeral adapter success does not auto-promote. Promotion follows §Durable capability promotion.

### Example 3 — proactive SLA

```text
Goal: 99% of orders completed within 30 minutes
  → Responsibility: order operations continuity
  → scheduled goal evaluation (every 5 min)
  → projection: 3 orders at 27 min, SLA at risk
  → worker creates/prioritizes escalation WorkItems (proactive)
  → governed Executions per order
  → evidence + SLA metrics updated
  → worker remains responsible for goal
```

This scenario shows **Worker vs Agent** clearly:

- No external task told the worker to check SLA.
- The worker owns the goal, evaluates on cadence, creates work, and persists responsibility across executions.

---

## Invariants — extended explanation

Identifiers match hub §Normative invariants. This section elaborates meaning; it does not change identifiers.

| ID | Extended meaning |
| --- | --- |
| **AW-INV-01** | Worker identity, lifecycle and responsibility are domain entities distinct from any AgentDefinition or AgentRun. An agent may be *used by* a worker execution; it is not *the* worker. |
| **AW-INV-02** | Worker state persists across Task/Run/Attempt/Execution boundaries. Completing or failing an execution does not terminate the worker. |
| **AW-INV-03** | Authority flows from Collaborative Principal binding. Worker role labels are descriptive; `Responsibility` and `WorkerGoal` cannot grant permissions. |
| **AW-INV-04** | A responsibility may exist with zero active executions. Host restart must preserve responsibility and goal state. |
| **AW-INV-05** | Business work (`WorkItem`) and execution dispatch (`Task`) remain separate planes. Worker bridges them; it does not collapse them. |
| **AW-INV-06** | `IDLE` is healthy. Wake-up is event/schedule/goal driven. No default infinite reasoning loop. |
| **AW-INV-07** | Recovery may synthesize parsers, adapters or workflows. It may not acquire credentials, expand DB scope or disable policy. |
| **AW-INV-08** | All generated executable code routes through CodeCraft + approved sandbox. No agent-private subprocess loop. |
| **AW-INV-09** | If hardened isolation is required and unavailable, execution fails closed — no silent downgrade to local sandbox. |
| **AW-INV-10** | Policy evaluation, HITL and authority decisions use canonical Governed Execution mechanisms. |
| **AW-INV-11** | Policy DENY is terminal for recovery — not a trigger for capability synthesis or authority expansion. |
| **AW-INV-12** | Execution facts live in HOS/RuntimeEvent. Worker dashboard projects from persisted state + correlation. |
| **AW-INV-13** | `worker_instance_id` survives host restart. Process death does not destroy worker identity. |
| **AW-INV-14** | Acquisition ladder steps 1–6 precede CodeCraft. Existing inventory is always searched first. |
| **AW-INV-15** | A1/A2 ephemeral success does not auto-publish. A3 promotion is a separate governed pipeline. |
| **AW-INV-16** | Proactive goal evaluation respects cadence, budget, authority and policy bounds. |
| **AW-INV-17** | Pause, quarantine, profile change, promotion and authority revocation require governance and evidence. |
| **AW-INV-18** | Tier-3 Virtual Workforce application consumes worker contracts; it does not own worker semantics. |
| **AW-INV-19** | Worker budget windows constrain child execution budgets. No parallel execution-cost ledger. |
| **AW-INV-20** | Obstacle classification is mandatory before any recovery strategy including CodeCraft. |

### Recommendations (not adopted)

AW-INV-21 through AW-INV-26 added for Long-Horizon Work Continuity at AW-0 (pending independent audit). Potential future candidates for independent audit consideration:

- explicit **idempotent wake-up** invariant (may be subsumed by AW-4 acceptance criteria),
- **recovery budget isolation** invariant (partially covered by AW-INV-19 and recovery attempts cap).

---


---

# Long-Horizon Work Continuity

Fundamental Virtual Worker property — not an optional Memory feature.

> A Virtual Worker must preserve effective work continuity across an effectively unbounded work horizon while keeping each active model context bounded, relevant, attributable and reconstructable.

> **The information space may be effectively unbounded. The active context must remain bounded.**

## Problem statement

Workers may operate for days, weeks or months across thousands of WorkItems, executions, millions of records, and many external systems. They may restart, change models, idle, or interleave unrelated work. They must answer orientation questions without reloading full history:

`	ext
Where am I? What am I responsible for? What am I trying to achieve?
What is currently open? What has already been done? What failed?
What did I learn? What information do I need now? What should happen next?
`

Autonomous Work coordinates Memory, Context Engineering, UCL, RAG and Token Optimization — without WorkerMemoryEngine, WorkerContextEngine, worker-private RAG, or alternate UCL.

## Five-level information model

| Level | Name | Owner | Purpose |
| --- | --- | --- | --- |
| 1 | Active Working Context | Context Engineering | Small model-facing context for the current call |
| 2 | Work Continuity State | Autonomous Work | Durable orientation — open work, blockers, learned refs, next action |
| 3 | Durable Memory | Memory | Facts, experiences, profiles, important events |
| 4 | Knowledge / External Information Space | RAG, Tools, Integrations, files, APIs, repos | Retrieved on demand |
| 5 | Durable Context Lifecycle / Optimization | UCL, CE, Token Optimization | Reuse, compaction, budgeting, degradation |

## Work Continuity State

See [§WorkContinuityState](#workcontinuitystate) conceptual contract. Orientation state — not a competing memory store.

## Restore-orientation flow

`	ext
Worker wakes
    ↓
load durable Worker state
    ↓
load Responsibilities + active Goals
    ↓
load open / blocked WorkItems
    ↓
load latest continuity checkpoint
    ↓
identify information needed now
    ↓
selectively recall Memory
    ↓
retrieve external knowledge only when needed
    ↓
Context Engineering assembles bounded context
    ↓
Agent reasons / Execution acts
`

Must work after: host restart, long idle, model replacement, context-window reset, unrelated executions, large work-history growth.

## Selective retrieval

Workers retrieve relevant information when needed instead of accumulating full work history into every active model context (AW-INV-23). Full-history replay is forbidden in normal operation (AW-INV-24).

## Context Engineering integration

CE owns assembly, global input budget, ranking, degradation and provenance. Autonomous Work supplies continuity requirements and anchor references.

## Memory integration

Memory owns stores, lifecycle, consolidation and recall. Workers persist valuable outcomes through canonical Memory write paths — not a worker-private store.

## RAG / external-space traversal

RAG and integrations answer external knowledge needs. Workers retain how to find information again (source refs, scopes, retrieval keys) — not entire corpora in active context.

## UCL / Token Optimization reuse

UCL coordinates durable artifact reuse-before-create and compaction. Token Optimization executes approved transformations. No worker-specific duplicate.

## Context decay / stale information

Continuity state and recalls may become stale. Checkpoints, revision semantics, freshness metadata and re-validation before high-impact actions are required.

## Checkpoint and restart continuity

Every bounded work episode leaves a durable checkpoint: progress, open items, blockers, next-action hint, anchor refs. Restart loads checkpoint + authoritative worker state — never process-local prompt history.

## Cross-WorkItem learning

Lessons may promote to Memory or update continuity anchors. Cross-item recall is profile-governed and must not flood active context.

## Forgetting / retention

Retention applies via Memory and worker profiles. Forgetting unresolved work is a failure mode — open/blocker refs must survive until resolved or escalated.

## Provenance

Restored context must preserve source identity (AW-INV-25): durable fact vs RAG hit vs execution evidence vs inferred summary.

## Context-efficiency observability

**Continuity:** continuation success after restart; continuation success after long idle; duplicate-work rate; lost-open-work rate.

**Context efficiency:** active context tokens per step; context tokens per completed WorkItem; retrieved-data volume; full-history-read count (expected near zero / forbidden in normal operation); artifact reuse rate; context compaction/reuse rate.

**Recall quality:** relevant recall precision; missed-critical-context rate; stale-context usage rate; incorrect-memory usage rate.

**Long-horizon scaling:** logical stress at 100 / 1,000 / 10,000 / 100,000 historical events — active context must remain bounded and continuation correct.

Key question: *With 1000× history growth, does active context stay bounded and can the Worker still continue correctly?*

## Failure modes

| Failure | Architectural safeguard |
| --- | --- |
| Full-history replay | AW-INV-24; orientation from continuity state + selective retrieval |
| Context accumulation | AW-INV-21/23; CE global budget; release transient context |
| Stale memory | Freshness metadata, re-validation, stale-context metrics |
| Irrelevant recall | CE ranking/filtering; scoped recall; precision metrics |
| Duplicate work after restart | Idempotent wake-up; checkpointed open-work refs |
| Forgetting unresolved work | open_work_refs / `blocked_work_refs` in continuity state |
| Historical goal treated as active | `active_goal_refs` vs archived goals |
| Obsolete external data | source refs + re-fetch; RAG scope validation |
| Summary drift | UCL revision semantics; provenance on compacted artifacts |
| Lost provenance | AW-INV-25; CE provenance emit |
| Excessive repeated retrieval | UCL artifact reuse; retrieval volume metrics |
| Token cost grows with worker age | bounded active context invariant |
| Model change breaks continuity | durable state independent of prior model context |

## Proof requirements

AW-11 must include long-horizon stress scenarios and metrics above. Future proof gates — no runtime tests at AW-0.

## Cross-references

| Topic | Authoritative hub |
| --- | --- |
| CodeCraft synthesis | [`CODE_CRAFT.md`](../CODE_CRAFT.md) |
| Collaborative identity / WorkItem | [`COLLABORATIVE_WORK.md`](../COLLABORATIVE_WORK.md) |
| Policy / authority | [`GOVERNED_EXECUTION.md`](../GOVERNED_EXECUTION.md) |
| Execution lifecycle | [`UNIFIED_EXECUTION_RUNTIME.md`](../UNIFIED_EXECUTION_RUNTIME.md) |
| Execution evidence | [`OBSERVABILITY.md`](../OBSERVABILITY.md) |
| Diagnostics input | [`DIAGNOSTICS.md`](../DIAGNOSTICS.md) |
| Host restart | [`APPLICATION_HOSTING.md`](../APPLICATION_HOSTING.md) |
| Implementation plan | [`../../maintainers/plans/AUTONOMOUS_WORK.md`](../../maintainers/plans/AUTONOMOUS_WORK.md) |
