# Autonomous Work - Implementation Plan

**Architecture (1:1):** [`../../architecture/AUTONOMOUS_WORK.md`](../../architecture/AUTONOMOUS_WORK.md)  
**ADR:** [`ADR-AW-001`](../../technical/adr/entries/2026-09-02/ADR-AW-001.md)  
**Architecture governance:** [`../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)

**Status:** Domain registered — **AW-0 canonical architecture accepted; runtime implementation NOT STARTED**

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** this read-scope block + one active `AW-*` row only.
- **Architecture:** [`../../architecture/AUTONOMOUS_WORK.md`](../../architecture/AUTONOMOUS_WORK.md) read-scope + sections named in the active row.
- **Cross-domain work:** load only the explicitly named owning domain pair for the current row.

---

## 1. Objective

Deliver a production-governable Autonomous Work domain in which Virtual Workers can own durable business responsibilities and goals across many executions, operate reactively and proactively, recover safely from obstacles, use existing Intergrax capabilities without duplicating their ownership, and remain observable and controllable by humans.

Delivery rule:

> **Semantics first → authority/persistence boundaries → bounded runtime → recovery → control/observability → proof → production qualification.**

---

## 2. Delivery principles

1. **Worker is a new domain entity, not a new Agent subclass.**
2. **Worker != Principal.** Collaborative Work remains identity/authority source.
3. **Worker != WorkItem != Task != Execution.** Preserve work-plane/execution-plane boundaries.
4. **No infinite LLM loop.** Persistent availability is event/schedule/goal driven.
5. **Reuse before create.** Tools/Skills/Integrations precede CodeCraft.
6. **Capability may grow; authority may not self-expand.**
7. **No second policy/HITL/evidence/memory/sandbox/budget engine.**
8. **Recovery classification precedes strategy.** `on_error → CodeCraft` is forbidden.
9. **Ephemeral generated capability precedes durable promotion.**
10. **Production hardening is an explicit gate, not implied by implementation.**

---

## 3. Program roadmap

| Wave | Purpose | Current status |
|---|---|---|
| AW-0 | Canonical architecture, ADR, documentation registration | **IN PROGRESS / documentation-only** |
| AW-1 | Core semantic contracts | NOT STARTED |
| AW-2 | Durable worker state and lifecycle | NOT STARTED |
| AW-3 | Principal / authority / workspace composition | NOT STARTED |
| AW-4 | Work intake and proactive goal evaluation | NOT STARTED |
| AW-5 | Worker → execution composition and budgets | NOT STARTED |
| AW-6 | Recovery Controller and obstacle taxonomy | NOT STARTED |
| AW-7 | Adaptive capability acquisition | NOT STARTED |
| AW-8 | Worker observability and evidence correlation | NOT STARTED |
| AW-9 | Worker control plane | NOT STARTED |
| AW-10 | Virtual Workforce reference application | NOT STARTED |
| AW-11 | Flagship proof / arena | NOT STARTED |
| AW-12 | Production security / reliability qualification | NOT STARTED |

---

## AW-0 - Architecture and governance

| ID | Task | Status |
|---|---|---|
| AW-0A | Strategic market + Intergrax capability audit | **DONE** |
| AW-0B | Independent architecture/gap review | **DONE** |
| AW-0C | Freeze DOMAIN vs FEATURE classification (`AUTONOMOUS_WORK` = DOMAIN) | **DONE** |
| AW-0D | Create canonical architecture/plan pair | **DONE** |
| AW-0E | ADR-AW-001 ownership/classification decision | **PLANNED IN THIS DOC SET** |
| AW-0F | Register domain in architecture/documentation hubs and public routes | **PLANNED IN THIS DOC SET** |
| AW-0G | Independent canonical-doc audit before runtime implementation | **OPEN GATE** |

**Exit gate:** canonical docs accepted with no unresolved ownership conflict; no runtime implementation before AW-0G acceptance.

---

## AW-1 - Core semantic contracts

| Field | Value |
|---|---|
| **ID** | AW-1A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Freeze minimum Tier-0 Autonomous Work contracts |
| **Dependencies** | AW-0G accepted |
| **Exact scope** | `WorkerDefinition`, `WorkerInstance`, `Responsibility`, `WorkerGoal`, worker lifecycle state contract, stable IDs/version/revision semantics |
| **REUSED** | platform ID/value-object conventions; profile references to existing domains |
| **NEW** | Autonomous Work contract module only |
| **Explicit out of scope** | repositories, services, HTTP APIs, execution dispatch, CodeCraft, UI, persistence adapters |
| **Acceptance** | worker distinct from Agent/Principal/Task/WorkItem; responsibilities/goals cannot grant authority; lifecycle semantics frozen |
| **Proof requirements** | focused contract tests including invalid cross-identity assumptions and revision semantics |
| **Next step** | AW-1B |

| Field | Value |
|---|---|
| **ID** | AW-1B |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Freeze worker profile-reference composition |
| **Exact scope** | typed refs for governance, budget, memory, capability, codecraft, risk, schedule, escalation, collaboration, observability profiles |
| **Acceptance** | no embedded duplicate policy/memory/sandbox authority configuration; references are explicit/versionable |
| **Next step** | AW-2A |

---

## AW-2 - Durable worker state and lifecycle

| Field | Value |
|---|---|
| **ID** | AW-2A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Repository ports + in-memory adapter for WorkerDefinition/WorkerInstance/Responsibility/Goal state |
| **Dependencies** | AW-1 complete |
| **Exact scope** | optimistic revision semantics, tenant/workspace scoping where applicable, idempotent create, deterministic update conflict |
| **REUSED** | existing repository/concurrency patterns |
| **Explicit out of scope** | production DB adapter, dispatch, UI |
| **Acceptance** | durable semantics are process-independent; no silent last-write-wins |
| **Proof requirements** | repository contract and concurrency tests |
| **Next step** | AW-2B |

| Field | Value |
|---|---|
| **ID** | AW-2B |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Authoritative worker lifecycle transition service |
| **Exact scope** | PROVISIONING/ACTIVE/IDLE/WORKING/WAITING_EXTERNAL/WAITING_FOR_HUMAN/RECOVERING/DEGRADED/PAUSED/QUARANTINED/STOPPED transitions; transition guards; restart-safe rehydration |
| **Acceptance** | lifecycle independent of execution state; invalid transitions fail deterministically; restart keeps worker identity |
| **Next step** | AW-2C |

| Field | Value |
|---|---|
| **ID** | AW-2C |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | First production-qualified persistence adapter |
| **Dependencies** | AW-2A/B |
| **Acceptance** | cross-process transaction/concurrency qualification; migration/recovery tests |
| **Next step** | AW-3A |

---

## AW-3 - Principal / workspace / authority composition

| Field | Value |
|---|---|
| **ID** | AW-3A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Bind WorkerInstance to canonical Collaborative Principal without creating worker-private authority |
| **Dependencies** | AW-2; Collaborative Work authority contracts |
| **REUSED** | `CollaborativePrincipal`, Membership, Delegation, effective authority |
| **NEW** | worker-principal binding contract/repository if required |
| **Explicit out of scope** | new Principal type unless Collaborative Work ADR explicitly approves it |
| **Acceptance** | worker role/goal never authorizes; missing/revoked Principal binding fails closed |
| **Proof requirements** | authority non-amplification and stale/tampered binding tests |
| **Next step** | AW-3B |

| Field | Value |
|---|---|
| **ID** | AW-3B |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Worker-level authority snapshot/correlation into canonical Execution intake |
| **Acceptance** | resulting Execution authority <= bound effective authority; scheduling/recovery/agent changes cannot expand authority |
| **Next step** | AW-4A |

---

## AW-4 - Work intake and proactive goal evaluation

| Field | Value |
|---|---|
| **ID** | AW-4A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Event/subscription wake-up model |
| **Exact scope** | external event, queue/work assignment, schedule, human approval, dependency recovery and operator wake-ups |
| **REUSED** | existing background/task intake and hosting/event mechanisms where appropriate |
| **Acceptance** | worker can remain IDLE with zero LLM activity; duplicate event handling is idempotent |
| **Next step** | AW-4B |

| Field | Value |
|---|---|
| **ID** | AW-4B |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Bounded proactive goal evaluation |
| **Exact scope** | scheduled/cadenced goal checks, progress projection input, create/prioritize work when success criteria/SLA require action |
| **Acceptance** | cadence/budget/policy bounded; no uncontrolled self-prompt loop; every created work reason is evidenced |
| **Next step** | AW-4C |

| Field | Value |
|---|---|
| **ID** | AW-4C |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Collaborative WorkItem/Assignment bridge |
| **Dependencies** | MP-2 ownership/runtime status reviewed |
| **Acceptance** | no permanent duplicate business WorkItem model; execution Task remains separate |
| **Next step** | AW-5A |

---

## AW-5 - Execution composition and worker budgets

| Field | Value |
|---|---|
| **ID** | AW-5A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Canonical worker → Execution dispatch/correlation |
| **REUSED** | Unified Execution Runtime / Nexus / orchestration |
| **Acceptance** | worker may create many executions; execution lifecycle never becomes worker lifecycle; all executions correlate to worker/goal/work context |
| **Next step** | AW-5B |

| Field | Value |
|---|---|
| **ID** | AW-5B |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Worker accounting windows over existing durable execution budgets |
| **Exact scope** | daily/monthly cost cap, concurrency cap, recovery/codecraft caps, proactive cadence budget |
| **REUSED** | durable execution budget ledger and runtime budget enforcement |
| **Explicit out of scope** | second execution-budget engine |
| **Acceptance** | worker-level budget limits constrain all child work and survive restart |
| **Next step** | AW-6A |

---

## AW-6 - Recovery Controller

| Field | Value |
|---|---|
| **ID** | AW-6A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Freeze canonical obstacle taxonomy and recovery decision contract |
| **REUSED** | DIAG problem evidence, reliability retry, HITL, policy decisions |
| **NEW** | worker-level obstacle→strategy contract/controller |
| **Acceptance** | deterministic classification precedes strategy; DENY never becomes retry/recovery; credential/authority obstacles escalate |
| **Next step** | AW-6B |

| Field | Value |
|---|---|
| **ID** | AW-6B |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Recovery orchestration with resume-original-work semantics |
| **Acceptance** | recovery has bounded attempts/time/cost; successful recovery returns to original WorkItem/goal with evidence chain |
| **Next step** | AW-7A |

---

## AW-7 - Adaptive capability acquisition

| Field | Value |
|---|---|
| **ID** | AW-7A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Capability discovery/acquisition policy |
| **Exact scope** | ordered search Tool → Skill → Integration → approved alternate/configuration → CodeCraft; A0-A4 risk classification |
| **Acceptance** | CodeCraft is not default; A4 self-authorization impossible |
| **Next step** | AW-7B |

| Field | Value |
|---|---|
| **ID** | AW-7B |
| **Priority** | P0 |
| **Status** | BLOCKED by CodeCraft/sandbox hardening prerequisites |
| **Purpose** | A1 ephemeral generated capability path |
| **Dependencies** | CodeCraft authority defects closed; anti-downgrade strong isolation available |
| **Acceptance** | generated parser/helper static-gated, strongly sandboxed, tested, verified, ephemeral, evidence-linked |
| **Next step** | AW-7C |

| Field | Value |
|---|---|
| **ID** | AW-7C |
| **Priority** | P0/P1 |
| **Status** | NOT STARTED |
| **Purpose** | A2 scoped adaptive integration path |
| **Dependencies** | enforceable egress + scoped secret brokering |
| **Acceptance** | only approved hosts/secrets, narrow scope, runtime evidence of enforced controls |
| **Next step** | AW-7D |

| Field | Value |
|---|---|
| **ID** | AW-7D |
| **Priority** | P1 |
| **Status** | NOT STARTED |
| **Purpose** | A3 durable promotion lifecycle |
| **Exact scope** | evidence bundle → security/contract tests → shadow → canary → governed promotion → versioned publication → rollback |
| **Acceptance** | promotion is control-plane mutation; no silent global ToolRegistry persistence |
| **Next step** | AW-8A |

---

## AW-8 - Observability and evidence

| Field | Value |
|---|---|
| **ID** | AW-8A |
| **Priority** | P0 |
| **Status** | NOT STARTED |
| **Purpose** | Worker/goal/work/recovery correlation into canonical evidence |
| **REUSED** | HOS / RuntimeEvent / journals / DIAG / Proof Receipts |
| **Acceptance** | no second execution event source; every worker-triggered Execution reconstructable |
| **Next step** | AW-8B |

| Field | Value |
|---|---|
| **ID** | AW-8B |
| **Priority** | P1 |
| **Status** | NOT STARTED |
| **Purpose** | Deterministic worker-centric projections |
| **Exact scope** | fleet status, goals/KPI, work, interventions, recoveries, policy denials, generated capabilities, cost/budget, evidence drill-down |
| **Acceptance** | operator data derives from persisted state/evidence, not LLM narrative |
| **Next step** | AW-9A |

---

## AW-9 - Worker control plane

| Field | Value |
|---|---|
| **ID** | AW-9A |
| **Priority** | P0 |
| **Status** | BLOCKED by platform control-plane governance coverage |
| **Purpose** | Register/instantiate/activate/pause/resume/stop/quarantine/release operations |
| **REUSED** | canonical governance/evidence mechanisms |
| **Acceptance** | every privileged mutation authenticated, authorized, evidenced, fail closed |
| **Next step** | AW-9B |

| Field | Value |
|---|---|
| **ID** | AW-9B |
| **Priority** | P0/P1 |
| **Status** | NOT STARTED |
| **Purpose** | Governed profile changes, authority revocation binding and active-work containment |
| **Acceptance** | budget/policy/capability changes cannot bypass control-plane governance; quarantine stops new privileged work safely |
| **Next step** | AW-10A |

---

## AW-10 - Virtual Workforce reference application

| Field | Value |
|---|---|
| **ID** | AW-10A |
| **Priority** | P1 |
| **Status** | NOT STARTED |
| **Purpose** | Tier-3 reference application consuming Autonomous Work contracts |
| **Exact scope** | worker builder, fleet, detail view, goals/KPI, approvals, recovery history, budget/risk, controls |
| **Explicit out of scope** | application-owned Worker/Principal/policy/evidence semantics |
| **Acceptance** | second application can reuse same worker domain without copying business/runtime internals |
| **Next step** | AW-11A |

---

## AW-11 - Flagship proof / arena

| Field | Value |
|---|---|
| **ID** | AW-11A |
| **Priority** | P1 |
| **Status** | NOT STARTED |
| **Purpose** | Autonomous Order Operations Worker end-to-end proof |
| **Chaos corpus** | normal orders; PDF/XLS; unknown/corrupted attachment; prompt injection; missing/contradictory data; duplicate; timeout; rate limit; API drift/410; revoked credential; supplier outage; policy deny; malicious code temptation; sandbox unavailable; human decision; host restart |
| **Metrics** | goal/autonomous completion, intervention, recovery success/time, zero policy violations/unauthorized egress/isolation downgrade, escalation quality, capability pass/rollback, evidence completeness, cost/work, SLA, side-effect idempotency |
| **Acceptance** | demonstrates durable responsibility + safe obstacle recovery, not scripted happy-path demo |
| **Next step** | AW-12A |

---

## AW-12 - Production qualification

| Field | Value |
|---|---|
| **ID** | AW-12A |
| **Priority** | P0 before production claim |
| **Status** | NOT STARTED |
| **Purpose** | Security/reliability qualification for production-style autonomous workers |
| **Required gates** | strong generated-code isolation; egress enforcement; secret brokering; sandbox escape/adversarial corpus; policy bypass attempts; restart/disaster semantics; quarantine/kill; control-plane governance; concurrency/scale; rollback/promotion safety |
| **Acceptance** | four-axis maturity statement updated from evidence; no production claim without accepted qualification |

---

## 4. Cross-domain dependency register

| Dependency | Why Autonomous Work needs it | Blocking level |
|---|---|---|
| Collaborative Work | Principal/workspace/authority and future WorkItem | P0 for authority; MP-2 dependency for mature work plane |
| Governed Execution | all privileged execution/control mutations | P0 |
| Unified Execution Runtime / Nexus | actual work execution | P0 |
| Reliability/HITL | retry/wait/human pause paths | P0 |
| Diagnostics | obstacle evidence/classification input | P0 for recovery |
| Observability | canonical execution truth and projections | P0/P1 |
| CodeCraft | missing-code capability synthesis | P0 for adaptive differentiator |
| Sandbox | safe generated-code execution | P0 production blocker |
| Memory/UCL/Context | continuity across work episodes | P1 |
| Application Hosting | durable host/restart composition | P1 |
| Multiplayer AI | future human-worker-worker collaboration | P2; not core worker blocker |

---

## 5. Explicit non-goals

This plan does not authorize:

- `VirtualEmployeeAgent(BaseAgent)`,
- a second execution engine,
- a second Principal/authority system,
- a second HITL path,
- a second policy engine,
- a second memory store,
- a second sandbox,
- a second observability event source,
- a global self-modifying tool registry,
- autonomous credential acquisition,
- continuous unconstrained LLM loops,
- production claims before AW-12 evidence.

---

## 6. First implementation gate

After AW-0 documentation review, the first code task must be **AW-1A only**.

No Recovery Controller, UI, CodeCraft adaptation or worker runtime loop should be implemented before the semantic contracts and ownership boundaries are frozen.
