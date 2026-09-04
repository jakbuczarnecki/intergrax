# Autonomous Work - Implementation Plan

**Architecture (1:1):** [`../../architecture/AUTONOMOUS_WORK.md`](../../architecture/AUTONOMOUS_WORK.md)  
**Extended architecture depth:** [`../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md)  
**ADR:** [`ADR-AW-001`](../../technical/adr/entries/2026-09-02/ADR-AW-001.md)  
**Architecture governance:** [`../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)

**Status:** Domain registered — **AW-0 CLOSED / FROZEN** (canonical documentation independently accepted; runtime implementation NOT STARTED)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** this read-scope block + one active `AW-*` row only.
- **Architecture:** [`../../architecture/AUTONOMOUS_WORK.md`](../../architecture/AUTONOMOUS_WORK.md) read-scope + sections named in the active row.
- **Extended depth:** load only the satellite section named in the active `AW-*` row — [`../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md). Never load the full satellite by default.
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
| AW-0 | Canonical architecture, ADR, documentation registration | **CLOSED / FROZEN** — AW-0G independently accepted |
| AW-1 | Core semantic contracts | **DONE** |
| AW-2 | Durable worker state and lifecycle | **DONE** |
| AW-3 | Principal / authority / workspace composition | **DONE** |
| AW-4 | Work intake and proactive goal evaluation | **IN PROGRESS** (AW-4A DONE, AW-4B DONE, AW-4C PARTIALLY_COMPLETE) |
| AW-5 | Worker → execution composition and budgets | **IN PROGRESS** (AW-5A DONE, AW-5B IN PROGRESS) |
| AW-6 | Recovery Controller and obstacle taxonomy | NOT STARTED |
| AW-7 | Adaptive capability acquisition | NOT STARTED |
| AW-8 | Worker observability and evidence correlation | NOT STARTED |
| AW-9 | Worker control plane | NOT STARTED |
| AW-10 | Virtual Workforce reference application | NOT STARTED |
| AW-11 | Flagship proof / arena | NOT STARTED |
| AW-12 | Production security / reliability qualification | NOT STARTED |
| AW-13 | Real-vendor end-to-end qualification | NOT STARTED |
| AW-14 | Final documentation and architecture audit | NOT STARTED |

**Final acceptance sequence:** AW-12 → AW-13 → AW-14 → **AUTONOMOUS WORK FINAL ACCEPTANCE**

---

## AW-0 - Architecture and governance

| ID | Task | Status |
|---|---|---|
| AW-0A | Strategic market + Intergrax capability audit | **DONE** |
| AW-0B | Independent architecture/gap review | **DONE** |
| AW-0C | Freeze DOMAIN vs FEATURE classification (`AUTONOMOUS_WORK` = DOMAIN) | **DONE** |
| AW-0D | Create canonical architecture/plan pair | **DONE** |
| AW-0E | ADR-AW-001 ownership/classification decision | **DONE** |
| AW-0F | Register domain in architecture/documentation hubs and public routes | **DONE** |
| AW-0G | Independent canonical-doc audit before runtime implementation | **DONE** |

**Exit gate:** **PASSED** — canonical docs independently accepted with no unresolved ownership conflict. Runtime implementation may begin with AW-1A only.

---

## AW-1 - Core semantic contracts

| Field | Value |
|---|---|
| **ID** | AW-1A |
| **Priority** | P0 |
| **Status** | **DONE** |
| **Purpose** | Freeze minimum Tier-0 Autonomous Work contracts |
| **Dependencies** | AW-0G accepted |
| **Exact scope** | `WorkerDefinition`, `WorkerInstance`, `Responsibility`, `WorkerGoal`, worker lifecycle state contract, stable IDs/version/revision semantics, **conceptual `WorkContinuityState` contract and reference fields** |
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Detailed domain model, §Long-Horizon Work Continuity §Work Continuity State |
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
| **Status** | **DONE** |
| **Purpose** | Freeze worker profile-reference composition |
| **Exact scope** | typed refs for governance, budget, memory, capability, codecraft, risk, schedule, escalation, collaboration, observability profiles |
| **Acceptance** | no embedded duplicate policy/memory/sandbox authority configuration; references are explicit/versionable, strongly typed, immutable, and carry no implementation/provider identity |
| **Next step** | AW-2A |

---

## AW-2 - Durable worker state and lifecycle

| Field | Value |
|---|---|
| **ID** | AW-2A |
| **Priority** | P0 |
| **Status** | DONE |
| **Purpose** | Repository ports + in-memory adapter for WorkerDefinition/WorkerInstance/Responsibility/Goal state |
| **Dependencies** | AW-1 complete |
| **Exact scope** | optimistic revision semantics, tenant/workspace scoping where applicable, idempotent create, deterministic update conflict, **durable continuity checkpoint/revision semantics for `WorkContinuityState`** |
| **REUSED** | existing repository/concurrency patterns |
| **Explicit out of scope** | production DB adapter, dispatch, UI |
| **Acceptance** | durable semantics are process-independent; no silent last-write-wins |
| **Proof requirements** | repository contract and concurrency tests |
| **Next step** | AW-2B |

| Field | Value |
|---|---|
| **ID** | AW-2B |
| **Priority** | P0 |
| **Status** | DONE |
| **Purpose** | Authoritative worker lifecycle transition service |
| **Exact scope** | PROVISIONING/ACTIVE/IDLE/WORKING/WAITING_EXTERNAL/WAITING_FOR_HUMAN/RECOVERING/DEGRADED/PAUSED/QUARANTINED/STOPPED transitions; transition guards; restart-safe rehydration |
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Worker lifecycle, §Long-Horizon Work Continuity §Checkpoint and restart continuity |
| **Acceptance** | lifecycle independent of execution state; invalid transitions fail deterministically; restart keeps worker identity |
| **Next step** | AW-2C |

| Field | Value |
|---|---|
| **ID** | AW-2C |
| **Priority** | P0 |
| **Status** | **DONE** |
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
| **Status** | **DONE** |
| **Purpose** | Bind WorkerInstance to canonical Collaborative Principal without creating worker-private authority |
| **Dependencies** | AW-2; Collaborative Work authority contracts |
| **REUSED** | `CollaborativePrincipal`, Membership, Delegation, effective authority |
| **NEW** | `WorkerPrincipalBinding` (tenant/workspace/principal scoped), `ResolvedWorkerPrincipal`, repository; schema v1→v2 migration |
| **Explicit out of scope** | new Principal type unless Collaborative Work ADR explicitly approves it |
| **Acceptance** | worker role/goal never authorizes; missing/revoked Principal binding fails closed |
| **Proof requirements** | authority non-amplification and stale/tampered binding tests |
| **Next step** | AW-3B |

| Field | Value |
|---|---|
| **ID** | AW-3B |
| **Priority** | P0 |
| **Status** | DONE |
| **Purpose** | Worker-level authority snapshot/correlation into canonical Execution intake |
| **Acceptance** | resulting Execution authority <= bound effective authority; scheduling/recovery/agent changes cannot expand authority |
| **Next step** | AW-4A |

---

## AW-4 - Work intake and proactive goal evaluation

| Field | Value |
|---|---|
| **ID** | AW-4A |
| **Priority** | P0 |
| **Status** | **DONE** |
| **Purpose** | Event/subscription wake-up model |
| **Exact scope** | external event, queue/work assignment, schedule, human approval, dependency recovery and operator wake-ups; **restore orientation from durable continuity state before accepting/creating next work** |
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Wake-up and scheduling semantics, §Long-Horizon Work Continuity §Restore-orientation flow |
| **REUSED** | existing background/task intake and hosting/event mechanisms where appropriate |
| **Acceptance** | worker can remain IDLE with zero LLM activity; duplicate event handling is idempotent |
| **Next step** | AW-4B |

| Field | Value |
|---|---|
| **ID** | AW-4B |
| **Priority** | P0 |
| **Status** | **DONE** |
| **Purpose** | Bounded proactive goal evaluation |
| **Exact scope** | scheduled/cadenced goal checks, progress projection input, create/prioritize work when success criteria/SLA require action |
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Reactive and proactive work |
| **Acceptance** | cadence/budget/policy bounded; no uncontrolled self-prompt loop; every created work reason is evidenced |
| **Next step** | AW-4C |

| Field | Value |
|---|---|
| **ID** | AW-4C |
| **Priority** | P0 |
| **Status** | **PARTIALLY_COMPLETE / boundary ready** — runtime activation **BLOCKED_BY_MP2** |
| **Purpose** | Collaborative WorkItem/Assignment bridge |
| **Dependencies** | MP-2 ownership/runtime status reviewed — canonical WorkItem intake **not implemented** |
| **Exact scope** | `GoalEvaluationDecision.ACTION_REQUIRED` → `CollaborativeWorkRequest` → `CollaborativeWorkIntakePort` → future MP-2 WorkItem/Assignment |
| **Acceptance** | no permanent duplicate business WorkItem model; execution Task remains separate; fail-closed when MP-2 unavailable |
| **Blocker** | Collaborative Work MP-2 canonical WorkItem/Assignment runtime |
| **Next step** | AW-5A (may proceed independently of MP-2 runtime activation) |

---

## AW-5 - Execution composition and worker budgets

| Field | Value |
|---|---|
| **ID** | AW-5A |
| **Priority** | P0 |
| **Status** | **DONE** |
| **Purpose** | Canonical worker → Execution dispatch/correlation |
| **REUSED** | `ExecutionRuntime`, `ExecutionBoundary`, `resolve_root_execution_context`, AW-3B `WorkerExecutionAdmissionService`, Collaborative Principal binding, `RuntimePolicyEngine` |
| **NEW** | `WorkerExecutionDispatchService`, `WorkerExecutionDispatchRequest/Result`, `WorkerExecutionSource`, `WorkerExecutionCorrelation`, `RootExecutionAuthorityAdmissionPort`, `RuntimeExecutionPolicyAdmissionPort`, `CanonicalExecutionIntakePort`, `CanonicalExecutionRuntimeAdapter` |
| **Trust boundary** | AW-3B collaborative authority context → independent Runtime/Governance policy admission → trusted `ParentExecutionAuthority` → `ExecutionRuntime.execute`. AW does **not** mint trusted execution authority. AW does **not** own Run/Attempt/Execution. |
| **Acceptance** | worker may create many executions; execution lifecycle never becomes worker lifecycle; collaborative ALLOW ≠ runtime ALLOW; fail-closed on denied/unavailable admission; no new DB schema for correlation (runtime IDs sufficient) |
| **Idempotency** | **NOT IDEMPOTENT** — same `RunId`/`AttemptId` mints a new `ExecutionId` per invocation; caller `RunId` is correlation only unless canonical intake adds durable idempotency |
| **Next step** | AW-5B |

| Field | Value |
|---|---|
| **ID** | AW-5B |
| **Priority** | P0 |
| **Status** | **IN PROGRESS** |
| **Purpose** | Worker accounting windows over existing durable execution budgets |
| **Exact scope** | daily/monthly cost cap, concurrency cap, recovery/codecraft caps, proactive cadence budget |
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Budgets |
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
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Recovery Controller |
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
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Capability acquisition, §A0–A4, §CodeCraft recovery, §Durable capability promotion |
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
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Observability, §Long-Horizon Work Continuity §Context-efficiency observability |
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
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Control plane |
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
| **Chaos corpus** | normal orders; PDF/XLS; unknown/corrupted attachment; prompt injection; missing/contradictory data; duplicate; timeout; rate limit; API drift/410; revoked credential; supplier outage; policy deny; malicious code temptation; sandbox unavailable; human decision; host restart; **long-horizon stress (100 / 1k / 10k / 100k historical events with bounded active context)** |
| **Architecture depth** | [`satellites/AUTONOMOUS_WORK_extended_depth.md`](../../architecture/satellites/AUTONOMOUS_WORK_extended_depth.md) §Enterprise examples, §Long-Horizon Work Continuity §Proof requirements |
| **Metrics** | goal/autonomous completion, intervention, recovery success/time, zero policy violations/unauthorized egress/isolation downgrade, escalation quality, capability pass/rollback, evidence completeness, cost/work, SLA, side-effect idempotency; **continuation success after restart/idle, duplicate-work rate, lost-open-work rate, active context tokens per step, retrieved-data volume, full-history-read count (near zero), artifact reuse rate, recall precision, stale-context usage rate** |
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
| **Next step** | AW-13A |

---

## AW-13 - Real-vendor end-to-end qualification

| Field | Value |
|---|---|
| **ID** | AW-13A |
| **Priority** | P0 before final enterprise acceptance |
| **Status** | NOT STARTED |
| **Purpose** | Full Autonomous Work E2E qualification against real configured vendors/backends |
| **Dependencies** | AW-12 accepted; flagship/reference flows available |
| **Next step** | AW-13B |

**Hard invariant — real means real:**

> Mock, fake, stub, monkeypatch or in-memory stand-in cannot satisfy a real-vendor acceptance claim. Mocks/fakes may exist in unit/integration suite; AW-13 acceptance requires a real provider.

Examples:

- PostgreSQL → real PostgreSQL instance
- LLM → real configured provider/model endpoint
- external tool → real provider sandbox/test account or canonical vendor test environment

Provider unavailable must not be marked PASS.

**Exact scope — representative production-qualified providers:**

AW-13A tests the full mechanism through real production boundaries. At minimum:

- real production persistence provider,
- real LLM/model provider where the scenario requires inference,
- real tool/integration provider for selected flows,
- real governance/HITL path where configured,
- real sandbox/code-execution provider where adaptive capability is exercised,
- real observability/evidence path.

Not every vendor in Intergrax is required. Test representative production-qualified providers for capabilities required by the scenario.

**Vendor configuration:**

Provider selection uses existing Intergrax profiles, provider bindings, integration configuration, and repository abstractions. Do not hardcode specific vendors into Autonomous Work semantics. AW-13 qualifies **capability × selected real provider**, not domain architecture tied to one brand.

**Full E2E flow — at least one complete scenario:**

create/register Worker → activate → goal/work intake → contextual orientation → tool/model execution → governance decision where applicable → persistence → continuity checkpoint → external wait/recovery if exercised → restart/reconstruction → resume → goal completion/escalation → evidence reconstruction.

Must not be repository-only testing.

**Real restart / recovery:**

- host/process restart,
- persisted Worker identity,
- lifecycle restoration,
- `WorkContinuityState` reconstruction,
- outstanding work preservation,
- no full-history replay dependency,
- resume from canonical checkpoint.

**Real failure injection — controlled cases where possible:**

provider timeout; rate limit; temporary outage; invalid/changed provider response; revoked/invalid credential; database connection interruption; restart during active work; duplicate delivery/request; human decision timeout; sandbox unavailable.

Not every provider must support every fault type. Report each case as **SUPPORTED**, **NOT SUPPORTED BY QUALIFICATION ENVIRONMENT**, or **FAILED**. Do not pretend coverage.

**Security / governance E2E:**

AW-13 does not replace AW-12 (security/reliability qualification). AW-13 is real end-to-end composition proof. Verify at minimum:

- policy DENY is not bypassed,
- revoked authority fails closed,
- quarantine prevents privileged continuation,
- generated capability cannot expand authority,
- no silent storage fallback,
- no silent provider fallback that changes security semantics.

**Real side-effect safety:**

When the scenario performs real side effects: idempotency, deduplication, retry safety, side-effect correlation, evidence. Use vendor sandbox/test environment where production side effect would be unsafe.

**Provider provenance — every real-vendor qualification run records (no secrets):**

repository commit SHA; qualification suite version; scenario version; provider capability; provider/vendor identifier; backend/model version when deterministically available; schema version; configuration provenance; start/end timestamp; result; known limitations.

Never record: API key, password, raw DSN, bearer token.

**Reproducibility:**

Qualification must be repeatable. Prefer declarative qualification config + runner + machine-readable result + human-readable report. Reuse existing Intergrax provider qualification/evidence patterns. Do not build a second global qualification framework.

**AW-13A acceptance — PASS only if all hold:**

1. real provider persistence executed,
2. real model/provider executed for model-dependent scenario,
3. real integration executed where scenario requires it,
4. Worker lifecycle works through full flow,
5. continuity survives real restart,
6. recovery works for at least one representative real failure,
7. governance/HITL semantics preserved,
8. no unauthorized authority expansion,
9. evidence reconstructs full business flow,
10. zero mock/fake used to support real-vendor claim,
11. provider provenance recorded,
12. deterministic PASS/FAIL evidence exists.

| Field | Value |
|---|---|
| **ID** | AW-13B |
| **Priority** | P1/P0 before portability claim |
| **Status** | NOT STARTED |
| **Purpose** | Demonstrate provider/configuration portability without changing Autonomous Work domain semantics |
| **Dependencies** | AW-13A |
| **Next step** | AW-14A |

**AW-13B scope:**

Run the same semantic qualification flow on at least two provider combinations where the platform has real supported alternatives. Example: same Worker domain, same repository ports, same lifecycle, same goals, same governance → provider set A / provider set B.

Do not require two vendors for a capability where the platform supports only one production-qualified provider.

**AW-13B core claim:**

> Provider swap changes infrastructure/configuration, not Autonomous Work domain semantics.

Must not modify `WorkerLifecycleService`, Worker contracts, Worker goal semantics, authority semantics, or repository ports to pass the second provider.

**AW-13B acceptance — PASS if:**

- provider change occurs through canonical configuration/binding,
- domain/service code unchanged,
- semantic qualification suite passes,
- provider-specific differences remain in adapters/integrations,
- evidence records both provider runs,
- no vendor branching appears in Autonomous Work domain.

---

## AW-14 - Final documentation and architecture audit

| Field | Value |
|---|---|
| **ID** | AW-14A |
| **Priority** | P0 final acceptance |
| **Status** | NOT STARTED |
| **Purpose** | Full implementation-to-documentation reconciliation audit |
| **Dependencies** | AW-13 accepted |
| **Next step** | AW-14B |

**Documentation inventory — audit must cover at minimum:**

- [`../../architecture/AUTONOMOUS_WORK.md`](../../architecture/AUTONOMOUS_WORK.md),
- architecture satellite(s),
- this plan,
- ADR-AW-*,
- overview/market docs for Virtual Workforce,
- operator documentation,
- developer documentation,
- configuration/persistence docs,
- qualification/evidence docs,
- proof/scenario docs,
- API/control-plane docs created during implementation.

Do not assume specific paths for future documents. Perform inventory first in the final audit.

**Claim → Code → Test → Evidence matrix — central AW-14 artifact:**

| Architectural / production claim | Owning code/component | Automated tests | Real-vendor qualification evidence | Documentation | Verdict |
|---|---|---|---|---|---|

Minimum verdict statuses: **SUPPORTED**, **PARTIALLY SUPPORTED**, **UNSUPPORTED**, **STALE DOCUMENTATION**, **MISSING EVIDENCE**.

**Invariant audit — all `AW-INV-*`:**

For each invariant verify: still applies; where implemented; how tested; production evidence exists; documentation matches real behavior. Invariant presence in Markdown alone is not sufficient.

**Persistence audit:**

Every durable Autonomous Work state must use abstractions. Hard audit path: durable read/write → domain repository port → configured adapter/provider. Search and classify: direct SQL, psycopg, sqlite, Redis, filesystem writes, JSON persistence, local mutable dictionaries, hidden process caches. Any production bypass requires remediation.

**Authority audit:**

Verify: capability may grow; authority must not. Worker role does not authorize; Worker goals do not authorize; generated capability does not increase authority; Principal binding canonical; revoked authority fails closed; recovery cannot bypass policy; CodeCraft cannot self-authorize A4; governance controls privileged mutations.

**Lifecycle audit:**

Exactly one canonical Worker lifecycle semantics; no duplicated lifecycle engines; STOPPED semantics; QUARANTINED semantics; Worker lifecycle independent from Execution lifecycle; transition paths match docs; persistence/restart preserves lifecycle.

**Continuity audit:**

Bounded active context; no full-history replay dependency; `WorkContinuityState` persisted; provenance-preserving recall; long-horizon tests; restart continuation; no hidden chat transcript dependency. Explicitly verify AW-INV-21 through AW-INV-26.

**Recovery audit:**

Compare documented obstacle taxonomy with actual controller. Verify: retry; wait/reschedule; rate limit; revoked credentials; DENY; human ambiguity; alternate plan; schema/API drift; missing capability; quarantine. No silent semantic collapse (e.g. DENY → retry).

**Adaptive capability audit:**

Verify A0–A4 and promotion pipeline: `CraftResult` → evidence → tests → shadow → canary → governed promotion → durable version → rollback. Every documented capability claim must have evidence.

**Observability audit:**

Worker state/evidence; execution correlations; recovery evidence; policy denials; cost/budgets; interventions; generated capabilities; operator projections. Do not create fictitious Execution records for observability only.

**Control-plane audit:**

Every privileged mutation: authenticate → authorize → policy/governance → mutation → evidence. No hidden admin bypass.

**Dead / stale documentation:**

Find: stale names; abandoned architecture; superseded diagrams; outdated roadmap status; dead links; stale TODO; unsupported production claims; duplicate explanations that disagree; old terminology conflicting with canonical vocabulary. Do not leave documentation "historically almost correct".

**Diagram reconciliation:**

Every important architecture diagram must match real dependency directions. Especially: Application → Autonomous Work → repository/runtime/governance abstractions → configured platform capabilities. Vendor must not appear as domain dependency.

| Field | Value |
|---|---|
| **ID** | AW-14B |
| **Priority** | P0 final acceptance |
| **Status** | NOT STARTED |
| **Purpose** | Produce formal Autonomous Work final enterprise audit and remediation verdict |
| **Dependencies** | AW-14A |
| **Next step** | FINAL ACCEPTANCE / remediation |

**Expected artifact (canonical location at execution time):**

[`docs/project/maintainers/audits/AUTONOMOUS_WORK_FINAL_ENTERPRISE_AUDIT.md`](../audits/AUTONOMOUS_WORK_FINAL_ENTERPRISE_AUDIT.md)

If repo convention differs at execution time, use canonical audit location.

**Final audit verdict — allowed values only:**

- **ACCEPTED**
- **ACCEPTED WITH REMEDIATIONS**
- **NOT ACCEPTED**

No marketing verdict.

**Final acceptance rule:**

Autonomous Work receives final enterprise acceptance only if implementation + automated tests + real-vendor E2E evidence + security/reliability qualification + documentation audit are consistent.

Core rule:

> No undocumented production behavior and no documented production capability without evidence.

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
| Memory/UCL/Context | continuity across work episodes; **long-horizon work continuity semantics** | P0 for worker orientation; P1 for full integration |
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

AW-12 evidence is required before any production claim. AW-13 real-vendor evidence and AW-14 final audit are additionally required for final Autonomous Work enterprise acceptance.

Gate distinction:

- **AW-12:** production safety qualification gate
- **AW-13 / AW-14:** final enterprise acceptance gates

---

## 6. First implementation gate

With AW-0 closed and AW-0G accepted, the first code task must be **AW-1A only**.

No Recovery Controller, UI, CodeCraft adaptation or worker runtime loop should be implemented before the semantic contracts and ownership boundaries are frozen.
