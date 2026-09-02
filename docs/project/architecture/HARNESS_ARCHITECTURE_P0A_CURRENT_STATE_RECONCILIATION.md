# Harness Architecture P0A — Current-State Reconciliation

## Status

**Purpose:** non-destructive synchronization of descriptive `CURRENT` state with verified repository reality before P0B implementation work.

**Baseline:** `development @ 7913f5e06a1f76af8120b183acd73ef9994e249c`  
**Audit source:** [`HARNESS_ARCHITECTURE_EVOLUTION_P0A_AS_BUILT_AUDIT.md`](../maintainers/plans/HARNESS_ARCHITECTURE_EVOLUTION_P0A_AS_BUILT_AUDIT.md)  
**Master roadmap:** [`HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md`](../overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md)

This document does **not** replace semantic architecture authority. UEA, UER, Tools, Skills, Context Engineering, Background Tasks, Observability, Platform Plugins, Governance, and other domain documents remain authoritative for the semantics they own.

For **descriptive CURRENT-state claims only**, this reconciliation document supersedes older statements in the listed documents when those statements conflict with repository evidence at the baseline above.

This precedence is deliberately narrow:

```text
semantic contract / invariant
    → owning canonical architecture document

CURRENT implementation status
    → verified code/tests at audit baseline
    → this reconciliation when older CURRENT prose conflicts
```

No implementation session may use an older CURRENT sentence to rebuild a component classified here as `DONE` or `PARTIAL`.

---

# 1. Unified Execution Runtime — reconciled CURRENT

## DONE / real foundations

The current Python runtime already contains:

- canonical `ExecutionId` contract and active identity propagation,
- root `ExecutionRuntime`,
- `ExecutionBoundary`,
- `ExecutionIdentityBinding`,
- strategy-neutral `StrategyExecutionRouter`,
- `RuntimeEvent.execution_id`,
- child Execution lineage through `parent_execution_id`,
- `ChildExecutionRunner`,
- child authority narrowing,
- child budget allocation/reservation primitives,
- canonical work-port / host-task execution surfaces,
- graph-node routing through child Execution on migrated paths.

Primary code evidence:

- `intergrax/contracts/execution_identity.py`
- `intergrax/runtime/execution/runtime.py`
- `intergrax/runtime/execution/boundary.py`
- `intergrax/runtime/execution/child.py`
- `intergrax/runtime/execution/strategy_router.py`
- `intergrax/runtime/execution/execution_work_port.py`
- `intergrax/runtime/execution/host_task.py`
- `intergrax/runtime/events/runtime_event.py`

Therefore the following older CURRENT claims are **superseded**:

- “canonical `ExecutionId` is not implemented”,
- “the identity spine stops at AttemptId → EventId”,
- “neutral Execution Boundary is only target”,
- “RuntimeEvent lacks ExecutionId”.

## PARTIAL / remaining convergence

UER is **not complete**. Remaining program scope includes:

- adoption on every remaining canonical entry path,
- subtree cancellation and cross-provider cancellation semantics,
- full retry-ownership taxonomy,
- complete budget dimensions/accounting,
- remaining authority convergence,
- neutral result ABI convergence,
- distributed/background/delegated execution convergence,
- effective profile revision binding,
- model-request evidence foundation.

Do **not** equate existing UER foundations with universal production qualification.

---

# 2. Execution Tree / checkpoint / recovery — reconciled CURRENT

## DONE / real foundations

`RuntimeCheckpoint` and Execution-tree recovery already include substantial canonical state:

- `ExecutionTreeSnapshot`,
- root/child Execution entries,
- parent lineage,
- tree validation and cycle checks,
- Execution status,
- historical-versus-active resume planning,
- completed-work adoption/skipping,
- interrupted-work resumption,
- UAEP cursor/state,
- graph/node state,
- prior outputs,
- pending decisions,
- pending human request.

Primary evidence:

- `intergrax/runtime/long_running/runtime_checkpoint.py`
- `intergrax/runtime/long_running/execution_tree_checkpoint.py`
- checkpoint/resume integration tests under `tests/unit/runtime/execution/` and long-running runtime tests.

Therefore the older statement “RuntimeCheckpoint does not persist canonical Execution Tree” is **superseded**.

## PARTIAL / remaining recovery audit

Still requires explicit verification/closure for supported failure models:

- durable checkpoint-store semantics across all canonical paths,
- budget reservation/consumption restoration,
- meaningful-side-effect fence/idempotency restoration,
- transport/delivery cursor recovery,
- credential reference/lease restoration where applicable,
- delegated external-child recovery,
- background-worker crash/restart semantics,
- checkpoint commit ordering relative to meaningful work,
- inspectable retry/redelivery/resume relationships.

Checkpoint remains durable state, **not runtime identity authority**.

---

# 3. Context Engineering — reconciled CURRENT

`ContextProvider` is **CURRENT**, not a new provider seam to invent.

Existing implementation includes provider contracts/context, orchestrator integration, builtin providers, workspace context, session semantic recall, and tool-output/context sources.

Primary evidence:

- `intergrax/context/protocols.py`
- `intergrax/context/contracts.py`
- `intergrax/context/orchestrator.py`
- `intergrax/context/providers/`
- `intergrax/runtime/nexus/context/`

Remaining work is provider lifecycle/provenance/lifetime/replacement semantics, lazy activation where justified, universal canonical adoption, explainability, and tighter integration with artifacts/compaction/model-request evidence.

Do **not** create a second ContextProvider architecture.

---

# 4. Platform Plugins — reconciled CURRENT

Platform Plugins canonical architecture is frozen and the PLATFORM-PLUGIN-1..9 implementation roadmap is closed for its defined package/control-plane scope.

Current model remains:

```text
COMMON PLATFORM COORDINATION
+
DOMAIN-OWNED CAPABILITY CONTRACTS
```

It coordinates packaging, discovery, manifest/config/secrets/DI conventions, trust/qualification vocabulary, compatibility, and lifecycle vocabulary. It intentionally does **not** become a global Tool/Skill/RAG/Memory execution registry or universal lifecycle engine.

Roadmap Initiative Q therefore applies only to **new dynamic/scoped runtime registration requirements** such as reversible owner-scoped activation, version coexistence/draining, atomic activation, and later governed runtime extension mounting.

Do **not** reopen or rebuild the closed Platform Plugins program merely to implement Initiative Q.

---

# 5. Background Tasks — reconciled CURRENT

Background Tasks is not merely conceptual. Current repository code contains real:

- `TaskRegistry`,
- `WorkerRuntime`,
- task definitions/handlers,
- background worker factory integration,
- transport/provider paths,
- execution/audit identity bridges on implemented paths.

Primary evidence:

- `intergrax/background_tasks/registry.py`
- `intergrax/background_tasks/worker_runtime.py`
- `intergrax/background_tasks/`
- `applications/local_workspace_application/host/background_worker_factory.py`

Older prose that says `TaskRegistry` / `WorkerRuntime` are not implemented is therefore **superseded as an implementation-existence claim**.

This does **not** promote the whole background stack to universal production maturity. Remaining convergence includes canonical UER identity across all transports, owner fencing, durable control surfaces, recovery/redelivery semantics, cancellation, provider-neutral execution, and qualification.

Initiative J remains migration/hardening + UX, **not a new Job runtime**.

---

# 6. Observability — reconciled CURRENT

`RuntimeEvent.execution_id` exists on the canonical event contract and execution identity is propagated on migrated runtime paths.

Therefore descriptive CURRENT statements saying the RuntimeEvent spine lacks `ExecutionId` are **superseded**.

Observability still does not own or mint execution identity. UER/lifecycle establishes identity; Observability records and projects canonical evidence.

Remaining work is coverage/convergence, not creation of a second identity model.

---

# 7. Skills — reconciled CURRENT and real blocker

Skills infrastructure is mature and CURRENT:

- `SkillManifest`,
- catalog/registry/profile,
- deterministic `SkillResolver`,
- transitive dependencies/cycle rejection,
- Tool requirements,
- plugin/import paths,
- large first-party catalog,
- optional AHI recommendation hook.

Do not rebuild Skills.

However P0A confirms a **real authority GAP**, not documentation drift:

`extend_tool_profile_for_skills()` currently expands `ToolProfile.enabled` from Skill requirements.

Target invariant:

```text
skill-required tool_ids ⊆ host ToolProfile availability
```

A Skill may declare requirements; it may not grant host capability availability.

This is **P0-SAFETY-2** and remains open for P0B code remediation.

Additional PARTIAL items include version identity, resolved-pack provenance retention, and universal prompt/policy bridge consumption.

---

# 8. Tools — reconciled CURRENT and real blocker

Tool contracts, registry/runtime, policy surfaces, scope policy, and ToolRuntime are real CURRENT foundations.

However P0A confirms a **real authority GAP**:

`resolve_allowed_tools_from_config(config, explicit=...)` currently returns the explicit caller list directly when present. This can bypass a stricter `RuntimePolicyBundle.tool_access` allow-list instead of intersecting with it.

Target invariant:

```text
effective tool authority
=
host availability
∩ agent/skill requirement
∩ RuntimePolicyBundle
∩ modality/plan narrowing
∩ invoker/per-call narrowing
```

No downstream caller list may widen stricter upstream authority.

This is **P0-SAFETY-1** and remains open for P0B code remediation.

---

# 9. Sandbox — reconciled CURRENT

Sandbox is **PARTIAL**, not absent.

Existing repository surfaces include:

- runtime sandbox models/session/manager,
- application sandbox wiring,
- hosted sandbox integration contract/resolution,
- sandbox Tool bundle,
- sandbox Skill bundle,
- CodeCraft sandbox resolution.

Initiative H is convergence into a canonical ExecutionEnvironment/isolation contract and operator UX; it is not permission to build a second sandbox subsystem.

---

# 10. Governance / meaningful side effects — reconciled CURRENT

Meaningful-side-effect governance is a real CURRENT foundation:

- `MeaningfulSideEffectRequest`,
- `RuntimePolicyEngine.evaluate_meaningful_side_effect`,
- policy rules/decisions,
- enforcement-gate reuse,
- HITL/continuation paths on implemented flows.

P0B must audit **fresh enforcement coverage and bypasses**, not invent a second side-effect policy runtime.

Hard invariant remains:

```text
proposal != permission != execution
```

and meaningful effects require fresh governed enforcement at the actual effect boundary.

---

# 11. P0A classification summary

| Area | Classification | Implementation consequence |
|---|---|---|
| ExecutionId | DONE | do not rebuild |
| root ExecutionRuntime | DONE | do not rebuild |
| ExecutionBoundary | DONE | do not rebuild |
| StrategyExecutionRouter | DONE | converge adoption |
| RuntimeEvent.execution_id | DONE | converge coverage |
| child Execution lineage | DONE/PARTIAL | extend, do not replace |
| child authority | DONE/PARTIAL | conformance hardening |
| child budget primitives | DONE/PARTIAL | complete dimensions/recovery |
| Execution-tree checkpoint | DONE/PARTIAL | close recovery gaps |
| ContextProvider seam | DONE | lifecycle/provenance hardening only |
| Platform Plugins core program | DONE for defined scope | do not reopen |
| Background TaskRegistry/WorkerRuntime | DONE as implementation existence | converge/qualify |
| ToolRuntime | DONE/PARTIAL | P0-SAFETY-1 required |
| Skills | DONE/PARTIAL | P0-SAFETY-2 required |
| Sandbox | PARTIAL | converge existing surfaces |
| ModelRequest reconstruction | GAP/PARTIAL | implement canonical evidence identity |
| ProfileResolution/effective snapshot | PARTIAL/GAP | P0C/P1 |
| Runtime inspection | GAP/PARTIAL | P0C/P1 |
| Runtime invariant runner | GAP/PARTIAL | P0C/P1 |
| Human continuation seam | PARTIAL/GAP | P0C/P1 |

---

# 12. P0A implementation gate

For all future implementation sessions in this program:

1. Read the master roadmap.
2. Read the P0A as-built audit.
3. Apply this CURRENT-state reconciliation before interpreting older descriptive CURRENT prose.
4. Use owning domain docs for semantic contracts/invariants.
5. Revalidate the relevant code paths against current `development` HEAD.
6. Never convert a `DOC-DRIFT` item into a feature implementation task.

The next code-changing phase is **P0B — Safety and Authority Closure**, beginning with P0-SAFETY-1 and P0-SAFETY-2.
