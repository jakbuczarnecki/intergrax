# Execution ↔ Governance Boundary Model (NPSC-4.2-H1)

**Classification:** `MAINTAINER_CERTIFICATION`  
**Status:** `CERTIFIED` (audit NPSC-4.2-H1 governance dependency boundary freeze on `development`)  
**Audience:** Maintainers, enterprise qualification, architecture gates  

**Parent:** [`EXECUTION_ENGINE_OWNERSHIP_MODEL.md`](EXECUTION_ENGINE_OWNERSHIP_MODEL.md) (EE-A1)  
**Identity freeze:** [`EXECUTION_IDENTITY_AUTHORITY_MODEL.md`](EXECUTION_IDENTITY_AUTHORITY_MODEL.md) (EE-A2)  
**Agent runtime governance:** [`intergrax/contracts/agent_runtime_governance.py`](../../../intergrax/contracts/agent_runtime_governance.py)  
**Bypass inventory (frozen):** [`../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md)

Re-verify this freeze against the current GitHub `development` branch before downstream certification; repository source is the sole authority of truth.

---

## Global Freeze Statement

The **Execution Engine** consumes governance outcomes; it does **not** own policy evaluation, approval logic, or alternate authorization authority.

Production execution **must not**:

- implement enterprise policy rules locally,
- mint governance decisions (`PolicyDecision`, terminal allow/deny) inside `intergrax/runtime/execution/**`,
- bypass configured governance ports on tool or side-effect paths,
- rewrite or downgrade an upstream governance decision.

Governance **must not**:

- mint `RunId`, `AttemptId`, or `ExecutionId`,
- own `RuntimeEventPersistence`, `RuntimeEventBus`, checkpoint durability, replay drivers, or recovery scheduling,
- call `execute()` on behalf of the execution lifecycle owner.

---

## Canonical decision flow

```text
Intent (application / task)
        |
        v
ExecutionRuntime  (lifecycle owner — no policy engine)
        |
        v
Governance Evaluation Port
  (agent_runtime_governance, control-plane mutation boundary,
   meaningful side-effect authorization, declarative policy → HITL bridge)
        |
        v
Governance Decision  (DecisionId / evaluation artifacts — not execution IDs)
        |
        v
ExecutionBoundary  (identity + authority propagation)
        |
        v
Nexus / ChildExecutionRunner / RuntimeToolInvoker
        |
        v
Provider / integration adapter
```

**Forbidden shortcut:**

```text
ExecutionRuntime
      +--> local_policy()
      +--> hidden_allow()
      +--> internal_decision_engine()
```

---

## Ownership matrix

| Concern | Governance owner | Execution consumer |
| --- | --- | --- |
| Tool invocation allow/deny/HITL | `agent_governance/**`, `RuntimeToolInvoker._require_agent_runtime_governance` | Stops or escalates on port outcome |
| Control-plane mutation authorization | `governance/control_plane_mutation_*` | Domain executors mutate only after ALLOW |
| Post-run / replay governance guard | `governance/execution_guard.py` | Does not re-open completed lifecycle |
| Declarative policy → HITL | `runtime/policy/**` + `declarative_policy_hitl_bridge.py` | Pauses; human resolution is governance outcome |
| Recovery re-admission | `RecoveryAdmissionPort` + injected `PolicyDecision` | Deny-only consumption in partial recovery |
| Retry / backoff / timeout | `execution/retry/**`, Nexus `PolicyEnforcer` (operational resilience) | **Not** enterprise governance policy |
| Slot partial recovery disposition | `contracts/partial_recovery.evaluate_slot_recovery_policy` | Mechanical WAIT/RECOVER — not governance rules |
| Council / decision deliberation artifacts | `execution/council_deliberation.py` | Decision **artifacts** for multi-agent work; finalization via governance ports |

---

## Identity integration

| ID kind | Producer | Consumer |
| --- | --- | --- |
| `RunId` / `AttemptId` / `ExecutionId` | `execution/identity_authority.py` | Governance reads bound active identity |
| `DecisionId` / evaluation / audit event IDs | Governance contracts (`agent_runtime_governance`, `runtime_policy`) | Execution logs and evidence reference only |
| `PolicySnapshotId` / bundle digest | Policy bundle evaluator | Attached to governance evidence |

Governance modules under `runtime/governance/**` and `runtime/agent_governance/**` **must not** call `mint_run_id`, `mint_attempt_id`, `mint_execution_id`, or authority delegate mint helpers.

---

## Evidence integration

Governance records **evaluation results** and audit envelopes. It does **not**:

- append durable runtime journal events as the evidence plane owner,
- trigger replay reconstruction,
- schedule recovery or retry.

Evidence plane ownership remains NPSC-5F (`RuntimeEventPersistence` and contracts).

---

## Recovery integration

Recovery **re-enters** canonical execution with **existing** identity. It **must not**:

- re-evaluate enterprise governance with execution-local rule engines,
- reset policy bundle state or expand authority scope,
- treat recovery as an implicit `ALLOW`.

Correct pattern:

```text
Recovery Request
        |
        v
Current Governance Evaluation (injected decision / admission port)
        |
        v
Recovery Allowed / Denied  --> ExecutionRuntime re-admission
```

`FanOutPartialRecoveryService` consumes an optional `PolicyDecision` and honors **DENY only**; slot disposition uses `SlotRecoveryPolicyAction` (operational), not a second governance engine.

---

## HITL integration

Human continuation is a **governance outcome**, not an execution override:

- Declarative policy raises `DeclarativePolicyHitlRequiredError` → canonical HITL bridge → `GovernanceResolution`.
- `RuntimeToolInvoker` maps `ToolGovernanceApprovalRequiredError` to pause/escalation — no `bypass_policy` flag.
- Nexus operational `PolicyEnforcer` may escalate to HITL on **retry exhaustion** (resilience), which still routes through governance hooks — not silent allow.

**Residual (documented, non-bypass):** `high_risk_tool_approvals` on `RuntimeState` is a host verification latch for post-invocation HIGH-risk tool verify; production tool path remains gated by `agent_runtime_governance` before invocation.

---

## Certification gates

Frozen architecture tests (must stay green):

| Gate | Path |
| --- | --- |
| NPSC-4 agent governance | `tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py` |
| NPSC-4.2-H1 freeze | `tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py` |
| EE-A1 ownership | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` |
| EE-A2 identity | `tests/unit/runtime/architecture/test_ee_a2_*` |

**Inventory verdict:** `BYPASS = 0`, `DUPLICATE POLICY ENGINE in execution = 0`, `SECOND GOVERNANCE ENGINE in execution runtime = NO`.
