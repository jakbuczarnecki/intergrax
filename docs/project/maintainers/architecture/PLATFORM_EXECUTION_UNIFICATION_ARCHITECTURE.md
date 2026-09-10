# Platform Execution Unification — Architecture (P0 baseline)

**Status:** `U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFIED` (P0 inventory + U1 entry + U2 compensation / tool side-effect closure)  
**Architectural baseline:** NPSC-5E Final `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7` (reference only; work proceeds on current `development`)  
**Evidence companions:** [`../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md) · [`../qualification/PLATFORM_EXECUTION_UNIFICATION_U1_APPLICATION_SCENARIO_ENTRY_QUALIFICATION.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_U1_APPLICATION_SCENARIO_ENTRY_QUALIFICATION.md) · [`../qualification/PLATFORM_EXECUTION_UNIFICATION_U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFICATION.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFICATION.md) · [`../qualification/PLATFORM_EXECUTION_UNIFICATION_U3_AGENT_PLUGIN_EXECUTION_QUALIFICATION.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_U3_AGENT_PLUGIN_EXECUTION_QUALIFICATION.md)

## Purpose

Define what counts as **canonical platform execution**, frozen **ownership**, and **bypass classification** for the Platform Execution Unification program. P0 inventories production-capable paths; U1–U5 close gaps.

## Canonical execution model

Every supported production path that performs platform work or meaningful side effects should traverse:

```text
entrypoint
  → canonical execution entry (host task / scenario host / admitted background handler)
  → ExecutionRuntime / ExecutionBoundary
  → identity (Run / Attempt / Execution where applicable)
  → effective authority
  → governance / policy (where required)
  → budget / deadline (where applicable)
  → lineage (child paths)
  → trace / evidence (where applicable — gaps tracked separately)
  → attempt lifecycle
  → terminal lifecycle
  → Nexus (orchestration strategy only)
  → ChildExecutionPort / ExecutionWorkPort (child work)
  → tool / agent / integration (via RuntimeToolInvoker or governed delegate)
```

**Rule:** reaching a correct component late in the chain is **not** sufficient. The **full admission path** must be canonical.

Subordinate references (not duplicated here):

- [`docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md`](../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md) — semantic authority (UEA)
- [`docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) — ExecutionRuntime ownership

## Frozen ownership (do not re-home in P0)

| Owner | Responsibility |
| --- | --- |
| `ExecutionRuntime` | Root execution lifecycle |
| `AttemptLifecycleService` | Attempt lifecycle |
| `ExecutionLineagePersistence` / active lineage | Lineage |
| Governance ports / policy gates | Whether execution or side effect is permitted |
| Authority plane | Effective authority (identity ≠ authority) |
| `Nexus` / `NexusLoop` | Topology / scheduling (not lifecycle ownership) |
| `ChildExecutionPort` / `ExecutionWorkPort` | Child execution admission |
| R1 / R2 / R3 recovery plane | Frozen retry / checkpoint / partial recovery |
| HITL continuation | Human continuation through same execution admission |

## Production entry surfaces (taxonomy)

Discovered Tier-1 / Tier-3 surfaces inventoried in P0:

| Surface | Typical canonical entry |
| --- | --- |
| Interaction / API task intake | `HostTaskExecutionExecutor` → `HostTaskExecutionPort` → `Execution` facade |
| Scenario / platform proof | `execute_scenario_task` → `build_environment_host_task_execution` |
| Harness HTTP task control | `mount_canonical_harness_task_routes` → host task port |
| Queue / background worker | `execute_logical_task` + admitted `BackgroundExecutionIdentity` |
| CLI `intergrax run` | ASGI host only; execution delegated to composed application |
| Agent step (UAEP) | `RuntimeExecutionContext.invoke_tool` → Nexus tool gateway |
| Orchestration / fan-out | `CoordinationIntentExecutor` / `GraphExecutor` under active root execution |
| Child delegation | `ChildExecutionRunner` only via `ExecutionWorkPort`, `GraphExecutor`, or explicit `ChildExecutionPort` adapter |

## What counts as a bypass

A **bypass** is a production-capable path that performs execution or a meaningful side effect while **skipping** a required boundary above (identity, authority, governance, execution admission, or canonical child port).

Examples proven in P0 inventory:

- Instantiating `ChildExecutionRunner` at application composition default instead of injecting `ExecutionWorkPort` / Nexus-backed child port.
- Background compensation drain invoking tools without `ExecutionRuntime` admission (**closed U2** — see U2 qualification).
- Tier-3 or diagnostic code constructing `UnifiedTaskRunner` or calling providers for mutations outside tool/execution boundaries (legacy / non-production paths listed separately).

## Classification verdicts (exactly one per path)

| Verdict | Meaning |
| --- | --- |
| `CANONICAL` | Evidence shows full canonical admission for this surface |
| `CANONICAL WITH GAP` | Canonical core path; missing trace/evidence or optional governance wiring |
| `LEGACY BUT NON-PRODUCTION` | Code exists; no supported production entry dependency |
| `UNSUPPORTED / DEAD` | No active runtime entry |
| `BYPASS` | Proven skip of required boundary |
| `AMBIGUOUS — REQUIRES OWNER DECISION` | Insufficient contract to classify without owner call |

## Severity (for bypasses and gaps)

| Level | Typical trigger |
| --- | --- |
| P0 | Authority / governance / dangerous side-effect bypass |
| P1 | Lifecycle / identity / lineage / direct child / scheduler bypass |
| P2 | Traceability / observability / non-critical architectural gap |
| P3 | Dead / legacy cleanup |

## Static gates (P0)

Proven invariants are encoded in:

`tests/unit/runtime/architecture/test_platform_execution_unification_p0_bypass_inventory.py`

Gates freeze **import surfaces** and **inventory document presence** — not heuristic grep of legitimate adapters.

## Static gates (U1)

Application and scenario production entry closure (EP-02–EP-05) is encoded in:

`tests/unit/runtime/architecture/test_platform_execution_unification_u1_application_scenario_entry.py`

Complements NPSC-3G factory convergence (`tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py`) and P0 scenario/host-task anchors.

**U1 rule (supported production):** application HTTP / harness task surfaces and scenario tasks must resolve host work through `build_host_task_execution` / `build_environment_host_task_execution` (or harness `build_harness_host_runtime` composition that delegates to the same wiring) — not `UnifiedTaskRunner` or application-local execution runtimes.

## Static gates (U2)

Compensation side-effect admission and worker surface closure:

`tests/unit/runtime/architecture/test_platform_execution_unification_u2_tool_integration_side_effect_closure.py`

## Closure waves (default proposal)

Adjust after inventory review:

| Wave | Focus |
| --- | --- |
| U1 | Application / scenario entry unification (host task only; no runner bypass) |
| U2 | Tool / integration side-effect closure (compensation, declarative out-of-band invokers) |
| U3 | Agent / plugin execution closure |
| U4 | Nexus / child closure (default `ChildExecutionPort` injection) |
| U5 | Final zero-bypass qualification |

P0 does **not** implement U1–U5.
