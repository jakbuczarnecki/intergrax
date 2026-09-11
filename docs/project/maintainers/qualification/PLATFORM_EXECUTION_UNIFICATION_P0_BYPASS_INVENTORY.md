# Platform Execution Unification — P0 Bypass Inventory

**Status:** `INVENTORY_QUALIFIED` (discovery only — **PRODUCTION CODE CHANGED: NO**)  
**Task:** Platform Execution Unification / P0 — Platform-Wide Execution Bypass Inventory  
**Architecture:** [`../architecture/PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md`](../architecture/PLATFORM_EXECUTION_UNIFICATION_ARCHITECTURE.md)  
**Baseline reference:** NPSC-5E Final `fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7`

## Session gate

| Field | Value |
| --- | --- |
| Branch | `development` |
| START_HEAD / START_ORIGIN | `b5cdef98200667b3b559673e91b7bbdf8ca013b9` |
| Cross-session drift | No tracked modifications to execution seams; unrelated untracked NPSC-5F artifacts left untouched |

## Method

1. Entrypoint discovery via CLI scripts, shared application wiring, scenario baseline, queue worker, and symbol references (`ChildExecutionRunner`, `RuntimeToolInvoker`, `execute_scenario_task`, `execute_logical_task`, `HostTaskExecutionPort`).
2. Call-path proof limited to files needed for each row (no whole-repo file walk).
3. Verdict + severity per architecture doc rules.

## Metrics (summary)

| Metric | Count |
| --- | ---: |
| Total execution-capable entrypoints inventoried | 22 |
| CANONICAL | 17 |
| CANONICAL WITH GAP | 2 |
| LEGACY BUT NON-PRODUCTION | 2 |
| UNSUPPORTED / DEAD | 0 |
| BYPASS | 0 |
| AMBIGUOUS | 1 |
| Supported execution bypasses (production) | 0 |
| Direct child execution bypasses | 0 |
| Direct tool / side-effect bypasses | 0 |
| Governance bypasses (proven) | 0 |
| Authority bypasses (proven) | 0 |
| Nexus scheduling bypasses | 0 |
| Uncontrolled parallel execution paths (execution-relevant) | 0 (bounded fan-out under coordination tests) |
| P0 bypasses | 0 |
| P1 bypasses | 0 |
| P2 gaps | 2 |
| P3 legacy cleanups | 2 |

## Central inventory

| ID | Entry point | Domain | Current path | Canonical boundary reached? | Side effect? | Execution identity? | Authority? | Governance? | Lineage? | Verdict | Severity | Evidence | Proposed owner | Closure task |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EP-01 | Interaction intake | runtime / API | `InteractionIntakeService` → `HostTaskExecutionExecutor` → `HostTaskExecutionPort.execute` → `Execution` / `ExecutionRuntime` | Yes | Agent/tool via runtime | Yes (root) | Yes (active authority) | Policy on tool path | Root | CANONICAL | — | `intergrax/runtime/interactions/task_executor.py`, `host_task.py`, `test_npsc3c_d_*` | ExecutionRuntime | — |
| EP-02 | Scenario task | applications / scenarios | `execute_scenario_task` → `build_environment_host_task_execution` → host task port | Yes | Yes | Yes | Yes | Via Nexus runtime | Root | CANONICAL | — | `scenario_runtime_baseline.py:427-451` | Host task / scenario baseline | — |
| EP-03 | Platform proof scenarios | platform_proofs | `execute_scenario` → `execute_scenario_task` | Yes | Yes | Yes | Yes | Same as EP-02 | Root | CANONICAL | — | `platform_proofs/scenarios/*/application/scenario.py` | Scenario baseline | — |
| EP-04 | Harness HTTP tasks | applications | `mount_canonical_harness_task_routes` → `HostTaskExecutionExecutor` | Yes | Yes | Yes | Yes | Harness wiring | Root | CANONICAL | — | `task_control_wiring.py`, `test_ue_11gp_*` | Applications harness | — |
| EP-05 | Tier-3 app hosts (LKW, legal, contractor, …) | applications | App factory → `build_host_task_execution` / environment host execution (no `UnifiedTaskRunner` in app tree) | Yes | Yes | Yes | Yes | App composition | Root | CANONICAL | — | `test_lkw_canonical_execution.py`, `test_governed_contractor_canonical_execution.py` | Application host factories | — |
| EP-06 | Queue / background worker | queueing | Broker → `execute_logical_task` → registry handler (identity admission in bootstrap) | Yes (handler-dependent) | Handler-dependent | `BackgroundExecutionIdentity` | Admitted handler | Handler-dependent | When handler spawns child | CANONICAL | — | `queueing/worker/execution.py`, `test_diag_foundation_4_*` | Background execution bootstrap | — |
| EP-07 | Background task runtime | background_tasks | `worker_runtime` → `execute_logical_task` | Same as EP-06 | Same | Same | Same | Same | Same | CANONICAL | — | `background_tasks/worker_runtime.py` | Queue worker | — |
| EP-08 | CLI `intergrax run` | CLI | `uvicorn.run(module:app)` — no direct execution | Delegated to app | Via app | Via app | Via app | Via app | Via app | CANONICAL | — | `intergrax/cli/run.py` | Tier-3 host | — |
| EP-09 | Nexus agentic task | runtime / nexus | `NexusLoop.handle_task` → agent engine / tool invoker inside active execution | Yes | Yes | Yes | Yes | `RuntimeToolInvoker` + policy | Child via ports | CANONICAL | — | `NEXUS_EXECUTION_FLOW.md`, runtime_context `RuntimeToolInvoker` | Nexus + ExecutionRuntime | — |
| EP-10 | Graph orchestration | nexus | `GraphExecutor` → `ChildExecutionRunner` + declarative invoker | Yes | Yes | Yes | Child authority policy | Policy on tools | Child lineage | CANONICAL | — | `graph_executor.py:118,254` | GraphExecutor | — |
| EP-11 | Execution work port | runtime | `ExecutionWorkPort` → `ChildExecutionRunner` | Yes | Yes | Yes | Yes | Admission hooks | Child | CANONICAL | — | `execution_work_port.py` | ExecutionWorkPort | — |
| EP-12 | Coordination / fan-out | agent_distribution | `CoordinationIntentExecutor` → `MultiAgentCoordinationService` / `BoundedMultiAgentFanOutService` under `peek_governed_execution_task` | Yes | Delegated child | Under root | Governed coordination port | Multi-agent governance | Fan-out lineage | CANONICAL | — | `coordination_intent_executor.py`, `test_npsc5a_*`, `test_npsc5c_*` | Agent distribution | — |
| EP-13 | UAEP agent step | agents | `RuntimeExecutionContext.invoke_tool` → catalog gateway → `RuntimeToolInvoker` | Yes when inside Nexus step | Yes | Active execution | Active authority | Agent governance mandatory in `production_mode` | Under parent | **CANONICAL** | — | `runtime_tool_helpers.py`, `runtime_context.py`, `invoker.py`, U3 qualification | Agent governance wiring | **U3 closed** |
| EP-14 | ACP declarative catalog tools | applications / agents | `build_declarative_invoker_from_tool_wiring` → `RuntimeToolInvoker` (no governance param in wiring) | Partial | Yes | When bound to active run | Side-effect auth in invoker | Optional agent governance | When in run | CANONICAL WITH GAP | P2 | `declarative_tool_wiring.py:33-38` | Applications tool wiring | U2 |
| EP-15 | Delegated subtask factory | applications / agent_distribution | `build_production_delegated_subtask_child_execution_port` → `delegated_subtask_child_execution_work_port` → `DelegatedSubtaskServiceFactory.create(child_execution=...)` → `ChildExecutionPort` → `DelegatedSubtaskChildExecutionWorkPort` → `ChildExecutionRunner` | Yes — composition-root work port + domain adapter | Child agent work | Child IDs minted in runner | Parent context required | Physical delegation governance + child admission | Child lineage in runner | **CANONICAL** | — | `production_delegated_subtask_child_execution_wiring.py`, U4 qualification | Applications composition | **U4 closed** |
| EP-16 | Compensation queue drain | agents / persistence | `drain_pending_compensation_jobs` → `CompensationSideEffectExecutionPort` → `Execution` / `ExecutionRuntime` → `CompensationToolInvokeSession` → `RuntimeToolInvoker` | Yes | Yes (compensation tools) | Yes (root admission; preserves job `run_id`) | Active authority enforced in delegate | Decision lifecycle host + tool policy on invoke | Root segment for job run | **CANONICAL** | — | `compensation_queue_worker.py`, `runtime/execution/compensation_side_effect.py` | Runtime compensation admission | **U2 closed** |
| EP-17 | Work-stage capability loop | autonomous_work / catalog | `WorkStageCapabilityLoop` → `WorkStageToolExecutionPort.execute` | **Port-defined** | Tool when TOOL kind | Loop binds minted ids locally | Depends on port impl | Catalog governance context only | Correlation fields | **AMBIGUOUS — REQUIRES OWNER DECISION** | P1 | `work_stage_capability_loop.py:290-296`, identity mint in module | Capability catalog + AW | Owner decision |
| EP-18 | HITL governed continuation | runtime | Governed continuation tests / `test_npsc5d_r3_*` production wiring | Yes (qualified) | Resumes canonical execution | Yes | Yes | HITL governance | Resume lineage | CANONICAL | — | `test_npsc5d_r3_governed_continuation.py` | HITL / recovery | — |
| EP-19 | Recovery retry / resume / partial | runtime | R1 `ExecutionAttemptRetryService`, R2 checkpoint resume, R3 fan-out partial recovery | Returns to canonical execution | No new bypass plane | Yes | Authority gates in R2 | Policy in recovery contracts | Yes | CANONICAL | — | NPSC-5E finals | Frozen recovery plane | — |
| EP-20 | Experiments workflow | experiments | `UnifiedTaskRunner(loop).run_task` direct | **No** host task / ExecutionRuntime facade | Yes | Partial | Unclear | Unclear | Unclear | LEGACY BUT NON-PRODUCTION | P3 | `intergrax/experiments/workflow.py:153` | Experiments | Remove or gate |
| EP-21 | Nexus eval runner | eval | Constructs `UnifiedTaskRunner` for eval | Non-production eval | Yes | Eval scope | Eval scope | Limited | Limited | LEGACY BUT NON-PRODUCTION | P3 | `eval/nexus_eval_runner.py` | Eval | Document non-prod |
| EP-22 | Integration providers in app wiring | integrations | `integration_tool_wiring` resolves providers into tool context (not direct mutation from app code) | Yes (via tools) | Via tools | Via execution | Via tool auth | Tool policy | N/A | CANONICAL WITH GAP | P2 | `integration_tool_wiring.py` | Tools / integrations | U2 monitoring |

## Bypass graph (proven)

### BY-01 (P1) — closed in U4

```text
build_production_delegated_subtask_child_execution_port
  → ProductionAgentCapabilityRuntime.delegated_subtask_child_execution
  → DelegatedSubtaskServiceFactory.create(child_execution=...)
  → DelegatedSubtaskService.execute
  → specialist child work
```

See [`PLATFORM_EXECUTION_UNIFICATION_U4_CHILD_EXECUTION_CLOSURE.md`](PLATFORM_EXECUTION_UNIFICATION_U4_CHILD_EXECUTION_CLOSURE.md).

### BY-02 (P0) — closed in U2

Compensation queue drain no longer invokes tools directly. See [`PLATFORM_EXECUTION_UNIFICATION_U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFICATION.md`](PLATFORM_EXECUTION_UNIFICATION_U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFICATION.md).

## Ambiguous execution paths requiring owner decision

Not counted as proven bypass in Metrics or supported production bypass totals.

### EP-17 — work-stage tool port

| Field | Value |
| --- | --- |
| ID | EP-17 |
| VERDICT | AMBIGUOUS — REQUIRES OWNER DECISION |
| NOT COUNTED AS PROVEN BYPASS | Yes |

```text
WorkStageCapabilityLoop.run
  → WorkStageToolExecutionPort.execute
  → (no proven production implementation path that skips RuntimeToolInvoker / execution boundary)
```

Owner decision required before classification as bypass or canonical (see central inventory row EP-17).

## Direct `ChildExecutionRunner` import surface (production)

Frozen allowlist (`intergrax/` only):

| Module | Role |
| --- | --- |
| `runtime/execution/child.py` | Owner (defines `ChildExecutionRunner`; not an importer) |
| `runtime/execution/delegated_subtask_child_port.py` | Canonical adapter |
| `runtime/execution/execution_work_port.py` | Canonical adapter |
| `runtime/nexus/execution/graph_executor.py` | Nexus orchestration |
| `applications/_shared/production_delegated_subtask_child_execution_wiring.py` | Composition-root delegated subtask child port (U4) |

Static gate: `test_platform_execution_unification_p0_bypass_inventory.py`.

## Parallel execution

| Location | Classification |
| --- | --- |
| `BoundedMultiAgentFanOutService` / coordination | Bounded / canonical (NPSC-5B qualified) |
| `RuntimeToolInvoker` thread pool for timeouts | Bounded / not separate execution admission |
| Ad-hoc `asyncio.create_task` in product paths reviewed via coordination gates | No additional uncontrolled execution entrypoints proven in P0 |

## Closure ownership

| Wave | Scope (from inventory) |
| --- | --- |
| **U1** | ✅ Closed — EP-02–EP-05 qualified; see `PLATFORM_EXECUTION_UNIFICATION_U1_APPLICATION_SCENARIO_ENTRY_QUALIFICATION.md` |
| **U2** | ✅ EP-16 / BY-02 closed; EP-14 qualified; EP-22 qualified — see U2 qualification artifact |
| **U3** | EP-13 agent governance defaults |
| **U4** | ✅ EP-15 / BY-01 closed — see U4 qualification artifact |
| **U5** | Re-run inventory with zero production bypasses; extend static gates |

## Regression notes (P0 commit)

Frozen suites invoked for this qualification:

- NPSC-5E Final — `test_npsc5e_final_recovery_plane_qualification_and_freeze.py`
- NPSC-5D Final — `test_npsc5d_final_multi_agent_governance_qualification.py`
- NPSC-5B Final — `test_npsc5b_final_production_fanout_fanin_qualification.py`
- DG_001 — `tests/unit/contracts/test_execution_lineage_contracts.py` + `tests/unit/runtime/execution/lineage/`
- P0 gate — `test_platform_execution_unification_p0_bypass_inventory.py`
