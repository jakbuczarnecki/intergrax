# Intergrax — Runtime Implementation Plan



Status: Working draft (2026-05-27, synced post A.5-min gate)  

Canonical source: [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  

Baseline: [`INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md`](INTERGRAX_IMPLEMENTATION_GAP_ANALYSIS.md) §14–§16  

Documentation map: [`README.md`](README.md)  

Principle: **evolve, not rewrite** · **reuse Tier-0** (§5.2)



---



## 1. Plan Objective



Transform Intergrax into an **internal agent experimentation laboratory** (§2, §35) aligned with the canonical architecture:



```text

hypothesis → capability → contract → registration → Nexus → trace → evaluation → decision

```



**Success metric:** time from idea to first running experiment **< 1 hour**.



**Current alignment:**



| Scope | Score | Notes |

|-------|-------|-------|

| Architecture §1–41 (tiers, Nexus, graph, repo split) | **~82–88%** | Phases A–C complete |

| §42 Unified Execution Runtime | **~45–50%** | P4.1–P4.3 wired; tool gateway + agent migration pending |

| Laboratory workflow (inspect, decide) | **~55%** | D.1 debug CLI done; experiment registry pending |

| Pre-P4.2 regression gate | **Done** | A.5-min (~10 tests, marker `gate`) |



---



## 2. Map: Architecture → Implementation Status



| Section | Requirement | Status | Location |

|---------|-------------|--------|----------|

| §5.1 Four tiers | Tier-0..3 model | **Done** | architecture doc + `agent_kit/tiers.py` |

| §5.2 Reuse Tier-0 | No redundant platform | **Doc + process** | §5.2, §8.8, §39.8 |

| §9.1 Nexus Loop | Global orchestration | **Done** | `nexus_loop.py` |

| §9.2 Local agent loop | Bounded steps | **Partial** | Legal pipeline; not UAEP |

| §12–16 Contracts / Registry | AgentContract, capabilities | **Done** | `intergrax/contracts/`, `runtime/registry/` |

| §22 ToolRuntime | Policy gateway | **Done** | `tool_runtime.py`, `ToolAccessPolicy` |

| §23 Task lifecycle | States + trace | **Done** | `task/`, `TaskTraceEmitter` |

| §24–25 Execution graph | Multi-agent | **Done** | `execution/`, `GraphExecutor` |

| §29 Validation | Nexus + agent | **Done** | `NexusValidationEngine` |

| §31 Retry | Runtime-managed | **Done** | `RetryEngine` |

| §33 Observability | Trace + events | **Partial** | Trace store ✅; P4.1 dual-emit ✅; D.1 CLI ✅ |

| §42 Execution runtime | UAEP, hooks, governance | **Partial** | P4.2 UAEP ✅; P4.3 governance ✅; P4.4+ pending |

| §7.4 Repo split | agents / applications | **Done** | `agents/legal`, `applications/legal_application` (no `legal_agent` shim) |

| §19 Debug surface | CLI / API | **Partial** | D.1 CLI ✅; D.2 API pending |

| §32 HITL | Approval flow | **Not started** | Phase F |

| §20–21 Shadow / Sandbox | Isolated exec | **Not started** | Phase F |



---



## 3. Implementation Phases



### Phase A — Foundation Stabilization



| # | Deliverable | Status |

|---|-------------|--------|

| A.1 | Unified run lifecycle | **Done** |

| A.2 | Task trace persistence | **Done** |

| A.3 | NexusLoop production path | **Done** |

| A.4 | EvalRunner integration | **Done** |

| A.5-min | Pre-P4.2 regression gate | **Done** |

| A.5 | Full regression suite (Legal E2E, all steps) | **Deferred** |

| A.6 | Shim cleanup | **Done** | Removed `applications/legal_agent/`; docs + duplicate `legal_application/tests/` cleaned |



**A.5-min completion criteria (gate before P4.2):**



```bash

uv run pytest tests/ -m gate -q

```



| Test area | File |

|-----------|------|

| TaskLifecycle transitions | `tests/unit/runtime/task/test_task_lifecycle.py` |

| TaskTraceEmitter + RuntimeEventBus | `tests/unit/runtime/task/test_task_trace_event_bus.py` |

| trace_bridge mapping | `tests/unit/runtime/events/test_trace_bridge.py` |

| AgentEngine.run / run_with_result | `tests/integration/agents/test_agent_engine_*.py` |

| NexusLoop + Echo (lifecycle + events) | `tests/integration/runtime/test_nexus_loop_echo.py` |

| GraphExecutor sequential stub | `tests/integration/runtime/test_graph_executor_stub.py` |



**Infrastructure fixes included:** circular import (`tool_runtime` ↔ `runtime_state`), missing `RegistryToolExecutor`, `ExecutionGraph` pydantic imports, lazy pipeline imports in `tests/conftest.py`.



**Explicitly not required before P4.2:** Legal through NexusLoop, full Nexus step matrix, E2E with real LLM.



---



### Phase B — Extended Nexus



| # | Deliverable | Status |

|---|-------------|--------|

| B.1–B.7 | Classifier, planner, validation, retry, tool policy, composer | **Done** |



---



### Phase C — Multi-Agent Readiness



| # | Deliverable | Status |

|---|-------------|--------|

| C.1–C.6 | ExecutionGraph, GraphExecutor, ContextManager, Research pipeline | **Done** |



---



### Phase D — Observability and Experiments



**Goal:** §19, §35 — laboratory tooling (not SaaS UI).



| # | Deliverable | Status | Notes |

|---|-------------|--------|-------|

| D.0 | §42 P4.1 Event Bus wiring | **Done** | `RuntimeEventBus`, `trace_bridge`, NexusLoop |

| D.1 | Debug CLI | **Done** | `python -m intergrax.debug tasks list\|show\|trace` |

| D.2 | Minimal debug API | Pending | FastAPI endpoints on trace store |

| D.3 | Experiment registry | Pending | hypothesis, keep/improve/pause/delete |

| D.4 | Notebook templates | Pending | `notebooks/experiments/` |

| D.5 | Cost in trace | Pending | `AgentExecutionResult.cost` from runtime stats |



---



### Phase E — Legal Agent Refactoring (parallel)



| # | Deliverable | Status |

|---|-------------|--------|

| E.1–E.4 | Thin Legal, ToolRuntime migration, governance, loop bounds | **Not started** |



---



### Phase F — Advanced / On-Demand



Long-running tasks, HITL, Shadow, Sandbox, Slack/Teams — **only with concrete use case**.



---



### Phase P4 — §42 Unified Execution Runtime



| Step | Deliverable | Status |

|------|-------------|--------|

| P4.1 | Event bus + trace bridge | **Done** |

| P4.2 | UAEP in AgentEngine | **Done** |

| P4.3 | Governance (interrupt, HITL) | **Done** |

| P4.4 | Tool gateway unification | **Next** |

| P4.5 | Agent migration (Echo, Research, Legal) | Pending |



**P4.3 delivered (2026-05-27):** `runtime/interrupts/`, `runtime/human/`, `RuntimePolicyEngine` wired in UAEP + NexusLoop; `WAITING_FOR_HUMAN` pause/resume via `human_approved` metadata. Gate: 21 tests.



**P4.2 delivered (2026-05-27):** `intergrax/agents/uaep.py`, `UAEPExecutor`, Echo `get_steps`/`run_step`, legacy fallback preserved.



---



## 4. Priority Order



```text

NOW:     P4.4 tool gateway · D.2 debug API

NEXT:    P4.5 Research/Legal UAEP migration

THEN:    Phase E (Legal thin) · D.3 experiment registry

LATER:   Phase F (on demand)

```



---



## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`experiment_guide.md`](experiment_guide.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)



---



## 6. Recommended Next Step

**Phase P4.4** — tool gateway unification (`ToolRequest`/`ToolResponse`, Legal bridge).

```bash
uv run pytest tests/ -m gate -q
```

**P4.3 (Done):** `ExecutionInterruptHandler`, `HumanPauseCoordinator`, policy in UAEP, NexusLoop `WAITING_FOR_HUMAN` + resume.

**P4.2 (Done):** `UAEPExecutor` in `intergrax/agents/uaep.py`; `AgentEngine` UAEP + legacy fallback; Echo reference agent.



**Then:** P4.5 agent migration or D.2 debug API.



### D.1 Debug CLI (Done)



```bash

python -m intergrax.debug tasks list --tenant t1 --limit 20

python -m intergrax.debug tasks show RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1 --format json --runtime

python -m intergrax.debug --db path/to/trace.db tasks list

```



Reuse:



- `SQLiteRunTraceStore` / `RunTraceReader` — `intergrax/runtime/nexus/tracing/`

- `trace_bridge` — `intergrax/runtime/events/trace_bridge.py`

- `NexusLoop.event_bus` — in-process runs (not persisted; CLI uses SQLite trace)



---



*Plan synced with codebase after P4.3 governance (2026-05-27). Gate: 21 tests.*

