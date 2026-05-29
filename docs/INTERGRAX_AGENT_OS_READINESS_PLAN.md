# Intergrax — Agent OS Readiness Plan (Phase L)

**Status:** Active (2026-05-27)  
**Directive:** Formalize Intergrax as Agent Operating System before any business agents.  
**Canon:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md)

---

## 1. Strategic objective

Intergrax is **not** optimized for SaaS or polished products at this stage.

It is optimized for:

- experimentation speed
- agent creation speed
- runtime stability
- orchestration quality
- observability
- composability
- capability validation

**Phase L answers one question:**

> Can a new agent be created, registered, and executed **without modifying the runtime**?

Expected workflow:

```text
idea → hypothesis → capability → scaffold → implementation → registration
    → execution → trace inspection → evaluation → decision
```

---

## 2. Current readiness assessment

| Maturity level | Description | Status |
|----------------|-------------|--------|
| **L0 — Runtime exists** | NexusLoop, UAEP, graph, memory, HITL, checkpoints | **Done** |
| **L1 — Lab-ready** | Scaffold, lab app, acceptance suite, canonical guide | **In progress → Done (this phase)** |
| **L2 — Business-ready** | Problem Radar, Vendor Discovery, prod integrations | **Blocked until L1 gate passes** |

### Core question (pre-Phase L analysis)

| Question | Answer |
|----------|--------|
| Can UAEP agent register + run without Nexus changes? | **Yes** |
| Is workflow documented and repeatable? | **Yes** (AGENT_CREATION_GUIDE.md) |
| Is there a universal lab environment? | **Yes** (`applications/lab_application/`) |
| Is acceptance formally tested? | **Yes** (`tests/acceptance/agent_os/`) |
| Can new agent be created in < 1 hour? | **Yes** (scaffold + lab run) |

---

## 3. Architectural requirements mapping

| Req | Requirement | Deliverable | Status |
|-----|-------------|-------------|--------|
| R1 | Ready experimentation environment | `lab_application` + debug API | **Done** |
| R2 | Canonical agent creation recipe | `AGENT_CREATION_GUIDE.md` + scaffold | **Done** |
| R3 | Agent focuses on business logic only | UAEP-first scaffold | **Done** |
| R4 | Reuse Tier-0 platform | Documented in guide + §5.2 canon | **Done** |
| R5 | Nexus = OS, Agents = apps, Applications = environments | Tier model enforced | **Done** |

---

## 4. Phase L deliverables

| ID | Deliverable | Location | Status |
|----|-------------|----------|--------|
| L.1 | UAEP-first agent scaffold | `intergrax/scaffold/new_agent.py` | **Done** |
| L.2 | Agent creation guide | `docs/AGENT_CREATION_GUIDE.md` | **Done** |
| L.3 | Lab application (Tier-3) | `applications/lab_application/` | **Done** |
| L.4 | Reference technical agents | `agents/echo/`, `agents/lab/mock_agents.py` | **Done** |
| L.5 | Agent OS acceptance suite | `tests/acceptance/agent_os/` | **Done** |
| L.6 | Runtime independence verification | See §5 below | **Verified** |
| L.7 | Application composition verification | Lab + Legal/Research pattern | **Verified** |
| L.8 | Runtime readiness checklist | `RUNTIME_READY_FOR_BUSINESS_AGENTS.md` | **Done** |

---

## 5. Runtime independence (L.6)

New UAEP agents integrate via:

```text
agents/<name>/ → AgentRegistry.register() → NexusLoop.handle_task()
```

**No changes required** in:

- `NexusLoop`
- `UnifiedTaskRunner`
- `GraphExecutor`
- `TaskLifecycle`
- RuntimeEvent system
- Memory subsystem
- `ToolRuntime`
- Checkpoint subsystem

**Acceptable configuration** (not runtime code changes):

- capability ids in agent contract
- application registry wiring (Tier-3)
- task metadata flags (sandbox, shadow, long-running)

---

## 6. Acceptance criteria (L.5)

| # | Scenario | Test |
|---|----------|------|
| 1 | Single agent execution | `test_acceptance_01_single_agent_execution` |
| 2 | Sequential multi-agent | `test_acceptance_02_sequential_multi_agent` |
| 3 | Parallel multi-agent | `test_acceptance_03_parallel_multi_agent` |
| 4 | Human approval flow | `test_acceptance_04_human_approval_flow` |
| 5 | Checkpoint recovery | `test_acceptance_05_checkpoint_recovery` |
| 6 | Retry flow | `test_acceptance_06_retry_flow` |
| 7 | Partial results | `test_acceptance_07_partial_results` |
| 8 | Memory handoff | `test_acceptance_08_memory_handoff` |
| 9 | Sandbox tool execution | `test_acceptance_09_sandbox_tool_execution` |
| 10 | Shadow workspace | `test_acceptance_10_shadow_workspace` |

Run:

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

All scenarios are also in the `gate` marker set.

---

## 7. Remaining gaps (non-blocking for L1, blocking for L2)

| Gap | Impact | Phase |
|-----|--------|-------|
| UAEP mid-step checkpoint | Long-running step resume | Post-L / §42.9 |
| Full plan/graph snapshot checkpoint | Advanced resume | Post-L |
| Real Slack/Teams webhooks | Prod Organization Worker | Phase K+ |
| Policy engine facade | Unified replay/validation | K.3 |
| Dual AgentDecision cleanup | Contract convergence | K.4 |

These do **not** block agent creation in the lab.

---

## 8. Phase L implementation plan

```text
DONE  L.1  UAEP scaffold
DONE  L.2  AGENT_CREATION_GUIDE.md
DONE  L.3  lab_application
DONE  L.4  lab mock agents + Echo
DONE  L.5  acceptance suite
DONE  L.6–L.8  docs + verification
NEXT  Gate     pytest -m gate green
THEN  Phase K  business agents (only after checklist sign-off)
```

---

## 9. Recommendation: Problem Radar / Vendor Discovery

**Status: NOT READY to start implementation until checklist in `RUNTIME_READY_FOR_BUSINESS_AGENTS.md` is signed off.**

Phase L provides the platform proof. Phase K (K.1 Problem Radar, K.2 Vendor Discovery) should begin only when:

1. Acceptance suite is green
2. A developer completes a **new** scaffolded agent end-to-end in < 1 hour
3. No runtime files were modified during that exercise

---

## 10. Agent creation workflow (canonical)

See [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md).

Quick path:

```bash
python -m intergrax.scaffold new-agent my_agent --capability my_agent.basic
uv run pytest agents/my_agent/tests -q
uv run uvicorn lab_application.host.main:app --port 8090
curl -X POST localhost:8090/v1/lab/run -H "Content-Type: application/json" \
  -d '{"message":"test","capability":"my_agent.basic"}'
```

---

## 11. Documentation map update

| Document | Role |
|----------|------|
| This file | Phase L plan + readiness assessment |
| `RUNTIME_READY_FOR_BUSINESS_AGENTS.md` | Go/no-go checklist |
| `AGENT_CREATION_GUIDE.md` | Canonical agent workflow |
| `INTERGRAX_IMPLEMENTATION_PLAN.md` | Phase tracking (L active, K blocked) |
| `AGENT_CREATION_GUIDE.md` | Single canonical agent workflow |
