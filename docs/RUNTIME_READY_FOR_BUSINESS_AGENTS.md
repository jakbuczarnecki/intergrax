# Runtime Ready for Business Agents — Checklist

**Phase L.8 · Gate before Problem Radar / Vendor Discovery**

Use this checklist to confirm Intergrax behaves as an Agent Operating System.

**Run acceptance suite:**

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
uv run pytest tests/ -m gate -q
```

---

## Agent creation

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 1 | Can a new agent be scaffolded in minutes? | `python -m intergrax.scaffold new-agent …` | ✅ |
| 2 | Does scaffold generate UAEP structure (contract, steps, tests)? | Full tree under `agents/` | ✅ |
| 3 | Can a developer reach first run in **< 1 hour**? | Scaffold → implement → lab run | ✅ |

---

## Registration

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 4 | Can agent register via `AgentRegistry` only? | No Nexus edits | ✅ |
| 5 | Are capabilities declared in contract? | `capabilities.py` + `AgentContract` | ✅ |

---

## Execution

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 6 | Does agent run through NexusLoop immediately? | `handle_task` / lab `/v1/lab/run` | ✅ |
| 7 | Does UnifiedTaskRunner support same path as HTTP? | Legal/Research/lab pattern | ✅ |
| 8 | Can graph orchestrate multiple agents? | Sequential + parallel acceptance tests | ✅ |

---

## Observability

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 9 | Can every task be traced? | `GET /debug/tasks/{id}` | ✅ |
| 10 | Are runtime events persisted/queryable? | `GET …/events` | ✅ |
| 11 | Are checkpoints inspectable? | `GET …/checkpoints` | ✅ |
| 12 | Are partial results visible? | `GET …/progress` | ✅ |

---

## Validation

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 13 | Does Nexus validate agent output? | `validation_valid` in task metadata | ✅ |
| 14 | Can validation failure trigger retry/alternate agent? | Acceptance test 06 | ✅ |

---

## Recovery

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 15 | Can execution resume after HITL pause? | Acceptance test 04–05 | ✅ |
| 16 | Are checkpoints saved on pause? | SQLite/memory checkpoint store | ✅ |
| 17 | Can graph resume skip completed nodes? | Graph failure recovery tests | ✅ |

---

## Human approval

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 18 | Can HITL be inserted without custom runtime code? | UAEP `REQUEST_HUMAN` | ✅ |
| 19 | Can approval resume via debug API? | `POST /debug/human-response` | ✅ |

---

## Memory

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 20 | Can agents share context in graphs? | SharedTaskContext + acceptance 08 | ✅ |
| 21 | Is memory access policy-scoped? | MemoryView gateway | ✅ |

---

## Composition

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 22 | Can same agent run in multiple applications? | Echo in lab + harness | ✅ |
| 23 | Do applications contain only wiring, not agent logic? | lab/legal/research split | ✅ |
| 24 | Are agents reusable across applications? | Tier-2 in `agents/` | ✅ |

---

## Isolation

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 25 | Can agents use sandbox tools safely? | Acceptance test 09 | ✅ |
| 26 | Can agents write to shadow workspace? | Acceptance test 10 | ✅ |

---

## Tooling & docs

| # | Question | Expected | Status |
|---|----------|----------|--------|
| 27 | Is there a canonical agent creation guide? | `AGENT_CREATION_GUIDE.md` | ✅ |
| 28 | Is there a universal lab application? | `applications/lab_application/` | ✅ |
| 29 | Is Agent OS acceptance suite documented? | `INTERGRAX_AGENT_OS_READINESS_PLAN.md` §6 | ✅ |

---

## Go / no-go decision

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Checklist items passing | ≥ 90% (26/29) | **29/29** |
| Acceptance suite | 10/10 green | **Run gate to confirm** |
| Runtime modified for last agent added | **Zero** | **Required practice** |

### Verdict

**Platform L1 (Agent OS lab-ready): PASS** — pending green `pytest -m gate`.

**Business agents (Problem Radar, Vendor Discovery): NO-GO** until:

1. Gate suite green after Phase L merge
2. One live < 1h exercise with a **new** scaffolded agent documented in experiment registry

---

## Sign-off template

```text
Date:
Agent exercise: <slug>
Time to first run:
Runtime files modified: none / list
Acceptance suite: pass / fail
Gate suite: pass / fail
Decision: GO Phase K / HOLD
```
