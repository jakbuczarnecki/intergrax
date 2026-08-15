# Dispute Simulation Workspace (DSW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15 (do not diverge without updating architecture first)

Status: **DSW.0 Done** · **Active queue: DSW.1**

Platform register: [`docs/project/architecture/intergrax_runtime_architecture.md` §6.3a DSW.*](../../../docs/project/architecture/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)

Principle: **case-scoped RAG** · **shadow drafts** · **HITL on correspondence** · **simulation ≠ legal advice**

---

## 0. Product boundary reminder

| | Backend (`dispute_sim_application`) | Client |
|---|-------------------------------------|--------|
| **Runs** | Org-hosted API / MCP | Legal ops UI, Cursor, scripts |
| **Contains** | Nexus, 4 agents, case index | Task submit + artifact download only |
| **Must not** | Send mail / file court docs autonomously | Bypass HITL on outbound drafts |

See architecture §3–§4.

---

## 1. Wave queue

| ID | Title | Depends | Status | Priority |
|----|-------|---------|--------|----------|
| DSW.0 | Scaffold + architecture v1 + ADR-001 | — | **Done** | — |
| DSW.1 | Intake UAEP: validate paths + `rag.ingest_document` per case | DSW.0 | **Active** | Critical |
| DSW.2 | Graph `dispute.pipeline` (intake→analyze→strategy→scenario) | DSW.1 | Planned | High |
| DSW.3 | Analyst matrix + strategist brief steps | DSW.1 | Planned | High |
| DSW.4 | Scenario variants + correspondence review + HITL hook | DSW.3 | Planned | High |
| DSW.5 | Optional `legal.review` subgraph for clause drill-down | DSW.3 | Planned | Medium |
| DSW.6 | Case persistence model + retention in host settings | DSW.1 | Planned | Medium |
| DSW.7 | Polish dispute eval fixtures + regression | DSW.4 | Planned | Medium |

---

## 2. Active wave — DSW.1

### Tasks

| ID | Task | Module | Owner |
|----|------|--------|-------|
| DSW.1.1 | Intake step: `case_id` + `source_paths` validation | `agents/dispute_intake`on_next_step` / cognitive pattern hooks` | Tier-2 |
| DSW.1.2 | Document type classifier stub (rule + LLM) | `agents/dispute_intake/steps` | Tier-2 |
| DSW.1.3 | Chronology builder → `case_timeline.json` shadow artifact | `agents/dispute_intake/steps` | Tier-2 |
| DSW.1.4 | RAG ingest loop scoped by `case_id` collection | Tier-3 tool profile + intake agent | Tier-2/3 |
| DSW.1.5 | Acceptance: fixture corpus → timeline + retrievable chunks | `tests` | Tier-3 |

### Acceptance criteria

- [ ] `POST /v1/dispute_sim/run` with `capability=dispute.intake` + `metadata.case_id` + `source_paths` completes
- [ ] Follow-up `dispute.analyze` retrieves ingested content (stub OK in DSW.1)
- [ ] Shadow artifact written under case namespace
- [ ] Agent + host smoke tests green
- [ ] Disclaimer string present in response metadata (Tier-3 router)

---

## 3. Agent-level queues

| Agent | Plan file |
|-------|-----------|
| `dispute_intake` | [`agents/dispute_intake/docs/IMPLEMENTATION_PLAN.md`](../../../agents/dispute_intake/docs/IMPLEMENTATION_PLAN.md) |
| `dispute_analyst` | [`agents/dispute_analyst/docs/IMPLEMENTATION_PLAN.md`](../../../agents/dispute_analyst/docs/IMPLEMENTATION_PLAN.md) |
| `dispute_strategist` | [`agents/dispute_strategist/docs/IMPLEMENTATION_PLAN.md`](../../../agents/dispute_strategist/docs/IMPLEMENTATION_PLAN.md) |
| `dispute_scenario` | [`agents/dispute_scenario/docs/IMPLEMENTATION_PLAN.md`](../../../agents/dispute_scenario/docs/IMPLEMENTATION_PLAN.md) |

---

## 4. Verification

```bash
uv run pytest agents/dispute_intake/tests agents/dispute_analyst/tests agents/dispute_strategist/tests agents/dispute_scenario/tests -q
uv run pytest applications/dispute_sim_application/tests -q
```
