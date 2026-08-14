# dispute_intake agent — architecture

**Status:** Scaffold baseline (2026-06-07) — DSW product agent

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
Product architecture: [`applications/dispute_sim_application/docs/ARCHITECTURE.md`](../../applications/dispute_sim_application/docs/ARCHITECTURE.md)

---

## Purpose

Tier-2 **case material intake** for Dispute Simulation Workspace (DSW).

When a legal ops user delivers dispute materials, this agent validates sources, classifies document types, builds a chronology, and ingests content into a **case-scoped RAG collection**.

## Capabilities

- `dispute.intake`

## Responsibilities

| Step (target) | Output |
|---------------|--------|
| Validate `case_id` + `source_paths` | Error or accepted batch |
| Classify documents | `contract`, `email`, `invoice`, … |
| Build timeline | `case_timeline.json` (shadow) |
| Ingest to RAG | Collection `dispute:{case_id}` |

## Layout

| Path | Role |
|------|------|
| `dispute_intake_agent.py` | UAEP entry |
| `contract.py` | `AgentContract` |
| ``on_next_step` / cognitive pattern hooks` | Domain execution (stub → DSW.1) |
| `schemas` | Case intake I/O models (DSW.1) |

## Runtime

- `Agent` + UAEP pipeline steps
- Tools via Tier-3 `dispute_sim_application` tool profile (`rag.*`, `workspace.*`)
- **No** `applications` imports

## Registration

`AgentBinding.mount(DisputeIntakeAgent, capabilities=["dispute.intake"], default=True)` in `dispute_sim_application/manifest.py`
