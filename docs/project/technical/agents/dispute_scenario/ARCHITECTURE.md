# dispute_scenario agent — architecture

**Status:** Scaffold baseline (2026-06-07) — DSW product agent

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
Product architecture: [`applications/dispute_sim_application/docs/ARCHITECTURE.md`](../../applications/dispute_sim_application/docs/ARCHITECTURE.md)

---

## Purpose

Tier-2 **process simulation and correspondence safety** for DSW.

Simulates court-process variants (settlement, injunction, full trial, appeal) with outcome **bands** and explicit assumptions. In `correspondence_review` mode, reviews draft emails, demand letters, and pre-litigation notices for procedural and evidentiary pitfalls.

## Capabilities

- `dispute.scenario`

## Modes

| Mode | Trigger | Output |
|------|---------|--------|
| `process_simulation` | `metadata.process_variants` | `scenario_report.md` |
| `correspondence_review` | draft in `input` | `correspondence_review.md` + `hitl_required` |

## Responsibilities

- Never predict certain win/loss — use favorable / neutral / adverse bands
- Flag: admissions, missed deadlines, tone escalation, unsupported demands
- HITL mandatory before any draft marked for external send (DSW.4)

## Runtime

- Inputs: strategy brief + case RAG context
- Tools: `rag.retrieve`, shadow writes
- CVL L2 critic path for correspondence (planned Phase CRIT-V integration)

## Registration

`capabilities=["dispute.scenario"]` in `dispute_sim_application/manifest.py`
