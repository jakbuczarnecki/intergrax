# dispute_strategist agent — architecture

**Status:** Scaffold baseline (2026-06-07) — DSW product agent

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
Product architecture: [`applications/dispute_sim_application/docs/ARCHITECTURE.md`](../../../applications/dispute_sim_application/docs/ARCHITECTURE.md)

---

## Purpose

Tier-2 **litigation strategy** for DSW. Translates the analyst's argument matrix into actionable attack/defense lines, emphasis priorities, negotiation posture, and explicit „do not argue" list.

## Capabilities

- `dispute.strategy`

## Responsibilities

| Output | Content |
|--------|---------|
| `strategy_brief.md` | Primary line, fallback line, emphasis map |
| Risk notes | Arguments that backfire if raised |
| Negotiation band | Settlement leverage estimate (qualitative, not financial advice) |

Inputs: `argument_matrix.json` + optional user constraints (e.g. „preserve relationship").

## Runtime

- Consumes analyst artifacts from same `case_id`
- Polish procedural context in `prompts/system.md` (default jurisdiction PL)
- CVL L1 critic on internal consistency with matrix (DSW.3)

## Registration

`capabilities=["dispute.strategy"]` in `dispute_sim_application/manifest.py`
