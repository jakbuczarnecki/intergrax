# dispute_analyst agent — architecture

**Status:** Scaffold baseline (2026-06-07) — DSW product agent

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
Product architecture: [`applications/dispute_sim_application/docs/ARCHITECTURE.md`](../../applications/dispute_sim_application/docs/ARCHITECTURE.md)

---

## Purpose

Tier-2 **argument analysis** for DSW. Given an ingested case corpus, produces an argument inventory with strength/weakness ratings, evidence gaps, and party position map.

## Capabilities

- `dispute.analyze`

## Responsibilities

| Output | Content |
|--------|---------|
| `argument_matrix.json` | Claims per party, supporting/contradicting evidence refs |
| Gap report | Missing documents, weak chains, statute-of-limitations flags (heuristic) |
| Strength summary | Strong / weak / contested arguments |

All claims must cite RAG chunk ids — no unsupported assertions in production mode.

## Runtime

- Depends on prior `dispute.intake` for same `case_id`
- Tools: `rag.retrieve`, shadow `workspace.write_file`
- CVL L1 critic target (DSW.3) — semantic check on matrix completeness

## Registration

`capabilities=["dispute.analyze"]` in `dispute_sim_application/manifest.py`
