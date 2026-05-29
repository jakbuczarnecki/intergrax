# Intergrax — Documentation

**Last updated:** 2026-05-27

The `docs/` folder contains **exactly four documents**. Nothing else belongs here.

---

## Documents

| Document | Purpose |
|----------|---------|
| [**intergrax_runtime_architecture.md**](intergrax_runtime_architecture.md) | **Architecture canon** — tiers, Nexus, UAEP §42, contracts, forbidden patterns |
| [**INTERGRAX_IMPLEMENTATION_PLAN.md**](INTERGRAX_IMPLEMENTATION_PLAN.md) | **Implementation map** — phases, status, gaps, priority, readiness (Appendix A), technical debt backlog (Appendix B) |
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Agent workflow** — scaffold → register → run → inspect → evaluate |
| **This file** | Navigation and update rules |

```text
Architecture (what)      →  intergrax_runtime_architecture.md
Implementation (status)  →  INTERGRAX_IMPLEMENTATION_PLAN.md
Agent workflow (how)     →  AGENT_CREATION_GUIDE.md
```

---

## Start here

| I want to… | Read |
|------------|------|
| Understand the platform | Implementation plan §0, then architecture canon §1–§5 |
| See current phase and what's next | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §1–§4 |
| Check readiness for business agents | Implementation plan **Appendix A** |
| Review technical debt before Tier-1 work | Implementation plan **Appendix B** |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Deep-dive UAEP / hooks / governance | Architecture canon §42 |

---

## Current focus (2026-05-27)

| Phase | Status |
|-------|--------|
| Phase J — Unified execution entry | **Done** |
| Phase L — Agent OS certification (deliverables + sign-off) | **Done** |
| Phase K — Problem Radar / Vendor Discovery | **Ready to open** (product decision) |

Gate: `uv run pytest tests/ -m gate -q` (**228 tests**)

---

## Update rules

1. **Architecture** → `intergrax_runtime_architecture.md`, then sync §0 in the implementation plan.
2. **Phase / status / readiness** → `INTERGRAX_IMPLEMENTATION_PLAN.md` only.
3. **Agent workflow** → `AGENT_CREATION_GUIDE.md` only.
4. **Do not add** new markdown files to `docs/` without removing or merging an existing one.

Product-specific roadmaps belong under `agents/<name>/` (e.g. `agents/legal/`), not in `docs/`.
