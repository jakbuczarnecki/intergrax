# dispute_strategist agent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: Working draft (2026-06-07) — **Scaffold baseline**

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../architecture/intergrax_runtime_architecture.md)
Agent workflow: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../guides/AGENT_CREATION_GUIDE.md)

Principle: **evolve, not rewrite** · **reuse Tier-0** · **no Tier-3 imports in agent code**

---

## Documentation model

Do not maintain separate status/readiness files under this agent. Use:

| Topic | Where |
|-------|--------|
| Purpose, contracts, I/O, runtime layout | **ARCHITECTURE.md** (this directory) |
| Task status, phases, next steps | **This file** |
| Significant agent architecture decisions | **`adr`** — [`adr/README.md`](adr/README.md) |
| Platform harness work | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` (gate maintenance) |
| UAEP / Nexus workflow | `docs/project/technical/guides/AGENT_CREATION_GUIDE.md` |

---

## 0. Scope at a glance

| Field | Value |
|-------|-------|
| Agent id | `dispute_strategist` |
| Class | `DisputeStrategistAgent` |
| Primary capability | `dispute.strategy` |
| Tier | Tier-2 (`agents/dispute_strategist`) |
| Host wiring | Tier-3 application manifest (when mounted) |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| DISPUTE_STRATEGIST-1 | Replace scaffold stub in ``on_next_step` / cognitive pattern hooks` | Planned | High | One PR per domain step |
| DISPUTE_STRATEGIST-2 | Extend `prompts/system.md` for domain | Planned | Medium | Keep prompts versioned here |
| DISPUTE_STRATEGIST-3 | Register skills/tools on `contract.py` | Planned | Medium | See `docs/project/architecture/SKILLS.md` |
| DISPUTE_STRATEGIST-4 | Agent smoke test green | Done | High | `tests/test_dispute_strategist_agent.py` |
| DISPUTE_STRATEGIST-5 | Mount in Tier-3 host (optional) | Planned | Medium | `AgentBinding.mount(DisputeStrategistAgent, ...)` |

---

## 2. Verification

```bash
uv run pytest agents/dispute_strategist/tests -q
```

After host wiring:

```bash
uv run pytest applications/<app>_application/tests -q
```


---

## 3. Platform alignment

Business agents and product-only work remain **end of plan** unless explicitly reprioritized —
see platform [`§6.3`](../../../maintainers/plans/PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).
