# Intergrax — Development Strategy

**Status:** Canonical (2026-06-01)  
**Audience:** Maintainers, architects, implementation agents, Cursor AI  
**Related:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) · [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) · [docs/README.md](README.md)

This document defines **how** Intergrax is developed and **what** overrides what when goals conflict. It does not duplicate technical contracts (those live in the architecture canon).

---

## Highest priority — strategic goal

Build a **modern, production-grade Harness AI** and **Agent Operating System** aligned with practices used by leading agent platforms (Google ADK-style labs, Anthropic Claude Code, OpenAI Codex, Cursor, Viktor, and comparable Agent Engineering stacks).

This is the **overriding** goal of the project.

All architectural, implementation, and organizational decisions MUST be evaluated against this goal.

If existing architecture does not support it — architecture MAY change.  
If the implementation plan does not support it — the plan MUST change.

---

## Decision hierarchy

| Priority | Source | Rule |
|----------|--------|------|
| **1** | **This document** — strategic goal | Production Harness AI / Agent OS |
| **2** | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) | Architecture canon — living spec, not immutable truth |
| **3** | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) | Implementation map — consequence of architecture; must not force bad architecture |

When priority 1 and priority 2/3 conflict, **update architecture and plan first**, then implement.

---

## Laboratory vs production harness

Intergrax deliberately supports **two modes** on one codebase:

| Mode | Purpose | Primary metric |
|------|---------|----------------|
| **Laboratory** | Fast hypothesis validation, discard failed ideas | Time from idea → first traced run **< 1 hour** |
| **Production harness** | Reliable Agent OS for real business agents at organizational scale | Reference agents + stable provider paths + ops SLOs |

**Laboratory is the adoption phase; production harness is the strategic destination.**

The architecture canon (§2, §50–§51) describes both. Phase **L** certified the OS; phases **Q / Q+ / R** hardened harness semantics; phases **S / T** delivered harness environment and cleanliness; phase **U** (see implementation plan) closes the gap to **production harness** (security, policy wiring, contracts). **Business agents (Phase K)** follow **U Done**.

Intergrax is **not** a finished multi-tenant SaaS today (§4 canon). That remains a **future** evolution (canon §50). Production harness **does** require: certified runtime, product reference agents, skill catalog depth, and selected integration **stable** tiers — not 99× beta breadth alone.

---

## Standard work cycle (every significant task)

Never implement automatically. Follow this order:

```text
ANALIZA
  → OCENA ARCHITEKTURY (zgodność z celem Harness AI)
  → OCENA PLANU WDROŻENIA
  → PROPOZYCJA USPRAWNIEŃ
  → AKTUALIZACJA DOKUMENTACJI (strategia → kanon → plan)
  → IMPLEMENTACJA
  → WERYFIKACJA (gate + getattr audit where harness touched)
  → WNIOSKI
```

Think as a **Harness AI architect** first, then as an engineer.

---

## Critical evaluation obligation

Do **not** assume the architecture or implementation plan is ideal.

Before implementation, always assess:

- Does the solution move Intergrax toward a **production-grade Harness AI**?
- Is it aligned with current **Agent Engineering** practice?
- Is there a better architectural pattern?
- Should architecture or plan change **first**?

Implementation is not the goal. Correct architecture is not the goal.  
**A high-quality Harness AI platform** is the goal. Architecture and implementation are tools.

---

## Document maintenance rules

| Change type | Update |
|-------------|--------|
| Strategic direction | **This file** |
| Tiers, Nexus, UAEP, Harness terms | `intergrax_runtime_architecture.md` (§5.3 Harness terms = single vocabulary source) |
| Phase status, deliverables, gates | `INTERGRAX_IMPLEMENTATION_PLAN.md` |
| Agent author workflow | `AGENT_CREATION_GUIDE.md` |
| Integration / tool / skill catalogs | `INTEGRATIONS.md` / `TOOLS.md` / `SKILLS.md` |

After each merged harness PR: `uv run pytest -m gate -q` green; `python scripts/check_harness_no_getattr.py`; sync plan §0.5 gate count.

---

## Current strategic focus (2026-06-01)

| Milestone | Status |
|-----------|--------|
| Agent OS certification (Phase L) | **Done** — Appendix A 20/20 |
| Harness quality + hardening (Q, Q+) | **Done** — gate **450**; zero grandfathered getattr in harness paths |
| Harness AI alignment MVP (Phase R) | **Done** — Skill Library, context, delegation, policy |
| **Harness environment GA (Phase S)** | **Done** (2026-06-01) — see [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md) |
| Product agents K.1 / K.2 | **Next** — Phase K when product prioritizes |

See [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §4, Phase S, and (when added) `HARNESS_ENVIRONMENT.md`.
