# Intergrax — Development Strategy

**Status:** Canonical (2026-06-06) — default queue: §6.1 maintenance; Band 2ad (FAUDIT-32) **Done**  
**Audience:** Maintainers, architects, implementation agents, Cursor AI  
**Related:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) (hub) · [architecture/](architecture/README.md) · [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) · [plan/phases/](plan/phases/) · [README.md — Documentation index](../README.md#documentation-index)

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
| **2** | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) + [architecture/](architecture/README.md) | Architecture hub + domain canon — living spec, not immutable truth |
| **3** | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) | Implementation map — consequence of architecture; must not force bad architecture |

When priority 1 and priority 2/3 conflict, **update architecture and plan first**, then implement.

---

## Documentation boundary

Canonical **`docs/`** architecture and **`INTERGRAX_IMPLEMENTATION_PLAN.md`** describe the **Intergrax Harness AI / Agent OS platform** — the runtime and infrastructure for launching and governing agent environments.

They **do not** describe:

- the architecture or deployment plan of a **specific Tier-3 business environment** (product host under `applications/<name>/`), or
- the architecture or roadmap of a **specific Tier-2 business agent** (domain capability under `agents/<name>/`).

Each business environment and each business agent carries its own **`ARCHITECTURE.md`**, local **`IMPLEMENTATION_PLAN.md`** (where used), and product roadmap. Platform docs explain composition and wiring; product docs explain domain behavior and go-live.

See also: [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §1.1 · [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §4.0a · [`applications/USAGE.md`](../applications/USAGE.md).

---

## Laboratory vs production harness

Intergrax deliberately supports **two modes** on one codebase:

| Mode | Purpose | Primary metric |
|------|---------|----------------|
| **Laboratory** | Fast hypothesis validation, discard failed ideas | Time from idea → first traced run **< 1 hour** |
| **Production harness** | Reliable Agent OS for real business agents at organizational scale | Reference agents + stable provider paths + ops SLOs |

**Laboratory is the adoption phase; production harness is the strategic destination.**

The architecture canon (§2, §50–§51) describes both. Phase **L** certified the OS; phases **Q / Q+ / R** hardened harness semantics; phases **S / T** delivered harness environment and cleanliness; phase **U** closed the gap to **production harness** baseline (security, policy wiring, contracts). **Phase V** is the default post-U architecture hardening track (capability graph, lifecycle governance, context/prompt/evaluation hardening, metrics, security/cost governance). **Business agents (Phase K)** remain end-of-plan for K.1/K.2; **Local Knowledge Workspace (LKW)** started 2026-06-07 as the first harness-validation product — see [`applications/local_workspace_application/ARCHITECTURE.md`](../applications/local_workspace_application/ARCHITECTURE.md) and plan §6.3a **LKW.***.

Intergrax is **not** a finished multi-tenant SaaS today (§4 canon). That remains a **future** evolution (canon §50). Production harness **does** require: certified runtime, product reference agents, skill catalog depth, and selected integration **stable** tiers — not full-catalog beta breadth alone.

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
| Tiers, Nexus, UAEP, Harness terms | `intergrax_runtime_architecture.md` (hub) + `architecture/` (§5.3 Harness terms = `architecture/PLATFORM_FOUNDATION.md`) |
| Phase status, deliverables, gates | `INTERGRAX_IMPLEMENTATION_PLAN.md` |
| Agent author workflow | `AGENT_CREATION_GUIDE.md` |
| Integration / tool / skill catalogs | `INTEGRATIONS.md` / `TOOLS.md` / `SKILLS.md` |

After each merged harness PR: `uv run pytest -m gate -q` green; `python scripts/check_harness_no_getattr.py`; sync plan §0.5 gate count.

---

## Current strategic focus (2026-06-02)

| Milestone | Status |
|-----------|--------|
| Agent OS certification (Phase L) | **Done** — Appendix A 20/20 |
| Harness quality + hardening (Q, Q+) | **Done** — gate **990** (current); zero grandfathered getattr in harness paths |
| Harness AI alignment MVP (Phase R) | **Done** — Skill Library, context, delegation, policy |
| **Harness environment GA (Phase S)** | **Done** (2026-06-01) — see [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md) |
| **Harness production hardening (Phase U)** | **Done** (2026-06-01) |
| **Harness architecture hardening (Phase V)** | **Done** (2026-06-05) | Phase V + V-REM closed; L3/L4 CI closeout + operational L3 signed off |
| **Orchestration closeout (Phase ORCH)** | **Done** (2026-06-05) | planner/classifier wiring, graph spec plan seed, parallel cap |
| **Harness completion backlog** | **Done** (2026-06-02) | §4.1 — U-Leg, typing/CI, platform skills ([plan §4.1](INTERGRAX_IMPLEMENTATION_PLAN.md#41-harness-completion-backlog-execution-order)) |
| Product agents K.1 / K.2 | **End of plan** — [§6.3](INTERGRAX_IMPLEMENTATION_PLAN.md#63-end-of-plan--deferred-product-work-only) only after explicit product decision |
| New Tier-3 product applications | **End of plan** — same §6.3; lab + reference hosts sufficient for all harness work |

**Default implementation queue:** [plan §6.1](INTERGRAX_IMPLEMENTATION_PLAN.md#61-harness-platform-maintenance-default--band-1) maintenance only. Phase ORCH + GOV-AUDIT **Done**. **Not** business agents or new product apps.

**Governance / policy / observability authoring:** [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) Appendix H; audit §5 + §21 in [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

**Orchestration / graph / delegation authoring:** [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) Appendix I; audit §7–§10 in [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md).

See [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §4.0 (priority ladder) and §6.
