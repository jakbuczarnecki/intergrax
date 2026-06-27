# Intergrax — Development Strategy

**Status:** Canonical (2026-06-06) — default queue: §6.1 maintenance; Band 2ad (FAUDIT-32) **Done**  
**Audience:** Maintainers, architects, implementation agents, Cursor AI  
**Related:** [intergrax_runtime_architecture.md](../intergrax_runtime_architecture.md) (hub) · [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) (cross-domain never-violate index) · [LAYER_COMPLETION_MODE.md](LAYER_COMPLETION_MODE.md) (deep layer closeout) · [architecture/](../architecture/) · [plan/](../plan/) · [features/](../features/README.md) · [README.md — Documentation index](../../README.md#documentation-index)

This document defines **how** Intergrax is developed and **what** overrides what when goals conflict. It does not duplicate technical contracts (those live in the architecture canon and multi-layer feature docs).

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
| **1b** | [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) | Cross-domain architectural rules — must not contradict domain canon |
| **2** | [intergrax_runtime_architecture.md](../intergrax_runtime_architecture.md) + [architecture/](../architecture/) + [features/](../features/README.md) | Architecture hub + domain canon + multi-layer feature canon — living spec, not immutable truth |
| **3** | [plan/](../plan/) + [features/plan/](../features/plan/) | Implementation map — consequence of architecture; must not force bad architecture |

When priority 1 and priority 2/3 conflict, **update architecture and plan first**, then implement.

---

## Documentation boundary

Canonical **`docs/`** architecture and **`intergrax_runtime_architecture.md`** describe the **Intergrax Harness AI / Agent OS platform** — the runtime and infrastructure for launching and governing agent environments.

They are split into two paired documentation structures:

```text
docs/architecture/<DOMAIN>.md           ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

Domain pairs describe a single architecture layer and its implementation plan. Feature pairs describe cross-layer capabilities that require coordinated implementation across multiple domain pairs. Feature docs coordinate; domain docs remain implementation ownership truth.

They **do not** describe:

- the architecture or deployment plan of a **specific Tier-3 business environment** (product host under `applications/<name>/`), or
- the architecture or roadmap of a **specific Tier-2 business agent** (domain capability under `agents/<name>/`).

Each business environment and each business agent carries its own **`ARCHITECTURE.md`**, local **`IMPLEMENTATION_PLAN.md`** (where used), and product roadmap. Platform docs explain composition and wiring; product docs explain domain behavior and go-live.

See also: [intergrax_runtime_architecture.md](../intergrax_runtime_architecture.md) §Documentation topology · [`applications/USAGE.md`](../../applications/USAGE.md).

---

## Laboratory vs production harness

Intergrax deliberately supports **two modes** on one codebase:

| Mode | Purpose | Primary metric |
|------|---------|----------------|
| **Laboratory** | Fast hypothesis validation, discard failed ideas | Time from idea → first traced run **< 1 hour** |
| **Production harness** | Reliable Agent OS for real business agents at organizational scale | Reference agents + stable provider paths + ops SLOs |

**Laboratory is the adoption phase; production harness is the strategic destination.**

The architecture canon (§2, §50–§51) describes both. Phase **L** certified the OS; phases **Q / Q+ / R** hardened harness semantics; phases **S / T** delivered harness environment and cleanliness; phase **U** closed the gap to **production harness** baseline (security, policy wiring, contracts). **Phase V** is the default post-U architecture hardening track (capability graph, lifecycle governance, context/prompt/evaluation hardening, metrics, security/cost governance). **Business agents (Phase K)** remain end-of-plan for K.1/K.2; **Local Knowledge Workspace (LKW)** started 2026-06-07 as the first harness-validation product — see [`applications/local_workspace_application/docs/ARCHITECTURE.md`](../../applications/local_workspace_application/docs/ARCHITECTURE.md) and plan §6.3a **LKW.***.

Intergrax is **not** a finished multi-tenant SaaS today (§4 canon). That remains a **future** evolution (canon §50). Production harness **does** require: certified runtime, product reference agents, skill catalog depth, and selected integration **stable** tiers — not full-catalog beta breadth alone.

---

## Standard work cycle (every significant task)

Never implement automatically. Follow this order:

```text
ANALYSIS
  → ARCHITECTURE REVIEW (Harness AI goal alignment)
  → PLAN REVIEW
  → IMPROVEMENT PROPOSAL
  → DOCUMENTATION UPDATE (strategy → canon/feature → plan)
  → IMPLEMENTATION
  → VERIFICATION (gate + getattr audit where harness touched)
  → CONCLUSIONS (+ implementation journal entry when a deliverable closed)
```

**CONCLUSIONS:** For completed implementations, record an English episode in [`../implementation-journal/`](../implementation-journal/README.md) (prepend to `INDEX.md`; operator intent, `plan_ref`, impact) — narrative layer only; plan rows remain the status source of truth.

Think as a **Harness AI architect** first, then as an engineer.

**Deep layer closeout** (audit → maturity, multiple sprints on one domain pair): follow [LAYER_COMPLETION_MODE.md](LAYER_COMPLETION_MODE.md). Default Cursor iterations remain single plan items per [`.cursor/rules/intergrax-iteration.mdc`](../../.cursor/rules/intergrax-iteration.mdc). Multi-layer features follow their feature pair, then implement the smallest domain-owned plan item.

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
| Domain layer architecture or implementation plan | `architecture/<DOMAIN>.md` + `plan/<DOMAIN>.md` |
| Cross-layer feature architecture or implementation program | `features/architecture/<FEATURE>.md` + `features/plan/<FEATURE>.md`, then affected domain pairs |
| Phase status, deliverables, gates | owning domain plan; feature plan only for cross-layer coordination |
| Agent author workflow | `guides/AGENT_CREATION_GUIDE.md` |
| Integration / RAG / tool / skill catalogs | `architecture/INTEGRATIONS.md` / `architecture/RAG.md` / `architecture/TOOLS.md` / `architecture/SKILLS.md` |

After each merged harness PR: `uv run pytest -m gate -q` green; `python scripts/maintenance/check_harness_no_getattr.py`; sync plan §0.5 gate count.

---
