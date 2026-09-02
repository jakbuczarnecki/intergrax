<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multi-layer Feature Documentation

**Status:** Canonical documentation structure  
**Purpose:** Document cross-layer Intergrax capabilities that cut across multiple architecture domains while preserving the existing `docs/project/architecture/<DOMAIN>.md` ↔ `docs/project/maintainers/plans/<DOMAIN>.md` 1:1 layer rule.

---

## Why this structure exists

Intergrax has two kinds of technical work:

1. **Domain-layer work** - belongs to one architecture domain and its matching implementation plan.
2. **Multi-layer feature work** - delivers one product/runtime capability by coordinating changes across several domains.

Domain-layer work remains documented in the existing 1:1 structure:

```text
docs/project/architecture/<DOMAIN>.md
docs/project/maintainers/plans/<DOMAIN>.md
```

Multi-layer features use a parallel 1:1 structure under `docs/project/capabilities`:

```text
docs/project/capabilities/architecture/<FEATURE>.md
docs/project/capabilities/architecture/satellites/
docs/project/capabilities/plan/<FEATURE>.md
docs/project/capabilities/plan/satellites/
```

Feature **hubs** (`architecture/<FEATURE>.md`, `plan/<FEATURE>.md`) are Cursor entry points for analysis and implementation. **Satellites** hold bulky cross-domain sync registers and extended detail - same explicit-load rule as domain-layer `docs/project/architecture/satellites` and `docs/project/maintainers/plans/satellites` (listed in `.cursorignore`).

Do **not** create `docs/project/capabilities/satellites` at the features root. Satellites belong under each hub tier, not beside `architecture` and `plan`.

---

## Rules

1. Do not create `docs/project/maintainers/plans/<FEATURE>.md` for a multi-layer feature unless `<FEATURE>` is promoted into a full architecture domain with a matching `docs/project/architecture/<FEATURE>.md`.
2. Every file in `docs/project/capabilities/plan` must have a matching file in `docs/project/capabilities/architecture`.
3. Every file in `docs/project/capabilities/architecture` must have a matching file in `docs/project/capabilities/plan`.
4. Feature architecture documents describe cross-layer capability boundaries, ownership, integration points, invariants, and safety rules.
5. Feature plan documents coordinate phases across domains, but domain-specific implementation rows still belong in the owning `docs/project/maintainers/plans/<DOMAIN>.md` files when implementation begins.
6. Feature plans must identify the domain plan rows they depend on or intend to add.
7. Feature docs must not override domain-layer architecture. They coordinate it.
8. Feature satellites follow the same hub + satellite split as domain docs: `architecture/satellites` for architecture-side registers, `plan/satellites` for plan-side registers. Load at most one satellite per session unless RESUME cites more.

### Domain-vs-feature example: Autonomous Work / Virtual Workforce

`AUTONOMOUS_WORK` is a **DOMAIN**, not a feature, because it owns new reusable platform semantics: `WorkerDefinition`, `WorkerInstance`, durable Responsibility/Goal, worker lifecycle, work-intake/wake-up semantics and worker-level recovery composition. Its canonical pair is:

```text
docs/project/architecture/AUTONOMOUS_WORK.md
docs/project/maintainers/plans/AUTONOMOUS_WORK.md
```

`Virtual Worker` is the primary product-facing abstraction of that domain.

A future `VIRTUAL_WORKFORCE` feature pair may be created **only when** product/runtime composition needs a cross-layer coordinator spanning Autonomous Work, Collaborative Work, Governed Execution, CodeCraft, Observability, Hosting and other domain owners. That future feature must not become the owner of Worker contracts.

This is analogous to `MULTIPLAYER_AI`: Multiplayer is a cross-layer feature, while `COLLABORATIVE_WORK` is the anchor domain owning core collaborative semantics.

---

## Current multi-layer features

| Feature | Guide | Architecture | Plan | Status |
|---------|-------|--------------|------|--------|
| `TOKEN_OPTIMIZATION` | [`token_optimization/README.md`](token_optimization/README.md) | [`architecture/TOKEN_OPTIMIZATION.md`](architecture/TOKEN_OPTIMIZATION.md) | [`plan/TOKEN_OPTIMIZATION.md`](plan/TOKEN_OPTIMIZATION.md) | Implemented foundation; TOKEN-10 cache-aware runtime and proof planned |
| `LANGCHAIN_INDEPENDENCE` | - | [`architecture/LANGCHAIN_INDEPENDENCE.md`](architecture/LANGCHAIN_INDEPENDENCE.md) | [`plan/LANGCHAIN_INDEPENDENCE.md`](plan/LANGCHAIN_INDEPENDENCE.md) | Architecture and migration roadmap awaiting review; implementation not started |
| `MULTIPLAYER_AI` | - | [`architecture/MULTIPLAYER_AI.md`](architecture/MULTIPLAYER_AI.md) | [`plan/MULTIPLAYER_AI.md`](plan/MULTIPLAYER_AI.md) | **MP-0** - canonical architecture and roadmap (documentation only); MP-1…MP-9 planned, not started |

**Reserved future product/cross-layer concept:** `VIRTUAL_WORKFORCE` — no feature pair exists yet; canonical worker semantics belong to [`../architecture/AUTONOMOUS_WORK.md`](../architecture/AUTONOMOUS_WORK.md).

**Satellites (on demand):**

| Feature | Architecture satellite | Plan satellite |
|---------|------------------------|----------------|
| `TOKEN_OPTIMIZATION` | [`architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md) | [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) |
| `LANGCHAIN_INDEPENDENCE` | [`architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md) | [`plan/satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md`](plan/satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md) |

---

## Authoring flow

```text
idea / audit instruction
  → classify: DOMAIN or FEATURE
  → DOMAIN: docs/project/architecture/<DOMAIN>.md ↔ docs/project/maintainers/plans/<DOMAIN>.md
  → FEATURE: docs/project/capabilities/architecture/<FEATURE>.md ↔ docs/project/capabilities/plan/<FEATURE>.md
  → update affected domain architecture files
  → add concrete rows to affected domain plans when implementation begins
  → implement smallest domain-owned slice
  → gate + journal
```

Use feature docs for coordination. Use domain docs for ownership and implementation truth.
