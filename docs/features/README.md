<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Multi-layer Feature Documentation

**Status:** Canonical documentation structure  
**Purpose:** Document cross-layer Intergrax capabilities that cut across multiple architecture domains while preserving the existing `docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md` 1:1 layer rule.

---

## Why this structure exists

Intergrax has two kinds of technical work:

1. **Domain-layer work** — belongs to one architecture domain and its matching implementation plan.
2. **Multi-layer feature work** — delivers one product/runtime capability by coordinating changes across several domains.

Domain-layer work remains documented in the existing 1:1 structure:

```text
docs/architecture/<DOMAIN>.md
docs/plan/<DOMAIN>.md
```

Multi-layer features use a parallel 1:1 structure under `docs/features/`:

```text
docs/features/architecture/<FEATURE>.md
docs/features/architecture/satellites/
docs/features/plan/<FEATURE>.md
docs/features/plan/satellites/
```

Feature **hubs** (`architecture/<FEATURE>.md`, `plan/<FEATURE>.md`) are Cursor entry points for analysis and implementation. **Satellites** hold bulky cross-domain sync registers and extended detail — same explicit-load rule as domain-layer `docs/architecture/satellites/` and `docs/plan/satellites/` (listed in `.cursorignore`).

Do **not** create `docs/features/satellites/` at the features root. Satellites belong under each hub tier, not beside `architecture/` and `plan/`.

---

## Rules

1. Do not create `docs/plan/<FEATURE>.md` for a multi-layer feature unless `<FEATURE>` is promoted into a full architecture domain with a matching `docs/architecture/<FEATURE>.md`.
2. Every file in `docs/features/plan/` must have a matching file in `docs/features/architecture/`.
3. Every file in `docs/features/architecture/` must have a matching file in `docs/features/plan/`.
4. Feature architecture documents describe cross-layer capability boundaries, ownership, integration points, invariants, and safety rules.
5. Feature plan documents coordinate phases across domains, but domain-specific implementation rows still belong in the owning `docs/plan/<DOMAIN>.md` files when implementation begins.
6. Feature plans must identify the domain plan rows they depend on or intend to add.
7. Feature docs must not override domain-layer architecture. They coordinate it.
8. Feature satellites follow the same hub + satellite split as domain docs: `architecture/satellites/` for architecture-side registers, `plan/satellites/` for plan-side registers. Load at most one satellite per session unless RESUME cites more.

---

## Current multi-layer features

| Feature | Architecture | Plan | Status |
|---------|--------------|------|--------|
| `TOKEN_OPTIMIZATION` | [`architecture/TOKEN_OPTIMIZATION.md`](architecture/TOKEN_OPTIMIZATION.md) | [`plan/TOKEN_OPTIMIZATION.md`](plan/TOKEN_OPTIMIZATION.md) | Implemented foundation; TOKEN-10 cache-aware runtime and proof planned |

**Satellites (on demand):**

| Feature | Architecture satellite | Plan satellite |
|---------|------------------------|----------------|
| `TOKEN_OPTIMIZATION` | [`architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md`](architecture/satellites/TOKEN_OPTIMIZATION_domain_architecture_cross_references.md) | [`plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md`](plan/satellites/TOKEN_OPTIMIZATION_domain_plan_cross_references.md) |

---

## Authoring flow

```text
idea / audit instruction
  → docs/features/architecture/<FEATURE>.md
  → docs/features/plan/<FEATURE>.md
  → update affected docs/architecture/<DOMAIN>.md files
  → add concrete rows to affected docs/plan/<DOMAIN>.md files
  → implement smallest domain-owned slice
  → gate + journal
```

Use feature docs for coordination. Use domain docs for ownership and implementation truth.
