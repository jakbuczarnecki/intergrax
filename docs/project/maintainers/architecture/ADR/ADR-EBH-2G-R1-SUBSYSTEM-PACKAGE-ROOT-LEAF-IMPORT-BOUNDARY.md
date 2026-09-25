# ADR-EBH-2G-R1 — Subsystem package root leaf-import boundary

| Field | Value |
| ----- | ----- |
| **Status** | Accepted (EBH-2G-R1 — pending independent audit) |
| **Date** | 2026-09-25 |
| **Task** | EBH-2G-R1 — Package Import Boundary & Canonical Leaf API Closure |
| **Semantic authority** | [`INTEGRATIONS.md`](../../architecture/INTEGRATIONS.md) · [`TOOLS.md`](../../architecture/TOOLS.md) · [`SKILLS.md`](../../architecture/SKILLS.md) · [`RAG.md`](../../architecture/RAG.md) · [`PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md) |

---

## Context

Importing integration contracts (for example `IntegrationStatus` from a leaf contract module) previously triggered a wide package-initialization graph: registry bootstrap, observability contracts, skills, tools wiring, and circular imports on the RAG-MAINT-01 qualification path. That violated the invariant **contract import ≠ runtime composition**.

## Decision

1. **Subsystem package roots are namespace boundaries, not stable public SDK facades.**
2. **Canonical API ownership belongs to explicit owner leaf modules** (one symbol → one leaf owner).
3. **Importing contracts or leaf APIs must not trigger** runtime composition, registry bootstrap, resolver materialization, provider discovery, or unrelated subsystem initialization.
4. **Legacy root-level convenience re-exports** are pre-freeze, non-canonical surfaces and are removed before ARCH-FREEZE.
5. **No lazy compatibility facade** at subsystem roots: no `__getattr__`, `importlib` workarounds, proxy facades, or duplicate root+leaf API paths for the same symbol.

## Scope

In scope for this ADR:

- `intergrax.integrations`
- `intergrax.integrations.contracts`
- `intergrax.skills`
- `intergrax.tools`

Out of scope: global audit of every platform `__init__.py` (tracked as freeze debt under EBH-3 / EBH-4 / COMPAT-X when discovered elsewhere).

## Canonical import rule

Consumers import the **explicit owner leaf module** for each symbol, for example:

```python
from intergrax.integrations.contracts.base import IntegrationStatus
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.skills.core.contracts import SkillManifest
from intergrax.tools.registry.wiring import ToolWiringContext
```

## Root-package rule

Each in-scope `__init__.py` must remain **lightweight, deterministic, and side-effect-free**: no registry bootstrap, resolver bootstrap, tool wiring, provider registration, or eager aggregation of the subsystem graph.

## Compatibility classification

| Surface | Classification |
| ------- | -------------- |
| Historical root-level re-exports | **LEGACY CONVENIENCE IMPORT SURFACE** — non-canonical, not frozen |
| Leaf contract / registry / resolver modules | **Canonical** — import from owner leaf only |

## Relation to COMPAT-X

**COMPAT-X** (later) freezes and versions deliberately chosen public/stable contracts. This migration runs **before** ARCH-FREEZE and removes accidental API surfaces; it does **not** defer removal to COMPAT-X and does not introduce a versioned SDK facade in this task.

## Forbidden patterns (subsystem roots)

- `def __getattr__(name): ...` compatibility shims
- Lazy imports / `importlib` dynamic symbol resolution
- `try/except ImportError` silent backwards compatibility at roots
- Permanent dual path (`from intergrax.integrations import X` and leaf import for the same symbol)

## Migration rule

- Production, tests (except explicit root-lightweight gates), docs, examples, and scaffolds: **leaf import path only**; semantics unchanged.
- If a historical export has no leaf owner → **STOP — architecture decision required** (do not invent a facade).

## Regression protection

- Mechanical gates: `tests/unit/architecture/test_ebh_2g_r1_subsystem_package_import_boundary.py` (root `__init__` AST + leaf-import isolation subprocess proofs).
- RAG qualification path: RAG-MAINT-01 and `rag-guard` equivalent suite remain green without `sys.path` hacks or duplicate status enums.

## Consequences

- Qualification and contract consumers can import leaf modules without materializing runtime composition.
- Documentation and scaffolds must teach leaf paths only.
- A future stable public SDK facade, if needed, is a separate versioned contract — not subsystem root re-exports.
