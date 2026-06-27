# ADR-OBS-003: Layered runtime event identity (spine + kind)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-17 |
| **Deciders** | Harness platform |
| **Related** | [`architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §4.4 · [`plan/OBSERVABILITY.md`](../../plan/OBSERVABILITY.md) OBS-EVOL-9 · [ADR-OBS-001](../2026-06-08/ADR-OBS-001.md) · P1-ARCH-02 |

## Context

The Harness Observability Spine (HOS, ADR-OBS-001) standardizes on `RuntimeEvent` + `RuntimeEventType` as the primary audit signal. The catalog has grown to **74** spine types with parallel registries (`phase_coverage.py`, `payload_registry.py`, architecture tables). Without a scaling model, the platform risks:

- **Event explosion** — every domain feature adds a new enum member
- **Subscriber fatigue** — hooks, policy, metrics, and ops rules keyed on growing enum lists
- **Author confusion** — Tier-2/3 developers unsure whether to add `RuntimeEventType`, use `TASK_PROGRESS`, or emit `DiagnosticPayload` only

**Pre-release constraint:** external developers have not yet consumed the public API. This is the last low-cost window to establish a **stable extension contract** before publication.

Alternatives considered:

1. **Keep flat `RuntimeEventType` only** — rejected: does not scale; forces platform PRs for product semantics.
2. **Hierarchical enum (`TOOL.REQUESTED`)** — rejected: breaking wire/persistence; high migration cost.
3. **Separate buses per category** — rejected: violates unified spine (ADR-OBS-001).
4. **Layered identity: spine `event_type` + namespaced `event_kind`** — **accepted**.

## Decision

Adopt **layered runtime event identity** on the existing HOS bus:

1. **`event_type` (spine)** — small, platform-owned `RuntimeEventType` set (~50 lifecycle signals). Moratorium on growth after pre-release consolidation except via ADR.
2. **`event_kind` (semantic)** — stable namespaced string (`domain.action`, e.g. `agents.legal.clause_flagged`). Primary identity for domain extensions. Defaults to `event_type.value` for spine events.
3. **`event_category` (derived)** — coarse grouping (`tool`, `agent`, `plan`, `memory`, `platform`, …) computed from `event_kind` or catalog metadata — used for bus subscription, metrics cardinality control, and ops dashboards.
4. **`EventCatalog`** — single source of truth merging phase, ops hint, category, and preferred payload schema per spine type; generates gates and docs.
5. **`DOMAIN_SIGNAL` spine type** — canonical carrier for Tier-2/3 domain signals that are not Nexus lifecycle transitions.
6. **Public emit APIs:**
   - `emit_platform_event()` — platform only; requires spine `event_type` + catalog entry
   - `emit_domain_signal()` — Tier-2/3; requires `event_kind` + registered extension payload; uses `DOMAIN_SIGNAL`
7. **Plane B unchanged** — fine-grained debug remains `DiagnosticPayload` + `TraceComponent`; do not duplicate in spine unless operators must see it on the bus.

**Pre-release spine consolidation:** collapse non-lifecycle types (adaptive, capacity, scale, hook detail) into `DOMAIN_SIGNAL` + `platform.*` kinds before v1 publication (OBS-EVOL-9.7).

## Consequences

### Positive

- Developers extend via `event_kind` + payload registry — no platform enum PRs for product semantics
- Ops/metrics/hooks subscribe by `event_category` / `kind_prefix` — stable as catalog grows
- Single catalog reduces drift between `phase_coverage`, docs, and gates
- Pre-release consolidation avoids external migration pain

### Negative

- Three identity fields on `RuntimeEvent` require author education (mitigated by helpers and scaffold)
- Residual `phase_coverage.py` until catalog migration completes
- Metrics must enforce cardinality rules on `event_kind` labels

## Compliance

- ADR-OBS-001 unified spine preserved — one bus, one journal
- Tier-2 must not import trace stores; domain signals go through Nexus context helpers
- Linked: `architecture/OBSERVABILITY.md` §4.4, `architecture/UNIFIED_EXECUTION_RUNTIME.md` §42.1.6, `plan/OBSERVABILITY.md` OBS-EVOL-9

## Implementation notes

- Tracker: Phase **OBS-EVOL-9** (M0 doc → M4 optional `runtime_event.v2`)
- **SAR accepted (2026-06-17):** EmitContext, retention_class, profile subscriptions, traceparent, sampling, deprecation shim, JournalQuery, LLM namespace lint, domain redaction; per-category buses and hierarchical enum **rejected**
- Verification: `scripts/maintenance/check_event_catalog.py` (planned), existing phase/payload gates until migration
- Author guides: `AGENT_CREATION_GUIDE.md` Appendix Q §Q.5, `EXTENSION_AUTHOR_GUIDE.md` §11, `APPLICATION_CREATION_GUIDE.md` §8
