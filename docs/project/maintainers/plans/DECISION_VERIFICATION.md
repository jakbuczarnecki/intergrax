# Decision Verification — Implementation Plan

**Architecture (1:1):** [`architecture/DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Parent:** [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **DS-DOC-HARDEN (2026-08-30):** Target Verification Pipeline architecture **FROZEN**. Runtime still uses `CriticOrchestrator` until migration.

**Last updated:** 2026-08-30

---

## Cursor read scope (token budget)

- **Default:** hub status + open P0/P1 summary only.
- **Detail rows:** [`satellites/DECISION_VERIFICATION_implementation_pipeline.md`](satellites/DECISION_VERIFICATION_implementation_pipeline.md) — one satellite per session max.
- **Architecture:** [`DECISION_VERIFICATION.md`](../../architecture/DECISION_VERIFICATION.md) read-scope block.
- **Lifecycle context:** [`DECISION_SYSTEM.md`](../../architecture/DECISION_SYSTEM.md) — version binding on demand.
- **CURRENT code:** `intergrax/runtime/critic/**` — migration audit only; one module per session.

---

## Implementation satellite

| Satellite | Contents |
| --------- | -------- |
| [`satellites/DECISION_VERIFICATION_implementation_pipeline.md`](satellites/DECISION_VERIFICATION_implementation_pipeline.md) | DS-VER-PIPE · DS-VER-STAGES · migrated open requirements |

---

## Architecture frozen vs implementation planned

| Layer | Status |
| ----- | ------ |
| **Target architecture** | **FROZEN** |
| **Verification Pipeline runtime** | **PLANNED** |
| **CURRENT production** | `CriticOrchestrator` + L0/L1/L2 gateways |

---

## Phase index

| Phase | Status | Satellite |
| ----- | ------ | --------- |
| DS-VER-PIPE | PLANNED | [`implementation_pipeline`](satellites/DECISION_VERIFICATION_implementation_pipeline.md) |
| DS-VER-STAGES | PLANNED | same satellite |

---

## Explicit non-goals (this plan)

- L2 Human verification stage — **DELETE** from verification; use HITL via Lifecycle.
- `policy_bridge` verdict → action mapping — **SPLIT** to Policy + Lifecycle.
- Offline/shadow eval ownership — remains **OUTSIDE** pipeline ([`CRITIC_VERIFICATION.md`](../../architecture/CRITIC_VERIFICATION.md) eval boundary).

---

## Delivery rule

One **DS-VER-\*** ID per PR → update pipeline satellite → parent [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md) disposition when Critic capability retired.
