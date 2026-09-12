# ERL-QUAL-004 — Scenario dataset

Vendor-neutral **logical** qualification inputs for [Enterprise Payment Uncertainty Recovery](../README.md). Architecture: [ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md](../docs/ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md).

This package is **data only** — no loaders, adapters, provisioning, or runtime logic.

## Layout

```text
dataset/
  manifest.json                 # package index and variant paths
  shared/                       # structural entities common to variants A/B/C
  variants/<variant_id>/        # external reality, reconciliation, expected outcomes
```

## Shared structural entities

| File | Business concept |
| --- | --- |
| `shared/order.json` | Order identity, customer context, amount, currency, importance |
| `shared/external_effect.json` | External payment capture effect and immediate **unknown** integration outcome |
| `shared/communication_event.json` | Uncertainty condition (e.g. lost response) — describes condition, not runtime behavior |
| `shared/inventory_context.json` | Reservation tied to order; pending payment truth |
| `shared/application_knowledge_at_entry.json` | What the application legitimately knows at UNKNOWN entry (≠ external reality) |

## Variants

| `variant_id` | External reality (SoR) | Expected proof result |
| --- | --- | --- |
| `payment_completed_after_unknown` | Payment completed | Safe continuation after reconciliation |
| `payment_failed_after_unknown` | Payment failed | Controlled recovery |
| `payment_truth_unavailable` | Truth not establishable | Governance escalation |

Select a variant via `manifest.json` → `variants[].path`. Provisioning implementations (see `../contracts/provisioning/` and `../provisioning/reference/`) read this package; they do not alter canonical business truth here.

## Versioning

- Package: `manifest.json` → `package_version` / `schema_version`
- Identifiers are fixed strings for deterministic, repeatable qualification runs.
