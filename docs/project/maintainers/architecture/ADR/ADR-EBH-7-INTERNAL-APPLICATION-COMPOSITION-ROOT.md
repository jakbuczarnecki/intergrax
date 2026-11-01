# ADR-EBH-7 — INTERNAL: Tier-3 application composition root

| Field | Value |
| --- | --- |
| **Status** | Ratified @ operator EBH-7 terminal authorization (`fd3bc0104`) |
| **Concern** | `application.composition_root` |
| **Classification** | `INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED` |

## Context

Tier-3 `applications/` composition roots wire sanctioned platform contracts per app. This is not vendor/backend replaceability at the integration-catalog layer.

## Decision

- RPL-H for `application.composition_root` is **INTERNAL**, not REPLACEABLE.
- Qualification evidence: composition root discovery + tier boundary gates (`test_ebh_7_terminal_rpl_gates`).
- No fake second composition root is required for EBH-7 B6 readiness.

## Consequences

- Matrix row `rpl_h=INTERNAL`; normative `INTERNAL_RATIFIED`.
- Live vendor neutrality for app wiring remains governed by underlying integration concerns, not this matrix row.
