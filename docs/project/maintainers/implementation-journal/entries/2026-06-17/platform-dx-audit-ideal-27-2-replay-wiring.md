---
id: IJ-2026-06-17-044
date: 2026-06-17
tiers:
  - tier-3
scope: EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
plan_ref:
  - AUDIT-IDEAL-27.2
status: completed
commit: pending
adr: none — wiring resolver mirrors trace_explorer pattern; no contract change
---

# AUDIT-IDEAL-27.2 — Replay environment HTTP wiring on product hosts

## Operator request

Continue Mode B architecture audit (`03_implement_plan_all_domains.txt`) from domain `TOOLS` through completion; implement or skip open P0/P1 backlog rows per domain.

## Summary

Added `resolve_replay_environment_wiring()` for `ApplicationProfile.PRODUCT` hosts, gated by `ApplicationFeatures.replay_environment_enabled` (default **true** in `product_defaults()`). CI gate `check_replay_environment_wiring.py` asserts `/harness/replay` is exposed. Registered gate in `check_audit_ideal_gates.py` umbrella.

## Project impact

Product host profiles now have a typed, gate-verified contract for harness replay HTTP API — consistent with trace explorer and agent simulator wiring patterns.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` |
| Plan | `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — AUDIT-IDEAL-27.2 |
| Audit / gap | AUDIT-IDEAL-27.2 |

## Changed artifacts

- `intergrax/applications/_shared/replay_environment_wiring.py` — product host resolver
- `intergrax/applications/contracts/application_host.py` — `replay_environment_enabled` flag
- `scripts/maintenance/check_replay_environment_wiring.py` — gate script
- `scripts/gates/check_audit_ideal_gates.py` — umbrella registration
- `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` — wiring unit test
- `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — row Done

## Verification

```bash
uv run python scripts/maintenance/check_replay_environment_wiring.py
uv run pytest tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_27_2_replay_environment_wiring -q
```

## Risks and follow-ups

- Host factories still mount routers via feature wiring at integration time; resolver + gate establish contract parity with AUDIT-IDEAL-27.1/27.3.
