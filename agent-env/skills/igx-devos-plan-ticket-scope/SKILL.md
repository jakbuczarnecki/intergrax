---
name: igx-devos-plan-ticket-scope
description: Plan approval workflow — digest, operator Approved line, verify_plan_approval.
---

# Plan ticket scope

1. `uv run python scripts/governance/plan_digest.py <plan.md>` → record `Plan digest: <sha256>` in ticket.
2. Operator adds `Approved: <name> @ <ISO8601>`.
3. Verify: `uv run python scripts/governance/verify_plan_approval.py governance/tickets/T-IGX-....md --plan <plan.md>`.
4. Transition lifecycle to `plan_approved` before implementation.

Architect-planner stays read-only.
