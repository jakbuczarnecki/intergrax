---
id: IJ-2026-06-13-012
date: 2026-06-13
tiers:
  - tier-0
  - tier-1
scope: CRITIC_VERIFICATION
plan_ref:
  - CVL-LC-3
  - CVL-LC-4
  - CVL-BACKLOG-01
  - CVL-BACKLOG-02
status: completed
commit: 8a1ed778
adr: none — bootstrap idempotency and trajectory doc clarification; no contract change
---

# CVL — Layer completion iteration II (bootstrap idempotency + trajectory docs)

## Operator request

Force another CVL layer completion iteration despite Architecturally Mature state from CVL-LC-1/2.

## Summary

Closed remaining P2 backlog: `register_default_tools()` now overrides pre-registered bundles (fixes critic graph tests in combined gate sessions). Documented dual-mode trajectory eval — heuristic `eval.trajectory` tool vs `eval.trajectory_judge` skill for LLM regression. Added bootstrap idempotency tests and NexusEvalRunner from_nexus_loop wiring test.

## Project impact

CVL gate tests stable in full session runs. Trajectory L1 contract is explicit — no false LLM-rubric expectation on `eval.trajectory`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/CRITIC_VERIFICATION.md` §7.3 |
| Plan | `docs/project/maintainers/plans/CRITIC_VERIFICATION.md` — CVL-LC-3/4, backlog |
| ADR | ADR-CRITIC-001 — no amendment |

## Changed artifacts

- `intergrax/tools/registry/bootstrap.py`, `catalog.py` — idempotent registration
- `tests/unit/tools/registry/test_bootstrap_idempotent.py` (new)
- `tests/unit/eval/test_nexus_eval_runner_semantic.py` — from_nexus_loop wire test
- `docs/project/architecture/CRITIC_VERIFICATION.md`, `docs/project/maintainers/plans/CRITIC_VERIFICATION.md`

## Verification

```bash
uv run pytest tests/unit/runtime/critic/ tests/unit/tools/providers/eval/ tests/unit/eval/ tests/unit/tools/registry/test_bootstrap_idempotent.py -m gate -q
```

Result: pass — 45 tests.

## Risks and follow-ups

- CVL-BACKLOG-05/06: L4 adaptive thresholds, FLOW-8 product host (P4)
- Optional future: LLM trajectory mode as separate tool id (P4)
