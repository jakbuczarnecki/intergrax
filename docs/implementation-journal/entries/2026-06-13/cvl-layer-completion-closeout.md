---
id: IJ-2026-06-13-011
date: 2026-06-13
tiers:
  - tier-0
  - tier-1
scope: CRITIC_VERIFICATION
plan_ref:
  - CVL-LC-1
  - CVL-LC-2
  - GAP-CVL-10
status: completed
commit: 93995259
adr: none — doc sync and NexusEvalRunner fail-closed wiring; no contract change
---

# CVL — Critic Verification layer completion closeout

## Operator request

Close the perception gap where CRITIC_VERIFICATION architecture still opens with pre-CRIT-V gap list despite CRIT-V-0…7 + FOLLOWUP being Done in code and plan register.

## Summary

Layer Completion Mode audit confirmed CVL runtime is **L3+** (eval.judge, eval.trajectory, EvaluatorLoopExecutor, semantic NexusEvalRunner, graph hooks, Tier-3 wiring). Updated architecture §2 to historical gap table with closure status; added plan audit register §CVL; regenerated audit prompt. Sprint CVL-LC-2: `NexusEvalRunner.from_nexus_loop` auto-wires L1 client from critic hooks; semantic mode fails closed when client missing.

## Project impact

Operators and audit agents see honest L3+ CVL status. Offline harness eval can reuse wired critic path without manual client injection; semantic eval no longer silently falls back to exact match.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CRITIC_VERIFICATION.md` §2, §13 |
| Plan | `docs/plan/CRITIC_VERIFICATION.md` — Audit Result CVL |
| ADR | ADR-CRITIC-001 — no amendment |
| Audit / gap | GAP-CVL-01…10 closed; CVL-BACKLOG-01…06 |

## Changed artifacts

- `docs/architecture/CRITIC_VERIFICATION.md` — §2 historical gaps + L3+ status
- `docs/plan/CRITIC_VERIFICATION.md` — audit register, removed duplicate CRIT-V section
- `scripts/generate_domain_audit_prompts.py` — CVL Done status
- `docs/audit/CRITIC_VERIFICATION.md` — regenerated
- `intergrax/eval/nexus_eval_runner.py` — semantic auto-wire + fail-closed
- `intergrax/runtime/critic/l1_gateway.py`, `critic_orchestrator.py` — tool_client accessors
- `intergrax/runtime/nexus/nexus_loop.py` — `critic_eval_tool_client()`
- `tests/unit/eval/test_nexus_eval_runner_semantic.py` — fail-closed test

## Verification

```bash
uv run pytest tests/unit/eval/test_nexus_eval_runner_semantic.py tests/unit/runtime/critic/test_critic_orchestrator.py tests/unit/runtime/critic/test_critic_closeout.py -m gate -q
uv run pytest tests/unit/runtime/critic/test_critic_graph_wiring.py tests/unit/runtime/critic/test_critic_evaluator_loop_graph.py -m gate -q
```

Result: pass — 40+ unit tests; graph integration tests pass in isolation.

## Risks and follow-ups

- CVL-BACKLOG-01: LLM trajectory judge optional path (eval.trajectory_judge skill)
- CVL-BACKLOG-02: critic graph test isolation when tool catalog pre-registered in session
- CVL-BACKLOG-05/06: L4 adaptive thresholds, FLOW-8 product host (P4)
