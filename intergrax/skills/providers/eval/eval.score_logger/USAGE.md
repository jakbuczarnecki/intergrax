# `eval.score_logger`

**Bundle:** `eval` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Log eval scores to Braintrust and query correlated traces.

## How it works

braintrust.log_eval + observability.query_traces.

## How to use

eval_skill_profile(); wire braintrust observability backend.

## What you get

Standard eval harness agent pack.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `braintrust.log_eval` | Log eval score |
| `observability.query_traces` | Correlate traces |

## Related skills

- `ops.trace_debug`
- `ops.workflow_runner`
