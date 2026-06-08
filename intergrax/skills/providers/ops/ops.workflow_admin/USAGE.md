# `ops.workflow_admin`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Workflow run administration: list, cancel, and inspect logs.

## How it works

workflow.list_runs + cancel_run + fetch_logs.

## How to use

ops_skill_profile(); batch orchestration hosts.

## What you get

Ops visibility beyond workflow_runner trigger path.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `workflow.list_runs` | List workflow runs |
| `workflow.cancel_run` | Cancel in-flight run |
| `workflow.fetch_logs` | Fetch run logs |

## Related skills

- `ops.workflow_runner`
- `eval.release_compare`
