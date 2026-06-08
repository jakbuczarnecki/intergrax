# `eval.trajectory_judge`

**Bundle:** `eval` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Trajectory-level eval judging for agent regression.

## How it works

eval.judge + eval.record_observation + eval.trajectory.

## How to use

eval_skill_profile(); wire eval harness backend.

## What you get

Step-by-step eval without custom tool wiring.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `eval.judge` | Judge trajectory outcome |
| `eval.record_observation` | Record observation |
| `eval.trajectory` | Fetch trajectory |

## Related skills

- `eval.score_logger`
- `eval.release_compare`
