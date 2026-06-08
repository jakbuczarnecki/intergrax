# `hitl.approval_gate`

**Bundle:** `hitl` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Human-in-the-loop approval for high-risk agent actions.

## How it works

hitl.* via HumanDecisionStoreBinding; notify.send for alerts.

## How to use

hitl_skill_profile(); enable HITL store on harness host.

## What you get

Governed approval without per-agent HITL wiring.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `hitl.list_pending` | List pending decisions |
| `hitl.submit_response` | Submit human response |
| `hitl.get_decision` | Fetch decision record |
| `notify.send` | Alert stakeholder |

## Related skills

- `ops.incident_dispatch`
- `legal.contract_review`
