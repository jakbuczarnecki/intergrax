---
name: igx-autonomy
description: Orchestration protocol and decision policy - sub-agent DONE/BLOCKED reports, when the parent decides alone, when and how to escalate to the Operator
---

# igx-autonomy

Goal: the approved plan runs without go-ahead questions; the Operator hears only about what was not planned.

## Sub-agent report

A sub-agent ends with exactly one line:

- `DONE: <one line>` - result, commit sha, tests.
- `BLOCKED: <type>: <what, options, recommendation>` - type is one of the escalation types below.

## Parent decides alone

When the answer follows from the approved plan, an ADR, the repository rules (`intergrax-core`, `AGENTS.md`) or an existing pattern in the code:

1. Pick the option that stays inside scope, is reversible and matches existing code.
2. Tell the sub-agent in one line (or re-launch it with the decision) and continue.
3. Record the decision as one line in the plan file under the step.

Never forward a sub-agent question to the Operator when this applies. Never ask the Operator whether to continue, commit, run tests, retry or start the next step.

## Escalate to the Operator

Only for these types, in ONE message that starts `ESCALATION: <type>` and batches every open item:

1. unplanned architectural problem
2. decision that changes scope or cost, or is hard to reverse
3. contradiction between plan and code or rules
4. forbidden action needed (push, PR, merge, settings, secrets, protected paths)
5. missing access

Per item: the problem, the options, a recommendation, the consequence and the cost. Then wait: the stop is allowed only with this prefix (or `PLAN COMPLETE`).

## Limits

- Sub-agents never launch sub-agents and never message the Operator; they report to the parent.
- Typed agents: `igx-researcher` (read-only), `igx-implementer` (worktree), `igx-reviewer` (no shell), `igx-architect` (plans only). `general-purpose` is denied.
- One `uv run pytest` at a time; the parent verifies a sub-agent's `DONE` against git before ticking the step.
