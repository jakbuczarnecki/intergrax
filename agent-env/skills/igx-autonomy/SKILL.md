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

## Cursor — parent and workers

- The parent orchestrator in Cursor may be Grok.
- The default worker is Composer 2.5. Pass Task `model` `composer-2.5` or `composer-2.5-fast` on every worker. Do not omit `model` (that inherits the parent).
- Claude, GPT, Gemini, DeepSeek, or a Grok worker needs an Operator override in that chat. IIAC and G6 independence are unchanged: a reviewer still does not declare G6 PASS.

## Cursor — background task lifecycle

`run_in_background: true` returns after spawn. The result arrives as the completion notification once the parent ends its turn. That receive is not polling.

`SPAWN` (Task) → end the turn or do other independent work → `RECEIVE` (the completion notification, or the foreground tool result) → `REVIEW` (check `DONE` against git) → `RESUME/CORRECT` (`resume` with the decision, or a new launch) → `NEXT TASK`.

Do not `AwaitShell`, poll, or sleep-loop for a background task. Foreground Task (`run_in_background` false) blocks until the result and is also a receive.

## Commit ownership

SSOT: `may_commit` in `governance/agent-control-plane/role_registry.yaml`.

- `implementer` (`igx-implementer`, `best-of-n-runner`) may commit only on the branch in its own worktree, with `Ref: governance/tickets/T-IGX-<NN>-<L>.md` and no AI attribution.
- `orchestrator` may not commit, merge, push, or open a pull request. Integrating a worker branch into the operator branch is a human action: escalate.
- A subagent commits only when its `maps_to` role has `may_commit: true`. Read-only roles never commit. No subagent closes a batch or declares G6 PASS.