# Agent autonomy

How agents execute an approved plan without go-ahead questions and come back to the Operator only for what was not planned. Everything lives in the repository; nothing is installed per developer. Works in plain Claude Code; see the Cursor section for the differences.

## Rules (always on)

`agent-env/rules/intergrax-core.md`, section Autonomy, is loaded in every session:

- Execute the approved plan step by step. Never ask "shall I continue / commit / run tests / start the next step". Decide from the plan, ADRs, repository rules or existing code.
- Stop only with a message that starts `ESCALATION: <type>` (types below), batching all open items with options, a recommendation, the consequence and the cost.
- When the plan is finished print `PLAN COMPLETE` and a short summary.
- Agents commit on a branch and prepare the PR text. `git push`, PR creation, merge and GitHub settings stay with the human.

## Escalation types

1. unplanned architectural problem
2. decision that changes scope or cost, or is hard to reverse
3. contradiction between plan and code or rules
4. forbidden action needed (push, PR, merge, settings, secrets, protected paths)
5. missing access

## Running a plan (Claude Code)

1. Write the plan as markdown with `- [ ]` step lines (a plan from plan mode in `agent-env/plans/` works).
2. Start: `/igx-run-plan <plan.md>`. It runs `python scripts/hooks/igx_stop_guard.py start <plan.md>`, which writes the path into `.tmp/session/autonomy/active-plan` (untracked).
3. The agent ticks `- [x]` as steps finish and stops only with `ESCALATION:` or `PLAN COMPLETE`.
4. End early: `/igx-run-plan stop` (or delete the active-plan file). `status` shows the next open step.

## What the Stop hook does

`scripts/hooks/igx_stop_guard.py`, registered for `Stop` in `.claude/settings.json`, stdlib only, fails open on any error, `IGX_HOOKS=off` disables it. Without an active plan it allows every stop. With one it reads the last assistant message:

| Last message | Result |
|--------------|--------|
| starts `ESCALATION:` | allowed |
| starts `PLAN COMPLETE` | allowed, active plan cleared |
| no open step left | allowed, active plan cleared |
| anything else, open step remains | blocked; the reason names the next open step |

Loop limit: the third consecutive block without a checkbox change lets one stop through with a warning, then the counter restarts, so a stuck session cannot spin forever. `stop_hook_active` turns are counted by the same limit instead of being allowed outright (that would block only once).

Verified with real `claude -p` runs (model sonnet): with two open steps and a first reply that was only a question, the hook blocked it, the agent created both files, ticked both steps and ended with `PLAN COMPLETE`; a plan whose first step needed `git push` ended with `ESCALATION: forbidden action needed` and was not blocked. Evidence under `.tmp/session/token-diet/stoprun_*.jsonl` of the session that built it.

## Orchestration

Skill `/igx-autonomy` (name-only) holds the protocol. Sub-agents (`igx-researcher`, `igx-implementer`, `igx-reviewer`, `igx-architect`) end with exactly `DONE: <line>` or `BLOCKED: <type>: <what, options, recommendation>` and report to the parent. The parent decides alone when the plan, an ADR, the rules or an existing code pattern answers it, records the decision as one line under the step, and escalates to the Operator only with the types above, in one message.

## Cursor vs Claude Code

| Piece | Claude Code | Cursor (checked with `cursor-agent`) |
|-------|-------------|--------------------------------------|
| core rule (no questions, `ESCALATION:`, `PLAN COMPLETE`) | yes | yes, via `.cursor/rules/intergrax-core.mdc` |
| typed agents with `DONE:`/`BLOCKED:` | yes | yes, via `.cursor/agents/` |
| `/igx-run-plan`, `/igx-autonomy` skills | yes | generated into `.cursor/skills/`, not exercised |
| Stop hook that blocks questions | yes | no: a first-reply question ended the run |

In Cursor the rule is text only, so an agent can still ask a question; use `/igx-run-plan` from Claude Code for runs that must not stop.

## Nothing to install

Rules, skills, agents, the hook and its registration are tracked files. The active-plan state is under `.tmp/` (gitignored). Needs only `python` on the path, like the read and bash guards.
