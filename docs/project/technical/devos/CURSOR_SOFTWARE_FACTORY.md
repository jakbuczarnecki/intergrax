# Cursor software factory - Operator runbook

Ticket: `governance/tickets/T-IGX-057-A.md`. Plan: `agent-env/plans/cursor-software-factory.md`. Cursor only; Claude Code
behaviour is unchanged. This supersedes the execution loop described in [CURSOR_OD_AF_INT_A.md](./CURSOR_OD_AF_INT_A.md).

## What it is

One Cursor parent chat (for example Grok) runs an approved multi-step plan to completion through Composer subagents,
without go-ahead prompts. State lives in `.tmp/factory/<plan-id>/state.json`, driven by
`python scripts/cursor/igx_factory.py` (commands `start`, `next`, `launch`, `receive`, `review`, `integrate`, `park`,
`status`, `complete`). Each step runs on its own worker branch `factory/<plan-id>/<step>`; accepted branches are merged
locally into `factory/<plan-id>/int` by `igx-integrator`. Nothing is pushed.

Protocol for the agent: `.cursor/rules/igx-factory.mdc` (always on) and `/igx-native-run-plan`.

## Prerequisites

- Cursor CLI user setting: set `"attribution": {"attributeCommitsToAgent": false}` in `~/.cursor/cli-config.json`.
  Otherwise Cursor appends `Co-authored-by: Cursor <cursoragent@cursor.com>` to every worker commit; `receive` rejects
  it and each step burns a correction. `start` prints a non-fatal `warnings` entry when the setting is on. The briefs also
  tell workers to amend such a trailer away, but the factory never edits user configuration.
- Headless runs need workspace trust; the supervisor passes `--trust` itself (see below).

## Plan format

See [templates/FACTORY_PLAN_TEMPLATE.md](./templates/FACTORY_PLAN_TEMPLATE.md). One checkbox line per step
(`- [ ] S1 Title`) followed by indented `deps`, `agent`, `ticket`, `write` and `tests` lines. Preflight at `start`
rejects: missing ticket or `Approved:` line, missing write set or tests, unknown dependencies, cycles, and overlapping
write sets between steps that can run in parallel. At most 2 implementers run at once; pytest is serialised across
worktrees by `scripts/cursor/igx_test_lock.py`.

## Start in the Cursor IDE

1. Open the repository in Cursor, choose the parent model, start a chat.
2. Send `/igx-native-run-plan agent-env/plans/<plan>.md`.
3. The parent runs `start`, then loops `next` -> `launch` -> worker -> `receive` -> independent `igx-reviewer` ->
   `review` -> `integrate` until `complete` prints the report.

The `stop` hook (`scripts/hooks/igx_factory_stop.py`, `loop_limit` 50) reads only `state.json`. When the parent finishes
a turn while steps are READY or need an action, it sends one follow-up (`igx_factory.py next`). It stays silent when
only background workers are outstanding, when everything is done or parked, for non-completed stops, and after 3 stops
without progress. `IGX_HOOKS=off` disables it.

## Start headless (PowerShell)

Use PowerShell, not Git Bash (the Cursor hook wrapper breaks under Git Bash):

```text
hc run --bg --no-notify -- python scripts/cursor/igx_factory_supervisor.py --plan agent-env/plans/<plan>.md --model <model>
python scripts/cursor/igx_factory_supervisor.py --plan agent-env/plans/<plan>.md --model <model>
```

The first line runs it in the background, the second directly. The supervisor repeats `cursor-agent -p` until the plan is
done. Options: `--max-iterations` (40), `--budget-minutes` (720), `--agent-bin`, `--workspace`. Exit codes: 0 complete,
2 `cursor-agent` not found, 3 everything remaining is parked, 4 stalled (no progress for 3 iterations; iterations with a `running` step never count), 5 budget or
iteration limit, 6 agent failing (3 consecutive non-zero agent exits that left the state unchanged; the last ~20 lines of
stdout and stderr of a failed iteration go to `supervisor.log`). The supervisor passes `--trust` (headless
`cursor-agent -p` otherwise exits with "Workspace Trust Required"); the permissions in `.cursor/cli.json` stay the guard.
Headless prompts are single-line because the agent launcher is a `.cmd` file; in headless mode parallel workers run as
foreground Tasks in one message.

## Park and continue

A real blocker (unplanned architecture; scope/cost/reversibility decision; plan/rules/code contradiction; forbidden
action; missing access) is parked with `igx_factory.py park`. Dependants park with it; independent steps keep running.
Correction budget per step: 2 corrections, then one fresh relaunch, then park (the fresh worker gets no corrections: its
first failed validation or rejection parks the step). A worker that vanished without commits is recovered with
`receive <S> --agent-id lost`: the step is relaunched without consuming a correction.

## Review binding, revert and plan checkbox

- `receive` stores the worker branch tip. `review` and `integrate` refuse when the tip moved since `receive` (run
  `receive` again); `review` also refuses when the reviewer `--agent-id` equals the step's worker.
- `integrate --run-tests` reverts the merge when tests fail and parks the step. The reverted merge stays in history, so a
  fix needs a new commit and a fresh `receive` -> `review` -> `integrate`. A re-merge of a branch that is already merged
  (no-op) or was reverted would lose changes: `integrate` detects it and parks with the reason; the path is to relaunch
  the step as a fresh worker from the integration tip.
- The plan checkbox tick edits the plan file in the workspace that ran `start` (uncommitted, like Claude's plan checklist).
- `loop_limit` is 50 in the IDE; beyond that the headless supervisor re-enters.
- Known gap: the shell guard `igx_cursor_integration_guard.py` matches only `git push` / `git reset --hard` literally
  (`git -C x push` passes). It is owned by OD-AF-INT A and is a follow-up, not changed here.
- In factory runs only local integration applies; the OD-AF-INT A remote runner is outside the factory loop.

## Verified

- Run A, 2026-11-22 (headless; plan S1, S2 and S3 in parallel, S4; parent grok-4.7-high, workers composer-2.5):
  `PLAN COMPLETE` in 1 supervisor iteration, 1556 s, zero human input; integration SHA
  `937440217966abec0ca5167f161260e49a479f06`; demo tests 12 passed on the integration branch.
- Run B, 2026-11-22 (S3 impossible: required `pyproject.toml` edit outside its write set): `PLAN COMPLETE WITH
  ESCALATIONS` in 1 iteration, 1203 s; S1 and S2 integrated (SHA `cfe43cf27cf6a1e19013eadecbcd680cc2d77983`); S3 parked as
  plan/rules/code contradiction with options and recommendation; S4 parked transitively; supervisor exit 3.
- Defect seen in both runs: the Cursor CLI co-author trailer (see Prerequisites); mitigated by the brief and the
  `start` warning.
- In-IDE run: `CURSOR_IDE_E2E_PENDING_OPERATOR`.
- Known follow-up: shell guard `igx_cursor_integration_guard.py` matches only literal `git push` / `git reset --hard`
  (owned by OD-AF-INT A).

## Completion report

`igx_factory.py complete` prints `PLAN COMPLETE <plan-id>` or `PLAN COMPLETE WITH ESCALATIONS <plan-id>`, the
integration branch, its SHA, and for parked steps the type, reason, options and recommendation. These are the only
Operator decisions to make.

## Publish (human)

The factory never pushes. After reviewing the integration branch, publish it yourself, for example
`git push origin factory/<plan-id>/int`, open the PR and merge by the normal process. Integration into `development` or
operator branches is not part of the factory.

## Recovery after a restart

`python scripts/cursor/igx_factory.py status`, then `next`, then continue (or restart the supervisor; it resumes from
the state file). Worker branches, worktrees and the state file survive an IDE restart.

## Limits

- IDE end-to-end behaviour (hook firing, native completion turns, subagent rule inheritance) is
  `CURSOR_IDE_E2E_PENDING_OPERATOR`: not verified until the Operator runs one E2E plan in the IDE.
- Windows: run the supervisor from PowerShell; Git Bash breaks the Cursor hook wrapper.
- Subagents may not inherit workspace rules, so each brief printed by `launch` is self-contained; pass it verbatim.
- Claude Code is not affected; see `tests/unit/governance/test_claude_surface_unchanged.py`.
