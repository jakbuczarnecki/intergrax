---
name: igx-native-run-plan
description: Run an approved Intergrax plan to completion in Cursor with the software factory (igx_factory.py), Composer subagents and local integration.
---

# Cursor software factory runbook

Cursor-only replacement for `/igx-run-plan`; Claude Code keeps its own flow unchanged. Protocol summary: `.cursor/rules/igx-factory.mdc`. Operator guide: `docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md`.

## Preconditions

- The plan is in factory format (`docs/project/technical/devos/templates/FACTORY_PLAN_TEMPLATE.md`): one `- [ ] <ID> <title>` per step with `deps`, `agent`, `ticket`, `write`, `tests` lines.
- Every step ticket exists and has an `Approved:` line. `start` refuses otherwise (preflight).
- Run from the repo checkout; never ask the Operator for a go-ahead.

## Commands (always `python scripts/cursor/igx_factory.py ...`)

```text
start <plan.md>                         # preflight, integration branch factory/<plan-id>/int, state file
next                                    # JSON: ready, running, awaiting_background, pending_actions, parked, done
launch <S> [--background]               # creates worker worktree, prints the brief
receive <S> --agent-id <id>             # validates commits, write set, Ref trailer, test log
review <S> --agent-id <id> --verdict ACCEPT|REJECT [--findings "..."]
integrate <S> [--run-tests]             # merge worker branch into the integration branch
park <S> --type "<escalation type>" --reason "..." [--options "..."] [--recommendation "..."]
status                                  # full state; use after a restart
complete                                # prints the completion report
```

## Loop

1. `start plans/x.md` once, then `next`.
2. For each entry in `ready` (max 2 implementers): `launch <S>` (add `--background` when running two), then start a Task with the printed brief verbatim, `subagent_type` = the entry's agent, `model` `composer-2.5`.
3. Worker returns `DONE:` -> `receive <S> --agent-id <id>`. A `DONE:` is never certification: `receive` validates the work.
4. `receive` ok -> start a separate `igx-reviewer` Task (brief: step, write set, `git diff` of the worker branch). Record the verdict with `review`.
5. `accepted` -> Task `igx-integrator` to run `integrate <S> --run-tests` (local merge only). `next`.
6. Repeat until `next` has `done: true`, then run `complete` and relay its output exactly.

## Worker outcomes

- `DONE:` -> receive. `receive` failure or reviewer `REJECT` -> state moves to `correction` (up to 2) with `correction_brief`/findings: `launch <S>` again and send the new brief. After the budget the step goes to `relaunch` (one fresh worker); the fresh worker's first failure parks the step. `review` needs a reviewer id different from the worker and an unmoved branch tip (else `receive` again). After a revert-on-failure the fix needs a new commit and a fresh `receive`/`review`/`integrate`; `integrate` parks an already merged or reverted branch, so relaunch it as a fresh worker.
- `BLOCKED: <type>: ...` -> decide alone if plan, ADR, rules or code answer it, then relaunch with the answer in the plan or ticket scope. Otherwise `park <S> --type ...` and continue with other READY steps. Dependants park automatically.
- Escalation types: unplanned architecture; scope/cost/reversibility decision; plan/rules/code contradiction; forbidden action; missing access.

## Turn discipline

- Never end a turn while `next` shows `ready` or `pending_actions`, unless background workers are outstanding (their completion turn resumes you). The `stop` hook `igx_factory_stop.py` re-prompts when you stop too early.
- Only the Operator publishes: no `git push`, PR, remote merge or integration into `development`. `complete` prints the publish command as text.

## Restart recovery

After a lost session or restart: `python scripts/cursor/igx_factory.py status`, then `next`, then continue the loop. A step left `running` whose worker is gone: run `receive <S> --agent-id lost`; if the branch has no commits it relaunches the step without consuming a correction; otherwise it validates the branch and moves the step to `received` or `correction`/`relaunch`.

## Headless option

From PowerShell (not Git Bash):

```text
python scripts/cursor/igx_factory_supervisor.py --plan agent-env/plans/x.md --model <parent-model>
```

Loops `cursor-agent -p` until done. Exit codes: 0 complete, 2 no `cursor-agent`, 3 all remaining parked, 4 stalled, 5 budget or iteration limit.
