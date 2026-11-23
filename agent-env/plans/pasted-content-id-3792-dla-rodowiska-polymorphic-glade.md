# CURSOR-FACTORY - autonomous Grok/Composer software factory for Cursor

## Context

Operator goal: give one Cursor chat (strong parent model, e.g. Grok) an approved multi-step plan and get the
result back only after it has run the whole plan by itself through Composer subagents. Claude Code does this
today. Cursor stalls at nearly every step.

The audit (2026-11-22) shows the subagent mechanism is not the main cause. Cursor's native background-Task
completion turn works (`agent-env/plans/cursor-orch-corrective.md` F04). The stalls are designed into the rules
and the integration path:

1. Every wave boundary needs push/PR/merge. The always-on `intergrax-core.mdc:46` makes this human-only, and
   `od-af-int-a.mdc` is not always-on (uncommitted `alwaysApply: false`). Result: an ESCALATION at every wave.
2. `igx_integrator.py merge` needs green CI, but polling is forbidden and there is no CI completion event.
3. The packet `base_sha` pin blocks the second of two parallel PRs for good.
4. Review evidence must be a JSON file, but the reviewer is readonly. Checkpoints must go into the plan, but the
   skill forbids `agent-env/**` edits. Implementers are not isolated automatically (`isolation` is dropped for
   Cursor). Two implementers run in parallel, but only one pytest may run at a time.
5. Competing protocols: generated `intergrax-core.mdc` and `igx-autonomy` still describe the Stop-hook/`WAITING:`
   flow, and `cursor-native-orchestration.mdc` tries to override them.
6. Nothing resumes Grok when it ends a turn early while READY work exists (no `stop` hook). Nothing survives an
   IDE restart.

Operator decisions (2026-11-22, this planning session):
- **Integration: local.** Workers commit in their own worktrees. The integrator merges them locally into a
  per-plan integration branch. No push, PR or merge to remote inside the loop. The Operator publishes at the end
  (Claude Code parity).
- **Resume: both.** A Cursor `stop` hook driven by a state file (no transcript parsing), plus a headless
  `cursor-agent` supervisor for long unattended runs.
- **Escalation: park and continue.** A blocked step and its dependants are parked. Independent steps keep
  running. All escalations are reported in one batch at the end.
- **Approvals: preflight.** The ticket grammar is unchanged. The run refuses to start unless every ticket in the
  plan already carries an `Approved:` line.

Hard constraint: Claude Code behaviour and files stay byte-identical. That covers `.claude/**`, `CLAUDE.md`,
`agent-env/**`, `scripts/hooks/igx_stop_guard.py` and `governance/agent-control-plane/*.yaml`. Every change is
Cursor-only: hand-written `.cursor/*` files, new `scripts/cursor/*` and `scripts/hooks/igx_factory_stop.py`, and
Cursor-only branches in `sync_agent_env.py`.

## Delivery frame

- Ticket `governance/tickets/T-IGX-056-A.md`, or the next free number. Scaffold it with `/igx-devos-ticket-scaffold`.
  Record the approval from this plan's acceptance in the strict grammar (Delegation, Delegate session,
  Implementer session).
- Worktree per `/igx-worktree`: branch `feat/cursor-software-factory` from `development`, path under
  `D:/Projekty/_wt/intergrax/`. The main checkout's uncommitted `od-af-int-a.mdc` and
  `EQO_EXECUTION_CHECKPOINT.json` edits stay untouched.
- Commits carry `Ref:` and no AI attribution. No push or PR: the Operator publishes.

## Checklist

- [ ] F0 Ticket, worktree, baseline (record base SHA, `sync --check`, current tests green)
- [ ] F1 Factory plan format + `scripts/cursor/igx_factory.py` state machine
- [ ] F2 Local integration + test lock + worker briefs
- [ ] F3 State-file `stop` hook for the Cursor IDE
- [ ] F4 Headless supervisor + `.cursor/cli.json` permissions
- [ ] F5 Cursor rule, skill, agent and generator cleanup (single protocol)
- [ ] F6 Tests (factory + Claude-surface regression) and docs
- [ ] F7 Headless E2E demo run, independent review, commit set

## F1 Factory plan format and state machine

The plan is a Markdown file. Checkbox lines are the single source of truth, so the format stays compatible with
the Claude `igx_stop_guard.py` parser. Each step's metadata goes in indented sub-bullets:

```text
- [ ] S2 Add parser for X
  - deps: S1
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-060-A.md
  - write: intergrax/foo/**, tests/unit/foo/**
  - tests: tests/unit/foo
```

`scripts/cursor/igx_factory.py` is stdlib only. State lives in `.tmp/factory/<plan-id>/state.json`, and every
write is atomic. Grok calls it instead of reasoning about state:

| Command | Effect |
|---------|--------|
| `start <plan>` | Preflight: plan parses, DAG has no cycle, every ticket exists with `Approved:`, write sets do not overlap between parallel-eligible steps, base SHA recorded. Creates the integration branch `factory/<plan-id>/int` in its own worktree. On any failure it refuses to start (one report, no mid-run stop). |
| `next` | JSON list of READY steps (deps integrated, not parked or running), capped at 2 implementers. |
| `launch <step> [--background]` | Creates the worker worktree `factory/<plan-id>/<step>` from the current integration tip. Marks the step running. Prints the full Task brief. |
| `receive <step> --agent-id ID` | Validates the worker against git: branch moved, diff stays inside the write set, `Ref:` trailer present, no AI attribution, test log present. On failure it returns a correction brief. Retry budget: 2 corrections, then 1 fresh worker, then park. |
| `review <step> --agent-id ID --verdict ACCEPT/REJECT [--findings F]` | Records the readonly reviewer's verdict and agent id. No fabricated JSON proof. REJECT re-enters correction. |
| `integrate <step>` | Run by `igx-integrator` only: `git merge --no-ff` of the worker branch into the integration branch, inside the integration worktree. The integration test command runs through the lock. Ticks the checkbox in the plan file. |
| `park <step> --type T --reason R` | Parks the step and all its transitive dependants. Records options and a recommendation. |
| `status` / `complete` | `complete` prints `PLAN COMPLETE` (everything integrated) or `PLAN COMPLETE WITH ESCALATIONS` followed by one batched `ESCALATION:` block for parked steps. It also prints the integration branch, SHA and the Operator publish command. |

Decision policy is encoded in the brief, not in prose rules. Grok decides alone when the plan, an ADR, the rules
or the code answer the question. It parks a step only for the five escalation types from
`agent-env/skills/igx-autonomy/SKILL.md:27-37`.

## F2 Local integration, test lock, worker briefs

- `scripts/cursor/igx_test_lock.py -- <cmd>`: an OS file lock (`msvcrt`) on `.tmp/factory/pytest.lock`. It blocks
  inside the process, so the agent does no polling. This makes "one pytest at a time" hold for 2 parallel
  implementers.
- Briefs printed by `launch` contain the rules that subagents may not inherit (forum: subagents do not reliably get
  rules):
  - the worktree path
  - the write set
  - the ticket
  - the test command wrapped in the lock
  - commit with the `Ref:` trailer
  - never push, reset `--hard`, or touch the main checkout
  - the last line must be `DONE: <branch> <sha> <tests>` or `BLOCKED: <type>: ...`
- `.cursor/agents/igx-integrator.md` (hand-written) gains the local `integrate` duty. Remote OD-AF-INT A stays as
  is but is outside the factory loop. `od-af-int-a.mdc` stays `alwaysApply: false`, adopting the Operator's
  pending edit, and `test_agent_env_sync.py:73` is updated to match.
- Reuse: `scripts/hooks/igx_cursor_integration_guard.py` keeps blocking push and `reset --hard`. A local merge is
  allowed. Verify that its relative `python scripts/hooks/...` path resolves from worktrees. If it does not, make
  the hook command root-independent the way `test_hook_commands.py` pins it for Claude.

## F3 Cursor `stop` hook (IDE)

New file `scripts/hooks/igx_factory_stop.py`, registered in `.cursor/hooks.json` as `stop` with an explicit
`loop_limit` of 50 and fail-open. It reads only the payload's `conversation_id` and `status`, plus `state.json`.
It never reads the transcript.

| Situation | Result |
|-----------|--------|
| No running factory, or the run belongs to another conversation | allow. Ownership is claimed on the first stop after `start`. |
| `status` is not `completed`, or the factory is complete | allow |
| Background workers launched and not yet received | allow (the native completion turn resumes the parent) |
| READY steps or received-but-unprocessed steps exist | `followup_message`: "FACTORY CONTINUE: run `python scripts/cursor/igx_factory.py next`; READY: S3, S4" |
| Same progress signature for 3 stops | mark stalled, allow (the supervisor or Operator takes over) |

The old `igx_cursor_stop.py` stays unregistered and untouched. Update the Cursor tests that pin "no `stop` key"
(`test_cursor_native_orchestration.py:16-18`, `test_subagent_guard.py:77`) so they pin "the `stop` hook is exactly
`igx_factory_stop.py`". Keep every Claude Stop-hook assertion.

## F4 Headless supervisor

New file `scripts/cursor/igx_factory_supervisor.py`. Launch it from PowerShell; Git Bash breaks the hook wrapper
(corrective plan, line 90). Run in the background with `hc run --bg --no-notify`. The loop:

1. Run `cursor-agent -p --model <parent> --output-format json --workspace <checkout> [--resume <chatId>] "<factory prompt>"`.
   Headless waits for delegated subagents before it exits.
2. After each exit, read `state.json`.
3. Stop on complete, all-remaining-parked, stalled, `--max-iterations` or a wall-clock budget.
4. Otherwise resume the same chat with "FACTORY CONTINUE".

Logs go to `.tmp/factory/<plan-id>/supervisor.log`. The parent model id is a required flag. No secrets go in argv
(the CLI uses the existing login).

`.cursor/cli.json` allowlists only what the factory needs:
- `Shell(git ...)` except push and reset; denies win
- `Shell(uv run ...)` and `Shell(python scripts/cursor/...)`
- `Write(**)` inside the workspace

Use it instead of `--force`.

## F5 One protocol for Cursor (no Claude change)

- Replace `cursor-native-orchestration.mdc` with an always-on `.cursor/rules/igx-factory.mdc`. Contents:
  - the factory contract
  - never end a turn while `next` is non-empty, unless background workers are outstanding
  - park and continue
  - local integration only
  - integration into `development` stays human
  - `PLAN COMPLETE` only from `igx_factory.py complete`
- Rewrite `.cursor/skills/igx-native-run-plan/SKILL.md` as the factory runbook. The generated `igx-run-plan` alias
  keeps pointing to it.
- `sync_agent_env.py`, Cursor branches only:
  - The Cursor `intergrax-core.mdc` replaces the "`/igx-run-plan` enforces it via Stop hook" sentence with a pointer
    to `igx-factory.mdc`.
  - The Cursor `igx-autonomy` adapter is a native stub, like the existing `igx-run-plan` alias, so Grok never
    follows the canonical `WAITING:` protocol.
  - Claude outputs are unchanged, proven by `--check` plus the new hash test (F6).
- Fix `igx_subagent_guard.py:24` so its role list names `igx-integrator`. This is message text only.

## F6 Tests and docs

New tests in `tests/unit/cursor/`:
- `test_igx_factory.py`: parse, DAG, READY cap, park propagation, preflight refusals. Receive validation,
  integrate and complete run on a temp git repo.
- `test_igx_factory_stop.py`: the decision table.
- `test_igx_test_lock.py`
- `test_igx_factory_supervisor.py`: runs against a fake `cursor-agent` executable.

New test `tests/unit/governance/test_claude_surface_unchanged.py`: the git tree hashes of `.claude/`, `CLAUDE.md`,
`agent-env/`, `scripts/hooks/igx_stop_guard.py` and `governance/agent-control-plane/` equal the F0 base SHA.

Existing guards stay green:
- `test_stop_guard.py`
- `test_hook_commands.py`
- `test_agent_env_sync.py`
- `test_agent_role_routing.py`
- `test_cursor_integrator.py`
- `test_cursor_stop_guard.py`

Docs:
- `docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md`: Operator runbook (write a plan, start in the IDE or
  with the supervisor, read the report, publish).
- `docs/project/technical/devos/templates/FACTORY_PLAN_TEMPLATE.md`
- Update the Cursor column of `docs/project/technical/guides/AGENT_AUTONOMY.md`.
- Mark `cursor-orch-corrective.md` and `CURSOR_OD_AF_INT_A.md` loop sections as superseded, by pointer only.

## Verification

1. Run one pytest command covering every test path listed in F6.
2. Run `uv run ruff check` and `uv run ruff format --check` on the touched paths, then
   `uv run python scripts/governance/sync_agent_env.py --check` and
   `uv run python -m scripts.governance.igx_gov ticket`.
3. Run a headless E2E with the supervisor (Grok parent, `composer-2.5` workers). Use a demo plan on a throwaway
   scratch branch that is never merged: S1, then S2 and S3 in parallel, then S4. It must reach `PLAN COMPLETE` with
   zero human input.
4. Run variant 3b, where S3 is made impossible. It must end with `PLAN COMPLETE WITH ESCALATIONS`: S3 and S4
   parked, S1 and S2 integrated, one batched escalation.
5. Store proof under `.tmp/proof/cursor-factory/<run>/` (state.json, supervisor.log, integration SHA).
6. The Operator runs the same demo inside the Cursor IDE with `/igx-native-run-plan`. Until then the status is
   `CURSOR_IDE_E2E_PENDING_OPERATOR`. The agent cannot drive the IDE.
7. Before the commit set, run an independent `igx-reviewer` pass over the diff against this plan.
