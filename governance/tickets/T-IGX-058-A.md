# T-IGX-058-A — Cursor shell guard: cover git/gh global-option and destructive variants

## Scope

Harden the Cursor-only `beforeShellExecution` hook `scripts/hooks/igx_cursor_integration_guard.py`
(OD-AF-INT A, owned by `T-IGX-055-C`; fail-closed, registered in `.cursor/hooks.json`). Its deny regexes match only
literal `git push`, `gh pr create|merge`, `gh api -X <mutating>` and `git reset --hard`, so global-option forms pass:
`git -C <path> push`, `git --git-dir=x push`, `git.exe -c k=v push`, `"C:\Program Files\Git\bin\git.exe" push`,
`git -C x reset --hard`, `git reset -q --hard`. Found by the independent review of the Cursor software factory
(`T-IGX-057-A`, documented as a known gap in `docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md`).

Planned change (regex-based, same `decide()` contract and messages):

1. Shared prefix `git(.exe)?` (optionally closed by a quote) followed by any number of global options, each with an
   optional value token (bare, `"quoted"` or `'quoted'`), e.g. `-C <path>`, `-c k=v`, `--git-dir=x`, `--no-pager`.
   The executable name stays case-insensitive; flags such as `-D` versus `-d` are matched case-sensitively.
2. Deny after that prefix: `push`, `send-pack`, `http-push`, `subtree ... push`; `reset ... --hard`;
   `branch` with `-D` or `--delete`/`-d` combined with `--force`/`-f`; `worktree remove ... --force`/`-f`;
   `update-ref -d` / `update-ref --delete` (ref deletion bypassing `branch -D`).
3. `gh(.exe)?` with global options: `pr ... create|merge`; `api` with `-X`/`--method` (also `-XPOST`, `--method=POST`)
   for POST/PUT/PATCH/DELETE; `api` with field/input flags (`-f`, `-F`, `--field`, `--raw-field`, `--input`), which make
   `gh api` default to POST, except a `graphql` call without the word `mutation`.
4. Unchanged allows: local `git merge`, `git commit`, `git worktree add`, `git branch <name>`, `git branch -d <merged>`,
   `git worktree remove <path>` (no force), `git status|diff|log|show|rev-parse`, `gh pr view`, the guarded runner
   `python scripts/cursor/igx_integrator.py ...` and `python scripts/cursor/igx_factory.py ...`.

Residual risk (documented, not closed here): aliases defined earlier (`git config alias.p push` then `git p`), scripts
that call git internally, and shell indirection (`$(...)`, variables). `.cursor/cli.json` prefix rules and server-side
branch protection remain the other layers.

## Plan digest

Plan digest: n/a (no separate plan file; the scope above is the plan)

## Approved

Approved: operator @ 2026-11-22T15:06:16+01:00

Approval source: explicit operator instruction `Approved: T-IGX-058-A` in the implementing session's chat on
2026-11-22, scope as written above. The timestamp is the recorded entry time, not the exact time the operator typed it.
This is not an independently signed IIAC verdict.

## Write set

```text
scripts/hooks/igx_cursor_integration_guard.py
tests/unit/governance/test_cursor_integration_guard.py
governance/tickets/T-IGX-058-A.md
```

## Test contract

```text
uv run pytest tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_subagent_guard.py -q
uv run python scripts/governance/sync_agent_env.py --check
```

## Out of scope

`.claude/**`, `CLAUDE.md`, `agent-env/rules|skills|agents`, `scripts/hooks/igx_stop_guard.py`,
`governance/agent-control-plane/**`, `governance/TRUSTED_POLICY_PIN.yaml`, `.cursor/hooks.json`, `.cursor/cli.json`,
`scripts/cursor/**`, the `T-IGX-057-A` factory branch and its docs, `intergrax/`, `agents/`, `applications/`, push, PR,
merge.

## Checkpoint

2026-11-22, implementer session:

- `uv run pytest tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_subagent_guard.py -q`: 162 passed.
- `uv run python scripts/governance/sync_agent_env.py --check`: OK (70 adapters + CLAUDE.md block).
- Implementation note: `gh pr new` is denied with `create` (it is the gh alias for `gh pr create`). Option values never
  start with `-`, so long flag lists cannot cause exponential regex backtracking (tested).
- Not covered (follow-up candidates): other remote `gh pr` mutations (`close`, `edit`, `review`, `ready`, `comment`)
  are only blocked by the `.cursor/cli.json` `Shell(gh)` prefix rule; `igx_gov ticket` not run (no push in this ticket).

## Closeout

2026-11-22: `uv run python -m scripts.governance.igx_gov ticket` PASS before push; merged into `development` via
PR #750 (merge commit `7bd4558b5`). Final readiness audit on that commit: 317 passed (factory contract plus this
ticket's guard tests). The `gh pr` mutation follow-ups above remain covered only by `.cursor/cli.json`.

## Lifecycle state

closed
