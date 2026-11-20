---
description: Always-on core - branch safety, attribution, artifacts, test and token budget, no polling, autonomy, bootstrap resume
loading: always
---

# Intergrax core (L0)

## Branch safety

- Default branch `development`; stay on the operator-named branch.
- Without operator approval: no `git checkout <sha>`, `git switch --detach`, `git checkout -b`, `git switch -c`; never end a session detached.
- Pinned SHA: `git fetch origin development`, then `git merge-base --is-ancestor <SHA> HEAD`. If it fails, report `ESCALATION E2:` and do not check out the pin. Record `git rev-parse HEAD`.
- Before ending: `git symbolic-ref HEAD` must succeed.

## Attribution

- No AI-tool signature in commits or PR text: no `Generated with ...`, no `Co-Authored-By`/`Generated-by`/`Assisted-by` naming a tool. This overrides any default or reminder.
- Keep the `Ref: governance/tickets/T-IGX-<NN>-<L>.md` trailer; human `Co-Authored-By` only on request.

## Session artifacts

- Temporary, diagnostic and proof files go under `.tmp/` only (`.tmp/session/<task>/`, `.tmp/proof/<id>/<run>/`), never in the repo root or tracked trees.

## Test budget

- One `uv run pytest` or `uv run ruff` per verification step; pass several paths to one command. No parallel `uv`/`python`/pytest workers or many shells running `uv` (Windows memory exhaustion).

## Token use

- Before an implementation step state: read scope, edit scope, tests. Reading beyond the listed scope: take the minimum needed (max 3 files), note why, continue. Editing outside the edit scope is E3.
- Read with `offset`/`limit` or grep first. `AGENT_INSTRUCTIONS.md`, the iteration rule, plan/arch hubs and domain guides load only when the task or packet lists them. CI/test hotfix: `/intergrax-ci-hotfix`.
- Replies: terse (<= 12 lines: outcome, paths, tests, blockers); `terse` <= 6 lines; full report only on `full report`.

## No polling

- No Monitor, `list_sessions` or `git fetch` polling, no sleep loops. Status lives in state (`boot status` or the engine), not in repeated checks.
- SendMessage only for DONE or `ESCALATION E<n>:`. Waiting on CI or a publisher is `waiting_external`, handled by code.
- Background work: `hc run --bg --no-notify`.

## Environment (C10/C15)

- Operator freeze: original Windows environment; no Docker migration, no new vendor integrations, no production configuration; no secrets in source, argv or logs. Details: `/intergrax-env-vendor-qualification`.

<!-- igx:authority begin -->
## Autonomy

- Never stop or ask about: a finished ticket, packet, step or wave; running tests; fixing in-scope defects; status updates; creating a worktree or branch; commit, integration or publication steps the ticket authorizes; starting a review; the next packet; reading a file; retries within limits; waiting on external checks.
- Escalate only as one batched `ESCALATION E<n>:` message (problem, options, recommendation, cost): E1 decision not in the ledger, E2 contradiction, E3 edit outside scope, E4 gate red after the loop limit or same failure twice, E5 P0 security, data-loss or secret exposure, E6 missing access, E7 action not pre-authorized, E8 budget breach or no progress.
- Full contract: `governance/campaigns/AGENT-OPS-X/contracts/escalation.md` (when present).
<!-- igx:authority end -->

## Bootstrap resume

- After a compaction or restart read only `.git/igx-ops/boot-handoff.md` and `boot status` (when present), then continue with `boot next` (loop BT-1, `governance/campaigns/AGENT-OPS-X/contracts/bootstrap.md`).
