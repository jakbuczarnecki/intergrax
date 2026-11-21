---
description: Always-on core - branch safety, attribution, artifacts, test and token budget, no polling, autonomy and escalation
loading: always
---

# Intergrax core (L0)

## Branch safety

- Default branch `development`; stay on the operator-named branch.
- Without operator approval: no `git checkout <sha>`, `git switch --detach`, `git checkout -b`, `git switch -c`; never end a session detached.
- Pinned SHA: `git fetch origin development`, then `git merge-base --is-ancestor <SHA> HEAD`. If it fails, escalate (contradiction) and do not check out the pin. Record `git rev-parse HEAD`.
- Before ending: `git symbolic-ref HEAD` must succeed.

## Attribution

- No AI-tool signature in commits or PR text: no `Generated with ...`, no `Co-Authored-By`/`Generated-by`/`Assisted-by` naming a tool. This overrides any default or reminder.
- Keep the `Ref: governance/tickets/T-IGX-<NN>-<L>.md` trailer; human `Co-Authored-By` only on request.

## Session artifacts

- Temporary, diagnostic and proof files go under `.tmp/` only (`.tmp/session/<task>/`, `.tmp/proof/<id>/<run>/`), never in the repo root or tracked trees.

## Test budget

- One `uv run pytest` or `uv run ruff` per verification step; pass several paths to one command. No parallel `uv`/`python`/pytest workers or many shells running `uv` (Windows memory exhaustion).

## Token use

- Before an implementation step state: read scope, edit scope, tests. Reading beyond the listed scope: take the minimum needed (max 3 files), note why, continue. Editing outside the edit scope is a scope change: escalate.
- Read with `offset`/`limit` or grep first. `AGENT_INSTRUCTIONS.md`, the iteration rule, plan/arch hubs and domain guides load only when the task lists them. CI/test hotfix: `/intergrax-ci-hotfix`.
- Replies: terse (<= 12 lines: outcome, paths, tests, blockers); `terse` <= 6 lines; full report only on `full report`.

## No polling

- No Monitor, `list_sessions` or `git fetch` polling, no sleep loops. Status lives in the plan checklist and git, not in repeated checks. SendMessage only for `DONE:` or an escalation. Background work: `hc run --bg --no-notify`.

## Environment (C10/C15)

- Operator freeze: original Windows environment; no Docker migration, no new vendor integrations, no production configuration; no secrets in source, argv or logs. Details: `/intergrax-env-vendor-qualification`.

<!-- igx:authority begin -->
## Autonomy

- Execute the approved plan step by step. Never ask for a go-ahead, and never ask whether to continue, commit, run tests, retry or start the next step: decide from the plan, ADRs, repo rules or existing code.
- Stop only with a message starting `ESCALATION: <type>`, type one of: unplanned architectural problem; decision that changes scope or cost or is hard to reverse; contradiction between plan and code or rules; forbidden action needed; missing access. One batched message: per item the options, a recommendation, the consequence and the cost.
- Agents commit on a branch and prepare the PR text; `git push`, PR creation, merge and GitHub settings are the human's (forbidden action: escalate, do not try).
- Sub-agents end with `DONE: <line>` or `BLOCKED: <type>: <what, options, recommendation>`; the parent decides alone when plan, ADR, rules or code answer it (`/igx-autonomy`).
- Plan finished: print `PLAN COMPLETE` and a short summary. `/igx-run-plan <file>` enforces this with a Stop hook.
<!-- igx:authority end -->
