# Intergrax - Antigravity Workspace Rules

This file is the **Antigravity-native** counterpart to `CLAUDE.md` (loaded automatically alongside root `AGENTS.md`).

- **Single source of truth (SSOT):** canonical rules, skills, and plans live in `agent-env/`; canonical role authority lives in `governance/agent-control-plane/role_registry.yaml`.
- **Tool isolation:** `.agents/` and `GEMINI.md` are the Antigravity adapter layer. Never modify `CLAUDE.md`, `.claude/`, `.cursor/`, or hash-pinned governance surfaces unless explicitly instructed with a ticket.
- **Plans directory:** save plans under `agent-env/plans/` (shared with Claude Code and Cursor).

## Core Invariants (L0 — mirrors `agent-env/rules/intergrax-core.md` & `intergrax-english-only-commits.md`)

### Branch safety
- Default branch is `development`; stay on the operator-named branch.
- Without operator approval: never run `git checkout <sha>`, `git switch --detach`, `git checkout -b`, or `git switch -c` on the main checkout; never leave the main checkout detached.
- Approved multi-step implementation uses isolated workspaces (`Workspace: "share"` or `"branch"` in `invoke_subagent`) or `git worktree add` outside the repo per `/igx-worktree`. Never alter another worktree or the main checkout from a worker.
- Pinned SHA: fetch `development`; `git merge-base --is-ancestor <SHA> HEAD` must pass, else escalate; record `HEAD`.
- Before finishing: `git symbolic-ref HEAD` must succeed on the main checkout.

### English-only VCS & GitHub · No AI attribution
- `git commit` subject/body, PR titles/bodies, `gh pr create` text, and merge/PR commentary: **English only** (operator chat language is separate).
- **No AI-tool signature** in commits or PR text: no `Generated with ...`, no `Co-Authored-By` / `Generated-by` / `Assisted-by` naming any AI tool.
- Every commit touching `intergrax/`, `agents/`, `applications/`, or `scripts/` requires the ticket trailer `Ref: governance/tickets/T-IGX-<NN>-<L>.md`.

### Session artifacts · Test budget · Environment freeze (C10/C15)
- Temporary, diagnostic, and proof files go under `.tmp/` only (`.tmp/session/<task>/`, `.tmp/proof/<id>/<run>/`), never in the repo root or tracked trees.
- **Serial verification only:** run one `uv run pytest` or `uv run ruff` command per verification step (pass multiple paths to one command). Never spawn parallel `uv`/`python`/`pytest` workers (Windows memory exhaustion).
- **Environment freeze (C10/C15):** preserve the original Windows environment; no Docker migration, no new vendor integrations, no production configuration changes; never put secrets in source, argv, or logs (details: `agent-env/rules/intergrax-env-vendor-qualification.md`).

### Token use & read discipline
- Before any implementation step, state: `read scope`, `edit scope`, `tests`.
- Use `StartLine`/`EndLine` on `read_file`, or narrow search first. Reading beyond the listed scope: take the minimum needed (max 3 extra files), note why, and continue. Editing outside `edit scope` is a scope change: escalate.
- Never recurse searches into `.claude/worktrees/`, `.tmp/`, `.venv/`, or `_wt/`.
- Load `docs/project/technical/guides/AGENT_INSTRUCTIONS.md`, domain hubs, and audit protocols only when the task requires them.
- Keep replies concise (<= 12 lines by default; `terse` <= 6; full report only on request).
- **No polling:** never run sleep loops or poll background tasks/subagents; rely on Antigravity's reactive wakeup.

### Autonomy & Escalation
- Execute approved plans step by step without routine go-ahead questions; decide from the plan, ADRs, repo rules, and existing code.
- Stop only with `ESCALATION: <type>` (unplanned architecture; scope/cost/reversibility decision; plan/rules/code contradiction; forbidden action; missing access) in one batched message with options, recommendation, consequence, and cost.
- **Commit SSOT:** `may_commit` in `governance/agent-control-plane/role_registry.yaml` is authoritative. Only `implementer` (`igx-implementer`) commits on its worktree branch with `Ref:`. Top-level `orchestrator` has `may_commit: false`. `git push`, PR creation, merge, branch integration, and GitHub settings stay human unless explicitly authorized (forbidden action: escalate, do not try).
- Subagents end with `DONE: <one line>` or `BLOCKED: <type>: <what, options, recommendation>`.
- When an approved plan finishes with no open steps, print `PLAN COMPLETE` and a short summary.

## Antigravity Tool Vocabulary Mapping

Canonical files in `agent-env/` use Claude/Cursor tool names. Translate them as follows:

| Canonical (`agent-env/`) | Antigravity native equivalent |
|--------------------------|-------------------------------|
| `Read` (`offset`/`limit`) | `read_file` (`StartLine`/`EndLine`) |
| `Edit` / `Write` | `edit_file` / `write_file` |
| `Bash` | `run_command` (PowerShell on Windows) |
| `Grep` / `Glob` | Targeted `rg` / `git grep` / `Get-ChildItem` via `run_command` (scoped path only) |
| `Agent` / `Task` tool | `invoke_subagent` (`TypeName`, `Role`, `Prompt`, `Workspace`) |
| `isolation: worktree` | `Workspace: "share"` (or `"branch"`) on `invoke_subagent`, or `/igx-worktree` |
| `general-purpose` (denied) | Do not delegate generic implementation to `self`; use typed `igx-*` subagents |
| `@file` / `/<skill>` | `@` rule/file mention / `/<skill>` slash command |

## Role & Subagent Separation (Delivery DevOS)

Default top-level session role: **`orchestrator`** (`may_commit: false`). Route operator intent cleanly across roles (SSOT: `governance/agent-control-plane/role_registry.yaml`):

| Phase / Intent | DevOS Role | Antigravity Agent / Session | Workspace | `may_commit` | Max Parallel |
|----------------|------------|-----------------------------|-----------|--------------|--------------|
| Route / coordinate | `orchestrator` | Main conversation | main | `false` | 1 |
| Codebase / docs discovery | `architect-planner` | `igx-researcher` (or built-in `research`) | `inherit` | `false` | 3 |
| Architecture / plan / ticket scope | `architect-planner` | `igx-architect` | `inherit` | `false` | 1 |
| Scoped product implementation | `implementer` | `igx-implementer` | `share` / `branch` | `true` (own branch + `Ref:`) | 2 (serial `uv`/`pytest`) |
| Verification / Test contract | `tester` | `igx-tester` | `inherit` / worker wt | `false` | 1 (serial `uv`/`pytest`) |
| Code review handoff | `reviewer` | `igx-reviewer` (no shell) | `inherit` | `false` | 1 |
| PR creation (when authorized) | `delivery-closure` | Skill `igx-devos-github-pr-workflow` | main | `false` | 1 |
| G6 / IIAC independent audit | `independent-quality-auditor` | **Separate new conversation only** | — | `false` | 1 |

**Hard independence boundary:** No implementer, tester, reviewer, or subagent may declare G6 PASS, IIAC PASS, certification, or FRZ closure.

## On-Demand Rules & Skills

- **Canonical skills (auto-discovered via `.agents/skills.json` -> `agent-env/skills/`):** all `igx-devos-*`, `sfx-*`, and `igx-worktree`.
- **Antigravity-adapted skills (`.agents/skills/`):** `/igx-autonomy`, `/igx-run-plan`, `/intergrax-iteration`, `/intergrax-ci-hotfix`, `/intergrax-hep-step`, `/intergrax-micro-implement`, `/intergrax-independent-audit`.
- **Modular rules (`.agents/rules/`):** auto-triggered (`intergrax-agent-roles`, `intergrax-independent-audit`, `intergrax-env-vendor-qualification`, `intergrax-scenario-factory`) and manual `@`-rules (`intergrax-iteration`, `intergrax-ci-hotfix`, `intergrax-hep-step`, `intergrax-micro-implement`).
- **Lifecycle safety hooks (`.agents/hooks.json`):** `.env` read deny, large un-ranged file read & shell dump guard, destructive git / bypass deny, branch-switch & push/PR ask gate, typed subagent guard, and plan-driven `Stop` guard. Disable temporarily with `IGX_HOOKS=off`.
