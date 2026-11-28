# Intergrax Codex adapter policy

This policy applies to Codex only. `AGENTS.md`, `agent-env/`, and the Delivery
DevOS registries remain authoritative. Do not apply Cursor direct-execution
exceptions, Claude model names, or Antigravity tools in Codex.

## Loading and skills

- Root `AGENTS.override.md` is generated from root `AGENTS.md`, the shared
  always-on rules, and this policy. Codex does not expand Claude `@file` imports.
- Discover Codex workflows through the local `intergrax-codex` plugin. Every
  canonical skill/rule has a `codex-<canonical-id>` adapter. For example, use
  `$intergrax-codex:codex-intergrax-hep-step` or
  `$intergrax-codex:codex-igx-devos-ticket-driven-work` (native plugin namespace).
- The `.agents/skills` adapters belong to Antigravity. Their overlapping names
  are disabled in Codex configuration only. Do not invoke them from Codex.
- Read canonical workflow bodies only when selected. Preserve `explicit` and
  `disable-model-invocation` modes; never activate them from keyword matching.
- Resolve canonical paths relative to the Git worktree root, even when a plugin
  skill is loaded from the user's plugin cache or the working directory is nested.
- Translate canonical `Read`/`Grep`/`Glob` into scoped file reads and `rg`;
  `Edit`/`Write` into `apply_patch`; `Bash` into PowerShell `exec_command`;
  `Agent`/`Task` into the available Codex collaboration tools. A canonical
  `/<skill>` means `$intergrax-codex:codex-<skill>`. This changes vocabulary,
  never authority.

## Roles and orchestration

- The parent is `orchestrator` (`may_commit: false`). Use typed workers only
  when delegation is requested by the operator or an applicable workflow.
- Worker roles and concurrency come from
  `governance/agent-control-plane/role_registry.yaml`. The Codex total cap is
  three children; respect the lower per-role cap too.
- `igx-researcher`: architect-planner, read-only, max 3; no shell.
- `igx-architect`: architect-planner, max 1; writes only named plans/tickets.
- `igx-implementer`: implementer, max 2; approved isolated worktree only;
  commits only its own branch with `Ref:`. Spawning a worker does not isolate
  files: arrange and verify the worktree before giving it write access.
- `ci-investigator`: tester, max 1; serial Test contract execution, no commits
  or product edits. This uses the existing tester map, without adding authority.
- `igx-reviewer`: reviewer, read-only, max 1; no shell; never G6/IIAC PASS.
- Every worker receives the ticket, read/edit scopes, acceptance criteria,
  worktree path, test contract, and the required `DONE:`/`BLOCKED:` report.
- Workers never spawn workers. Inherit the parent model and reasoning settings;
  use an explicit model only when the operator authorizes it. Do not translate
  `haiku`/`sonnet`/`opus` or Composer names into guessed OpenAI model ids.
- Use completion notifications or the available bounded collaboration wait.
  Do not poll, sleep-loop, or use Cursor's `WAITING:` completion convention.
  Verify `DONE:` against the diff and test evidence before advancing the plan.
- A custom profile's no-write instructions are behavioral constraints, not a
  separate sandbox. Profiles inherit parent permissions. If a profile cannot
  be selected in this client, pass its complete brief to the worker explicitly;
  if the available tools cannot meet its constraints, report that limitation.
  No-shell researchers/reviewers need a file-reading tool or source/diff excerpts
  supplied by the parent. Never silently grant them a shell to compensate.
- SFX personas in `agent-env/agents/sfx-*.md` remain declarative briefs, selected
  only by the corresponding explicit SFX workflow; no new execution authority.
- Independent G6/IIAC audit requires a separate new operator session. No worker
  can certify its parent's work. The Cursor-only integrator is unavailable.
- Push, PR creation, merge, integration, and GitHub settings stay human under
  the shared core rules. Setup does not authorize any remote delivery action.

## Hooks and boundaries

- Codex hooks are `.codex/hooks.json`; implementation is in `.codex/hooks/`.
  They never use or modify another harness's active-plan/state files.
- Hook trust belongs to the operator/client. Do not bypass trust or claim a
  guard is active before the client has loaded and trusted it.
- `$intergrax-codex:codex-igx-run-plan <plan.md>` binds an approved checklist to the current
  session through `UserPromptSubmit`; `stop` clears it and `status` reports it.
  Tick steps only after verification. Stop continues open work, allows an
  `ESCALATION:` or user interruption, and stops retrying after three stalls.
- Without functioning hooks, continue authorized work using the same rules;
  describe the missing mechanical enforcement. Do not call Claude's stop guard.
- Keep state and proofs under `.tmp/session/codex/` or `.tmp/codex-setup/`.
  Do not read `.env`, `.env.local`, or `.env.production` or expose secrets via
  shell, connectors, source, arguments, or logs.
- Keep Windows, `uv`, tier boundaries, ticket-first delivery, serial test
  execution, and English-only VCS text exactly as required by shared rules.
- The project config deliberately inherits model, sandbox, approvals, network,
  credentials, and MCP connections. It adds no vendor or production settings.
