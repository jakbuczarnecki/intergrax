---
name: igx-autonomy
description: Antigravity orchestration protocol and decision policy — subagent DONE/BLOCKED reports, when the parent decides alone, when to escalate, and typed igx-* delegation.
---

# igx-autonomy (Antigravity adapter)

Read and follow the canonical decision policy, `DONE:`/`BLOCKED:` contract, escalation types, and commit ownership rules in `agent-env/skills/igx-autonomy/SKILL.md`.

## Antigravity-specific orchestration rules

Ignore the Claude/Cursor-only model-routing and Cursor `WAITING:` paragraphs in the canonical skill; apply these Antigravity equivalents instead:

1. **Typed subagents only (`invoke_subagent`):**
   - `igx-researcher` — read-only discovery (`Workspace: "inherit"`, `Model: "flash"`, max 3 parallel).
   - `igx-architect` — bounded plans under `agent-env/plans/` or named tickets (`Workspace: "inherit"`, `Model: "pro"`, max 1).
   - `igx-implementer` — scoped code changes and commits (`Workspace: "share"` or `"branch"`, `Model: "inherit"`, max 2 parallel; serial `uv`/`pytest`).
   - `igx-tester` — ticket Test contract and `igx_gov ticket` verification (`Workspace: "inherit"`, `Model: "flash"`, max 1; serial `uv`/`pytest`).
   - `igx-reviewer` — read-only code review, no shell (`Workspace: "inherit"`, `Model: "pro"`, max 1; never declares G6 PASS).
   - Do not delegate generic implementation to `self` or `general-purpose`.
2. **Reactive background wakeup (no polling):**
   - `invoke_subagent` runs asynchronously; stop calling tools after launching parallel workers and wait for Antigravity's reactive message notification.
   - Never sleep-loop or poll subagent status; use `send_message` only to deliver a parent decision or correction (max 1 correction/relaunch per worker per step; second failure -> `ESCALATION:`).
3. **Commit ownership (SSOT: `governance/agent-control-plane/role_registry.yaml`):**
   - Only `igx-implementer` (`may_commit: true`) commits on its own isolated worktree branch with `Ref: governance/tickets/T-IGX-<NN>-<L>.md` and no AI attribution.
   - Top-level `orchestrator` has `may_commit: false`. Integrating worker branches, `git push`, PR creation, merge, and GitHub settings stay human unless explicitly authorized (escalate as `ESCALATION: forbidden action`).
4. **Plan completion:**
   - Verify each worker's `DONE:` against git/test evidence before ticking the plan step (`- [x]`).
   - When all steps are complete, print `PLAN COMPLETE` and a concise summary.
