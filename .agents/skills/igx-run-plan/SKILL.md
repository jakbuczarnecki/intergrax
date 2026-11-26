---
name: igx-run-plan
description: Operator command (explicit only) — run an approved markdown plan to completion without routine go-ahead questions, backed by Antigravity's Stop hook.
---

# igx-run-plan (Antigravity adapter)

Explicit operator-invoked workflow only. Activate ONLY when the operator explicitly invokes `/igx-run-plan` or asks to execute a named approved plan file.

Canonical policy (SSOT): `agent-env/skills/igx-run-plan/SKILL.md`.

## Usage

- Start: `/igx-run-plan <path-to-plan.md>`
- Status: `/igx-run-plan status`
- Stop: `/igx-run-plan stop`

## Procedure (Antigravity)

1. **Start the active plan tracker:**
   - Run `python scripts/hooks/igx_stop_guard.py start <plan.md>` via `run_command`.
   - This writes `.tmp/session/autonomy/active-plan` (read by `.agents/hooks/igx_agy_stop_guard.py`). If the file is missing or has no open `- [ ]` step, stop and report the error.
2. **Execute steps in order:**
   - Follow `GEMINI.md` and `igx-autonomy`: no routine go-ahead questions; decide from the plan, ADRs, repo rules, and code.
   - Delegate through typed subagents (`igx-researcher`, `igx-architect`, `igx-implementer`, `igx-tester`, `igx-reviewer`) via `invoke_subagent`.
   - Tick each step (`- [x]`) in `<plan.md>` immediately after verifying the worker's `DONE:` against git and test output.
   - Keep `uv run pytest` and `uv run ruff` strictly serial (one process at a time).
3. **Escalation & Stop hook behavior:**
   - `.agents/hooks/igx_agy_stop_guard.py` blocks premature stops while open `- [ ]` steps remain and returns `{"decision": "continue", ...}` (capped at 3 consecutive blocks without checkbox progress; `IGX_HOOKS=off` disables it).
   - Stop early ONLY with `ESCALATION: <type>` (unplanned architecture; scope/cost/reversibility; plan/rules/code contradiction; forbidden action; missing access) in one batched message.
4. **Completion:**
   - When every `- [ ]` step is ticked, print `PLAN COMPLETE` followed by a concise summary. The Stop hook clears `.tmp/session/autonomy/active-plan`.
   - To cancel manually at any time, run `python scripts/hooks/igx_stop_guard.py stop`.
