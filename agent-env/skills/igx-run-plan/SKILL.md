---
name: igx-run-plan
description: Operator command - run an approved plan to the end without questions; a Stop hook keeps the session working while plan steps are open
disable-model-invocation: true
---

# igx-run-plan

Usage: `/igx-run-plan <plan.md>` to start, `/igx-run-plan stop` to end.

The plan is a markdown file with `- [ ]` / `- [x]` step lines. Tick a step (`- [x]`) as soon as it is done and verified.

## Start

1. Run `python scripts/hooks/igx_stop_guard.py start <plan.md>`. It writes `.tmp/session/autonomy/active-plan`; on an error (no file, no open step) stop and report it.
2. Execute the steps in order, following `intergrax-core` (Autonomy): no go-ahead questions, decide from the plan, ADRs, rules and code. Delegate through the typed `igx-*` agents per `/igx-autonomy`.
3. Stop only with `ESCALATION: <type>` (unplanned architectural problem; decision that changes scope or cost or is hard to reverse; contradiction between plan and code or rules; forbidden action needed; missing access). Batch all open items in that one message.
4. When no step is open print `PLAN COMPLETE` and a short summary. The hook then clears the active plan.

## Stop

`/igx-run-plan stop` runs `python scripts/hooks/igx_stop_guard.py stop`. `status` prints the next open step.

The Stop hook blocks any other stop while a step is open; after three blocks without a checkbox change it lets one stop through with a warning. `IGX_HOOKS=off` disables it.

## Cursor

Cursor registers the same script on `stop` in `.cursor/hooks.json` with `loop_limit: null`. Native Cursor input (no `last_assistant_message`) is handled by `scripts/hooks/igx_cursor_stop.py`; Claude Code input from `.claude/settings.json` keeps the behaviour above (`decision: block` and `reason`).

- A Cursor Plan Mode file (`~/.cursor/plans/*.plan.md`) may be the plan: its frontmatter `todos:` are the steps, `completed` and `cancelled` count as ticked. A file with `- [ ]` lines is parsed as a checklist.
- The plan belongs to the Cursor conversation whose transcript ran `igx_stop_guard.py start`. Other conversations in the same checkout are never continued.
- Only the final assistant text of the current turn is read. A line that starts with `ESCALATION:`, `PLAN COMPLETE` or `WAITING:` (after `>`, `*`, `_`, `#`, `-` or spaces) is an end marker.
- `ESCALATION: <type>` ends the turn. `PLAN COMPLETE` ends it and clears the plan only when no step is open; with open steps it is answered with the open step.
- `WAITING: <task ids>` ends the turn while background workers run; the completion turn resumes the parent. The hook checks only that this conversation spawned a background Task (`run_in_background: true`) at some point, not that it is still running, so use `WAITING:` only for workers that are really outstanding. Without any background spawn it counts as a stop without progress.
- Otherwise the hook sends one `followup_message` with the next open step. The second stop without a checkbox change gets a stall warning; the third is allowed and marks the plan stalled (no more follow-ups until a step is ticked). Follow-ups since `start` are capped at `max(10, 3 x steps)`.
- `status` `error` or `aborted`, a missing transcript, a repeated event or any hook error allows the stop. The hook only ever emits `followup_message`.

When the hook does not run, the Autonomy rule still applies: never ask for permission to take the next step.
