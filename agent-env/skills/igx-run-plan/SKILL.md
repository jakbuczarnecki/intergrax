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

Cursor registers this script on `stop` in `.cursor/hooks.json` with `loop_limit: null` (a numeric default of 5 would cut a plan that is still ticking steps). Native input is `status` plus `loop_count` and usually has no `last_assistant_message`. While a step is open the script emits one `followup_message`, unless the visible text (payload or a bounded transcript tail) starts with `ESCALATION:` or `PLAN COMPLETE`, the status is `error` or `aborted`, or three stops make no checkbox change. It never emits `decision` and `followup_message` together. Claude Code still receives `decision: block` and `reason` from `.claude/settings.json`. A second delivery of the same conversation and generation does not advance the stall counter. When a numeric Cursor `loop_limit` drops further follow-ups, the plan stays active; that stop is not `PLAN COMPLETE`. When the hook does not start, the Autonomy rule still applies: never ask for permission to take the next step.
