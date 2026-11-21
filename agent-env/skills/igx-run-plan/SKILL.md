---
name: igx-run-plan
description: Operator command - run an approved plan to the end without questions; a Stop hook keeps the session working while plan steps are open
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
