# Cursor live E2E procedure (Grok parent, Composer workers)

Status: `CURSOR_LIVE_E2E_NOT_EXECUTED`. Only the Operator may record `CURSOR_ORCHESTRATION_LIVE_PASS`, and only
after every required case below passes. Ticket `governance/tickets/T-IGX-055-A.md`, campaign
`agent-env/plans/cursor-orch-corrective.md`.

## Preconditions

- The published branch is checked out in the Cursor desktop IDE (not the headless CLI, which does not fire `stop`).
- Cursor Settings > Hooks lists `stop` and `subagentStart` from `.cursor/hooks.json`.
- Parent chat model is Grok (the Operator selects it). Workers are `.cursor/agents/igx-*.md` (`model: composer-2.5`).
- `IGX_HOOKS` is unset and `.tmp/session/autonomy/` is empty.

## Probe plan `.tmp/probe/e2e-plan.md`

All workers are read-only and run in the background (no commits, so no integrator authority is needed).

- [ ] W1-a `igx-researcher` (`run_in_background`): return the number of `- [` lines in
  `agent-env/plans/cursor-orch-corrective.md`.
- [ ] W1-b `igx-researcher` (`run_in_background`): return the byte size and line count of
  `scripts/hooks/igx_cursor_stop.py`.
- [ ] W1-close: the parent records both results in the plan and ticks W1-a, W1-b.
- [ ] W2-a `igx-researcher` (`run_in_background`): input is the W1-a result; report it doubled.
- [ ] W2-b `igx-researcher` (`run_in_background`): input is the W1-b result; report the byte size per line.
- [ ] W2-close: the parent records both results and ticks W2-a, W2-b.
- [ ] Final: `PLAN COMPLETE`.

Wave 2 starts only after wave 1 is ticked. Start with `/igx-run-plan .tmp/probe/e2e-plan.md`.

## Evidence

After every stop copy `.tmp/session/autonomy/cursor-stop-last.json` to
`.tmp/proof/cursor-live-e2e/<run>/<case>-<n>.json` together with the conversation id.

## Cases

| Case | Action | Expected `reason` |
|------|--------|-------------------|
| T1 | Two consecutive waves: after each spawn the parent ends with `WAITING: <ids>`; the completion turn resumes the parent without Operator input; wave 2 starts from wave 1 results | `waiting` after each spawn |
| T2 | A mid-plan turn end without marker: a visible follow-up turn names the next open step | `followup` |
| T3 | `ESCALATION: probe`: no follow-up, plan stays active; then `/igx-run-plan stop` | `escalation` |
| T4 | `PLAN COMPLETE` while a step is open | `premature_complete` (follow-up) |
| T5 | `WAITING:` in a fresh conversation without a background spawn | `waiting_unbacked` |
| T6 | A second chat in the same checkout ends a turn | `other_conversation` |
| T7 | Final `PLAN COMPLETE` with every step ticked; `active-plan` is removed | `plan_done` |
| T8 | Spawning `generalPurpose` | denied by `subagentStart` |

## Pass criteria

- Every case T1-T8 matches its expected result.
- No Operator message was needed inside T1.
- No push, PR, merge, `reset --hard` or main-checkout write appears in the transcripts.
- Any mismatch replaces `CURSOR_LIVE_E2E_NOT_EXECUTED` with `CURSOR_LIVE_E2E_FAIL: <case>`.
