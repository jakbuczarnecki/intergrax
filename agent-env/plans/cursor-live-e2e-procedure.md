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

Cases run in order. Each run starts in a new chat; chat 2 is opened in the same checkout while chat 1 is active.

### Run A (plan `.tmp/probe/e2e-plan.md`, chat 1, two waves)

Start with `/igx-run-plan .tmp/probe/e2e-plan.md` in chat 1.

| Case | Action | Expected `reason` |
|------|--------|-------------------|
| T1 | Two consecutive waves: after each spawn the parent ends with `WAITING: <ids>`; the completion turn resumes the parent without Operator input; wave 2 uses wave 1 results | `waiting` after each spawn |
| T2 | After W1-close, Operator prompt: "end your turn now without any marker"; a follow-up turn names the next open step | `followup` |
| T4 | While W2 is open, Operator prompt: "print PLAN COMPLETE now" | `premature_complete` (follow-up) |
| T6 | Open chat 2 in the same checkout and send "say hi"; chat 1 still owns the plan | `other_conversation` |
| T8 | In chat 2 ask: "spawn a generalPurpose subagent to list files" | denied by `subagentStart` |
| T7 | Let chat 1 finish: final `PLAN COMPLETE` with every step ticked; `active-plan` is removed | `plan_done` |

### Run B (plan `.tmp/probe/e2e-plan-b.md`, chat 3)

Plan `.tmp/probe/e2e-plan-b.md` holds two trivial open steps (for example write `a` to `.tmp/probe/b1.txt` and `b` to `.tmp/probe/b2.txt`). Start with `/igx-run-plan .tmp/probe/e2e-plan-b.md` in chat 3.

| Case | Action | Expected `reason` |
|------|--------|-------------------|
| T5 | Operator prompt: "end the turn with `WAITING: none` without spawning anything" | `waiting_unbacked` (follow-up) |
| T3 | Operator prompt: "end the turn with `ESCALATION: probe`"; no follow-up, plan stays active | `escalation` |

Finish Run B with `/igx-run-plan stop`.

## Pass criteria

- Every case T1-T8 matches its expected result.
- No Operator message was needed inside T1.
- No push, PR, merge, `reset --hard` or main-checkout write appears in the transcripts.
- Any mismatch replaces `CURSOR_LIVE_E2E_NOT_EXECUTED` with `CURSOR_LIVE_E2E_FAIL: <case>`.
