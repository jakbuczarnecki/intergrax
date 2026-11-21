# CURSOR-ORCH-CORRECTIVE - Grok parent / Composer workers in Cursor

Ticket: `governance/tickets/T-IGX-055-A.md`. Branch `fix/cursor-orch-corrective`, worktree
`D:/Projekty/_wt/intergrax/t-igx-055-a`, stacked on PR #742 head `ac0c0b753` (trusted pin fix, kept as a
separate dependency; this branch does not repeat it). Baseline `origin/development` = `a058cee19`.

Scope: Cursor-only behaviour. Claude Code paths stay byte-for-byte equivalent for every input they accepted
before. No edits under `intergrax/`, `agents/`, `applications/`, `.claude/settings.json`, `CLAUDE.md`,
`governance/agent-control-plane/*.yaml`, `governance/TRUSTED_POLICY_PIN.yaml`.

## Checklist

- [x] W0 baseline and dossier (this file)
- [x] W1 Cursor stop adapter (`scripts/hooks/igx_cursor_stop.py`, dispatch in `igx_stop_guard.py`, tests)
- [x] W2 Cursor contract docs (`igx-run-plan`, `igx-autonomy`, `PARALLEL-EXECUTION.md`)
- [x] W3 local integrator proposal (approval-gated, not applied)
- [x] W4 regression, sync check, policy pin, live probe or `CURSOR_LIVE_E2E_NOT_EXECUTED`
- [ ] W5 independent review and commit set

## W0 dossier (evidence, 2026-11-21)

Live evidence comes from the local Cursor install (`cursor-agent` 2026.11.19) and its transcripts under
`~/.cursor/projects/d-Projekty-intergrax/agent-transcripts/` (session `a3e38fe8`, which produced `a058cee19`).

| Id | Verdict | Evidence |
|----|---------|----------|
| F01 | CONFIRMED | Cursor Plan Mode files `~/.cursor/plans/*.plan.md` keep steps in YAML frontmatter `todos:` (`id`, `content`, `status`: `pending`, `in_progress`, `completed`, `cancelled`) and contain no `- [ ]` line, so `igx_stop_guard.py start` rejects them. The active-plan file is per checkout, not per conversation: any Cursor chat in the checkout is continued. The stop payload carries no plan id. |
| F02 | CONFIRMED | Stall warning is `systemMessage`, which is not part of the Cursor `stop` output (only `followup_message`); the stall is invisible in Cursor. `PLAN COMPLETE` is matched only at text start; the a058cee19 session ended with it on the last line. `PLAN COMPLETE` with open steps clears the plan. No overall follow-up cap. |
| F03 | CONFIRMED | Live payload keys: `conversation_id`, `generation_id`, `hook_event_name=stop`, `loop_count`, `session_id`, `status`, `transcript_path`, `workspace_roots`, no `cwd`, no `last_assistant_message`. Transcript is JSONL `{"role": ..., "message": {"content": [{"type": "text"}, {"type": "tool_use"}]}}` plus `{"type": "turn_ended"}`; it was flushed 0.9 s before the hook ran. The tail reader returns the last assistant text of any turn, so a stale text from an earlier turn can decide. Root discovery reads `cwd` only. |
| F04 | CONFIRMED (doc overclaim) | A background Task (`run_in_background: true`) completion arrives as a synthetic user turn "Perform any necessary follow-up actions in response to the subagent completion above". No transcript contains a delivered stop-hook `followup_message`; the a058cee19 "PASS" was driven by that completion turn, not by the hook. With an active plan, a parent that ends its turn to wait for workers is pushed by the hook into stall. |
| F05 | CONFIRMED, governance | `orchestrator.may_commit: false`, no integrator role. Changing it is an authority change: approval-gated proposal only (W3). |
| F06 | PASS | `.cursor/agents/igx-*.md` carry `model: composer-2.5`; `.claude/agents` keep `sonnet`/`opus`. Finding: `PARALLEL-EXECUTION.md` decision tree still names `generalPurpose`, which the `subagentStart` guard denies; `role_registry.yaml` still maps `generalPurpose` (governance file, listed in W3). |
| F07 | PASS | `igx-implementer`, `best-of-n-runner` `max_parallel: 2`; unchanged. |
| F08 | PASS | `git worktree add -b` under an approved plan; path `D:/Projekty/_wt/intergrax/t-igx-055-a` (37 chars). |
| F09 | PASS with #742 | `verify_trusted_policy_pin.py` OK at `ac0c0b753`; red on `a058cee19` (policy-registry hash). This branch changes no pinned file. |

Out-of-scope observation: the a058cee19 Cursor session ran `git reset --hard origin/development` in the main
checkout and pushed to `development` directly (rule violations, not repaired here).

## W1 specification - Cursor stop adapter

New stdlib-only module `scripts/hooks/igx_cursor_stop.py`. `igx_stop_guard.py` keeps its Claude path unchanged
and gains three small edits:

1. `decide()`: right after the `loaded is None` check, `if _native_cursor(data):` import `igx_cursor_stop`
   (insert `Path(__file__).resolve().parent` into `sys.path` first, because `.claude/settings.json` runs the
   script through `runpy.run_path`) and return its decision. When it asks to clear, call `_clear(root)`.
2. `hook()`: root discovery start = `cwd`, else the first string in `workspace_roots`.
3. `_steps_of()`: when the file has no checkbox line, return `igx_cursor_stop.frontmatter_todos(text)`
   (Cursor plan todos). A file with checkbox lines is parsed exactly as before.

Cursor decision (`status` != `completed` allows; every path below fails open on any error):

| Situation | Result |
|-----------|--------|
| no `conversation_id` (else `session_id`), or no readable transcript | allow |
| plan not yet owned | owner = this conversation only if its transcript contains `igx_stop_guard.py start`; else allow |
| owner differs | allow (another conversation) |
| same `conversation_id` + `generation_id` + `loop_count` + progress signature as last event | replay the stored result |
| final assistant text of the current turn (after the last `role: user` line) | absent -> allow |
| a line starting (after `>`, `*`, `_`, `#`, `-`, spaces) with `ESCALATION:` | allow, counters unchanged |
| no open step | allow and clear |
| `PLAN COMPLETE` line with open steps | premature: counts as a stop without progress, message names the open steps |
| `WAITING:` line and the transcript holds a `"run_in_background": true` Task spawn | allow, counters unchanged |
| `WAITING:` without such a spawn | counts as a stop without progress |
| stalled and no checkbox change | allow silently |
| follow-ups since activation >= `max(10, 3 * steps)` | allow, mark stalled |
| 1st stop at this progress signature | `followup_message`: next open step and the three end markers |
| 2nd stop at this signature | `followup_message`: stall warning, tick or `ESCALATION:`; next stop is allowed |
| 3rd stop at this signature (`MAX_BLOCKS`) | allow, mark stalled |

Output is only ever `{"followup_message": ...}` or nothing. State lives in
`.tmp/session/autonomy/cursor-stop-state.json`, keyed by activation (`plan path` + active-plan file
`mtime_ns`), so a new `start` resets it and the Claude state file is untouched. A diagnostic
`cursor-stop-last.json` records decision and reason code only (no message text, no e-mail).

## W2 contract text

- `igx-run-plan`: Cursor paragraph replaced by the table above in prose, Cursor plan files accepted.
- `igx-autonomy` (Cursor sections): `WAITING: <task ids>` when ending a turn with background workers
  outstanding; the completion turn is the receive; at most one retry per worker before `ESCALATION`.
- `PARALLEL-EXECUTION.md`: read-only search is `explore` / `igx-researcher`; `generalPurpose` is denied.

## W4 evidence (2026-11-21, proof under `.tmp/proof/cursor-orch-corrective/`, not tracked)

- `sync_agent_env.py --check` OK (70 adapters); `verify_trusted_policy_pin.py` OK; ruff clean.
- pytest (ticket contract): stop/cursor/subagent guard tests pass; `test_agent_env_sync.py` has 3 failures that
  pre-exist at base `ac0c0b753` (`.cursor/rules/intergrax-core.mdc` extract lost `git push` after the
  a058cee19 core rewrite). This branch changes no rule file; fixing it touches the Claude-loaded core rule and
  is left to a separate change.
- Live probe: headless `cursor-agent -p` (2026.11.19, `composer-2.5`, scratch worktree) ran `start` and step 1,
  then ended; it never invoked the project `stop` hook (no `stop-last-keys.json`). Launched from Git Bash, the
  CLI evaluates its PowerShell hook wrapper with `bash eval` and blocks every hooked tool; launch it from
  PowerShell.
- Offline replay of the real CLI transcript with a live-shaped `stop` payload: owner follow-up names step 2,
  duplicate event replays, another conversation is allowed, 2nd stop stall warning, 3rd allowed + stalled.
- IDE live E2E (hook-delivered `followup_message` in the Cursor editor): `CURSOR_LIVE_E2E_NOT_EXECUTED`.
  Procedure: in the Cursor IDE open the checkout, run `/igx-run-plan .tmp/probe/plan.md` with two steps, let the
  agent finish step 1 and end its turn, then check that a follow-up turn naming step 2 appears and
  `.tmp/session/autonomy/cursor-stop-last.json` shows `"reason": "followup"`.

## W3 local integrator (proposal, not applied)

See `agent-env/plans/cursor-local-integrator-proposal.md`. Requires an Operator decision because it adds
commit authority to `governance/agent-control-plane/role_registry.yaml` (a pinned file).
