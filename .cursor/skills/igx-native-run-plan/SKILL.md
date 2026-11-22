---
name: igx-native-run-plan
description: Execute an approved Integrax plan in Cursor through native Grok and Composer subagents, without Stop-hook loops.
---

# Cursor native plan execution

Use this skill for approved long-running work in Cursor IDE. This is the **Cursor-only** replacement for the older `/igx-run-plan` flow. Claude Code continues to use its existing `/igx-run-plan` unchanged.

## Before execution

1. Read the approved plan, exact SHA, tickets, allowed paths and current review/CI statuses. Verify the authorization for each gate.
2. Derive a READY set from predecessors, not from chat history; respect max two implementers, one pytest, isolated worktrees and all enterprise STOP gates.
3. Do **not** invoke `scripts/hooks/igx_stop_guard.py start` or create `.tmp/session/autonomy/active-plan`. No `stop` hook is registered in `.cursor/hooks.json`.
4. Use Cursor's native `Task` with explicit `composer-2.5` for scoped typed workers. The parent Grok orchestrator never takes an implementer's commit role.

## Continue without micro-approval

- Launch independent READY tasks in parallel where authorized; use foreground for task dependencies and background for work whose result is delivered through Cursor's built-in completion event.
- When a worker result arrives, verify exact branch/SHA, write set, test evidence, reviewer independence and gate eligibility. Accept or request one scoped correction; never count a worker's self-reported DONE as certification.
- For accepted packages, delegate only authorized Git operations to Cursor `igx-integrator` under OD-AF-INT A, then reload the actual integrated SHA and schedule further READY nodes immediately.
- If a background worker remains outstanding, conclude the current turn normally; Cursor is responsible for delivering native completion. Do not synthesize follow-up user messages, use `WAITING:` to force a turn, poll or scan transcripts.
- If no worker remains and READY nodes exist, keep running the next ready work in the current turn. On actual operator-only gates, report a single `ESCALATION:` with evidence and options; continue unrelated READY nodes when permitted.
- Report `PLAN COMPLETE` only when all mandatory acceptance criteria and reserved approvals are confirmed on the correct published SHAs.

## Recovery, not simulated continuity

Record minimal checkpoints in the existing canonical plan/tickets and Git/PR evidence after each accepted wave. At a fresh Cursor session, reconcile those durable artifacts and resume pending READY work; do not assume Cursor will automatically resurrect a terminated session.

## Scope boundary

Do not modify `.claude/**`, `CLAUDE.md`, `agent-env/**`, shared Claude tools/skills, pinned policies, or protected architecture gates as part of this Cursor execution adjustment. No new server, scheduler, SDK integration or background daemon is authorized by this skill.
