# Cursor-only integrator — OD-AF-INT A

**Authority:** operator-supplied `Approved: OD-AF-INT A` (2026-11-22). This is **not** a certification verdict or permission to change Claude Code.

**Agent:** Grok orchestrates; classic Composer 2.5 `igx-implementer` implements in isolated worktrees; `igx-reviewer` independently reviews; `igx-integrator` publishes/merges only an accepted SHA-bound packet.

## Boundaries

- Product code PRs → `arch-freeze/rw-train`; docs/ticket PRs → `development` only.
- Never merge train into development through the integrator (FCS remains operator-only).
- No `main`, freeze ladder G2/R/FCS/M3, IIAC activation, FRZ promotions, roadmap, GitHub settings, admin/force, or changes under `.claude/`, `agent-env/`.
- All existing Claude Code agents, settings, model routing and runtime rights stay unchanged.
- No direct remote mutation commands in Cursor agents; `beforeShellExecution` rejects them. This hook is defense in depth, **not** a server-enforced security boundary.

## Integration packet (`.tmp/proof/<task>/packet.json`)

Create locally in the isolated integration worktree, with a separate review JSON artifact in `.tmp/proof/<task>/review.json`. Replace placeholders with actual verified data; never fabricate reviewer or CI evidence.

```json
{
  "operator_decision": "Approved: OD-AF-INT A",
  "target": "arch-freeze/rw-train",
  "kind": "product",
  "branch": "af2/example-task",
  "head_sha": "<40-char candidate SHA>",
  "base_sha": "<40-char current target SHA>",
  "ticket": "governance/tickets/T-IGX-055-C.md",
  "write_set": ["intergrax/example.py"],
  "implementer": "worker-123",
  "reviewer": "reviewer-456",
  "reviewer_role": "igx-reviewer",
  "review_verdict": "ACCEPT",
  "review_head_sha": "<same candidate SHA>",
  "review_report_file": ".tmp/proof/example/review.json",
  "review_report_sha256": "<SHA-256 of exact review file>",
  "pr_title": "Example ticket change",
  "pr_body_file": ".tmp/proof/example/pr_body.md",
  "pr_number": 123
}
```

Reviewer report JSON must include `head_sha`, `reviewer` and `verdict`, matching the packet; its digest must match the exact file.

## Lifecycle

1. Grok obtains accepted worker commit(s), an approved ticket and explicit independent review at the exact candidate SHA.
2. Grok prepares the packet in the integrator's isolated worktree. Do not modify the main checkout. An RW train worktree predating OD-AF-INT A does not contain the runner; use an absolute path to the runner from the updated development checkout, with the **current working directory still set to the RW train worktree**. Approved tickets absent from the older train can be read from the published `development` via GitHub API.
3. Integrator checks with `python scripts/cursor/igx_integrator.py validate --packet .tmp/proof/example/packet.json`.
4. `python scripts/cursor/igx_integrator.py publish --packet .tmp/proof/example/packet.json` non-force pushes the candidate branch and opens its PR.
5. After required CI is green on that PR's current SHA, `python scripts/cursor/igx_integrator.py merge --packet .tmp/proof/example/packet.json` verifies all invariants and uses `gh pr merge --merge --match-head-commit`.
6. Integrator returns the actual GitHub merge SHA. Grok refreshes the DAG and launches the next authorized wave, without asking the Operator for routine approvals.

**Fail closed:** no required checks, red/pending/skipped checks, unexpected changed file, stale base/head, absent review, changed ticket approval, or absent GH credentials all produce BLOCKED. A historically expected-red *required* check is not bypassed; it needs formal resolution under GitHub branch policy.

## Closure validation

Run the targeted unit suite and role/pin verifiers. Then execute `agent-env/plans/cursor-live-e2e-procedure.md` in the Cursor desktop IDE (Grok parent + Composer workers). The program is not verified as fully autonomous until the operator records a real, two-wave live PASS. No developer prompt alone guarantees background execution.

## Cursor-native orchestration decision (2026-11-22)

The legacy `stop` hook that read the conversation transcript and auto-submitted `followup_message` is **disabled in Cursor only**. It is not an orchestration engine and its synthetic continuation created fragile protocol dependencies. The shared `scripts/hooks/igx_stop_guard.py`, `agent-env/skills/igx-run-plan/SKILL.md`, and Claude's `.claude/settings.json` Stop hook are left unchanged.

- Cursor configuration: `.cursor/hooks.json` retains only permission hooks (`subagentStart`, `beforeShellExecution`); it has **no** `stop` registration.
- Cursor-local policy: `.cursor/rules/cursor-native-orchestration.mdc` and `.cursor/skills/igx-native-run-plan/SKILL.md`.
- Legacy Cursor `/igx-run-plan` is an automatically generated **alias** to the native skill; the Claude alias still points to the canonical `agent-env/` implementation.
- Grok uses native typed Tasks with explicit `composer-2.5`, foreground for dependencies, parallel background for independent tasks, then consumes the parent-session completion events, verifies results and advances to the next READY node.
- OD-AF-INT A and its exact-SHA, ticket, reviewer, CI and protected-target restrictions remain unchanged. No new authority is granted.

### Acceptance (one representative workflow; no artificial T1–T8 loop)

Run a real, authorized, **noncritical** two-wave code task in Cursor IDE. The first wave must produce two independent Composer worktree results; the parent must consume both native completions, independently review them, and schedule a dependent second wave without an operator go-ahead. The integrator can publish/merge **only** a legitimately approved safe PR with current required checks. Record the actual artifacts and SHA. Do not fabricate a PR or bypass CI to demonstrate a green test.

The old `agent-env/plans/cursor-live-e2e-procedure.md` describes the now-retired Cursor Stop-hook experiment and its T1–T8 reason codes; **it is not the acceptance protocol for this native approach**. It remains in `agent-env/` to avoid altering shared Claude Code source. The legacy result `CURSOR_ORCHESTRATION_LIVE_PASS` is not claimed for the new architecture.

**Limit:** native IDE background-task completion does not prove resurrection after app exit, disconnected sessions or machine shutdown. Persistent unattended operation would require a separately selected and authorized coordinator; no new runtime, dependencies or scheduler are introduced by this change.
