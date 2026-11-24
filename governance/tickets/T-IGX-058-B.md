# T-IGX-058-B — Cursor direct execution compatibility qualification (WF-01)

## Scope

Complete the Cursor-only ChatGPT Direct Execution correction authorized by the Operator on 2026-11-24. Resolve conflicting generated role, autonomy, delivery, audit, ticket, worktree and PR instructions without modifying shared Claude sources. Preserve the scoped candidate publisher, qualify its read-only preflight before implementation, enforce exactly one canonical ticket Ref trailer and preserve non-force draft-only publication. No new orchestrator, Grok dependency, shared role registry or GitHub protection change. Direct development publication was explicitly requested by the Operator.

## Plan digest

Plan digest: n/a (bounded scope and write set in this ticket)

## Approved

Approved: operator @ 2026-11-24T12:15:27+01:00

Approval source: Operator chat instructions to complete WF-01, make changes directly on development and preserve Claude Code. Timestamp is the recorded entry time, not the exact user message time. No independent certification is claimed.

## Write set

```text
.cursor/rules/igx-cursor-direct-execution.mdc
.cursor/rules/igx-enterprise-delivery.mdc
.cursor/rules/intergrax-agent-roles.mdc
.cursor/rules/intergrax-core.mdc
.cursor/rules/intergrax-independent-audit.mdc
.cursor/rules/intergrax-iteration.mdc
.cursor/skills/igx-autonomy/SKILL.md
.cursor/skills/igx-devos-enterprise-quality-gate/SKILL.md
.cursor/skills/igx-devos-github-pr-workflow/SKILL.md
.cursor/skills/igx-devos-parallel-orchestration/SKILL.md
.cursor/skills/igx-devos-plan-ticket-scope/SKILL.md
.cursor/skills/igx-devos-qc-orchestrator/SKILL.md
.cursor/skills/igx-devos-role-architect/SKILL.md
.cursor/skills/igx-devos-role-implementer/SKILL.md
.cursor/skills/igx-devos-role-reviewer/SKILL.md
.cursor/skills/igx-devos-role-tester/SKILL.md
.cursor/skills/igx-devos-ticket-driven-work/SKILL.md
.cursor/skills/igx-devos-ticket-scaffold/SKILL.md
.cursor/skills/igx-worktree/SKILL.md
agent-env/rules/igx-cursor-direct-execution.md
governance/tickets/T-IGX-058-B.md
scripts/cursor/igx_publish_candidate.py
scripts/governance/sync_agent_env.py
tests/unit/governance/test_agent_env_sync.py
tests/unit/governance/test_cursor_candidate_publication.py
```

## Test contract

```text
uv run --no-sync python scripts/governance/sync_agent_env.py --check
uv run --no-sync pytest tests/unit/governance/test_cursor_candidate_publication.py tests/unit/governance/test_cursor_integration_guard.py tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_claude_surface_unchanged.py -q
uv run --no-sync ruff check scripts/cursor/igx_publish_candidate.py scripts/governance/sync_agent_env.py tests/unit/governance/test_cursor_candidate_publication.py tests/unit/governance/test_agent_env_sync.py
```

## Historical evidence and boundaries

PR #804 confirms actual Windows publication of d9717c15421a20f68232cbec574839c949455005. PR #803 ticket-governance failed on two historical missing Ref trailers; do not rewrite shared history. The 15 governance-fast IIAC/ACT-N11 failures on PR #803 match the 15 failures on its base a4f2b2677b88cfe8afef9ae409bae1c91a0ad8bd exactly. No exception, repin, gate weakening or product remediation in this ticket. Local IDE and CI qualification results must be reported separately.

## Out of scope

.claude/**, CLAUDE.md, shared agent-env rules/skills/agents (other than the Cursor-only source), role_registry.yaml, trusted policy pin, IIAC, FRZ, product code, CI workflow definitions, Git hooks and branch protections.

## Lifecycle state

in_progress
