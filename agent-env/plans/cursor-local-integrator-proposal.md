# Proposal - least-privilege local integrator role (NOT APPLIED)

Status: **NOT APPLIED - Operator decision required.** Ticket `governance/tickets/T-IGX-055-A.md`, campaign
`agent-env/plans/cursor-orch-corrective.md` (finding F05). Nothing in this file changes authority; the current
`governance/agent-control-plane/role_registry.yaml` stays as it is.

## Problem

A Cursor wave ends with N reviewed worker branches (`igx-implementer`, one worktree each). Integrating them
before the next wave needs a commit (merge or cherry-pick). Today:

- `orchestrator.may_commit: false` and the registry header says "no branch integration";
- only `implementer` may commit, and only on the branch of its own worktree;
- so local integration is a human step, and the next wave cannot start on the integrated result without the
  Operator.

The a058cee19 Cursor session worked around this by running `git reset --hard` in the main checkout and
pushing to `development`. Both are forbidden (`global_denied_commands`, `intergrax-core` branch safety). The
fix must make the legal path available, not widen the orchestrator.

## Proposed role

`integrator`: merges reviewed worker branches into one local integration branch in its own worktree. Nothing
else.

| Allowed | Denied |
|---------|--------|
| `git worktree add -b integ/<campaign>/<wave> <path> <base>` under `D:/Projekty/_wt/` per `/igx-worktree` | any write in the main checkout or another worktree |
| `git merge --no-ff <worker-branch>` of branches named in the wave record with reviewer verdict ACCEPT or ACCEPT_WITH_FINDINGS | merging an unreviewed or REJECT branch |
| resolve a conflict only inside the union of the workers' edit scopes; else `BLOCKED: contradiction` | editing files outside those scopes |
| run the ticket test contract once after the merges | skipping or narrowing tests |
| commit merge commits with `Ref: governance/tickets/T-IGX-<NN>-<L>.md` | push, PR, merge into `development`/`main`, `reset --hard`, rebase, force, branch delete |

The next wave starts from `integ/<campaign>/<wave>`. Push, PR and merge into `development` remain human.

## Exact patch (to apply only after approval)

`governance/agent-control-plane/role_registry.yaml`:

```yaml
roles:
  # ... existing roles unchanged ...
  integrator:
    description: Merges reviewed worker branches into one local integration branch in its own worktree; no push, PR or merge into protected branches.
    allowed_skills:
      - igx-devos-ticket-driven-work
      - igx-devos-enterprise-quality-gate
    may_commit: true
    commit_scope: own_worktree_integration_branch

subagent_role_map:
  # ... existing entries unchanged ...
  igx-integrator:
    maps_to: integrator
    max_parallel: 1
    read_only: false
    requires_worktree: true
```

Header comment line 3 becomes: `orchestrator=false: no commit, no push, no pull request, no merge; local
integration only through the integrator role in its own worktree.`

Companion files (same approval):

- `agent-env/agents/igx-integrator.md` (canonical agent, Claude `model: sonnet`, tools Read, Grep, Glob, Bash,
  `isolation: worktree`) and the generated adapters (`sync_agent_env.py`); Cursor adapter `model:
  composer-2.5`.
- `agent-env/skills/igx-autonomy/SKILL.md` Commit ownership: one bullet for `integrator`.
- `.cursor/hooks.json` is unchanged (the `subagentStart` guard denies only `generalPurpose`).

## Pin and verifier implications

- `role_registry.yaml` is pinned in `governance/TRUSTED_POLICY_PIN.yaml`. Applying the patch requires
  `uv run python scripts/governance/verify_trusted_policy_pin.py --write-pin` in the same approved change;
  re-pinning without this approval would mask a policy change, so it is not done here.
- Consumers to re-run: `verify_agent_control_plane.py`, `verify_control_plane_consistency.py`,
  `generate_control_plane_matrix.py`, `sync_adapter_fragments.py`, `igx_gov/inventory.py`, tests
  `test_agent_role_routing.py`, `test_dcp_control_plane_registry.py`. A new key `commit_scope` must be accepted
  by these verifiers or dropped from the patch.
- `may_commit` stays false for `orchestrator`.

## Related inconsistency (same decision)

`subagent_role_map.generalPurpose` maps to `orchestrator` with `max_parallel: 2`, but `.cursor/hooks.json`
`subagentStart` denies `generalPurpose` (fail closed) and Claude Code denies `general-purpose`. Recommendation:
remove the `generalPurpose` entry in the same pinned change.

## Options

1. **Approve the integrator role** (recommended): legal local integration, next wave without the Operator,
   push/PR/merge stay human. Cost: one pinned registry change + one agent file + re-pin.
2. **Keep human integration**: no authority change; each wave boundary waits for the Operator to merge worker
   branches locally.
3. Widen `orchestrator.may_commit` - rejected: gives commit rights to the role that also routes and reviews.
