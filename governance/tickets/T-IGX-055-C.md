# T-IGX-055-C — Cursor-only OD-AF-INT A least-privilege integration

## Scope

Implement operator-approved Cursor-only integration capability, adding a new runtime-scoped `integrator` role and `igx-integrator` subagent. Existing seven role records, all existing subagent mappings, Claude Code configuration, models and adapters must remain behaviorally and byte-for-byte unchanged.

Allowed actions: accepted worker-branch local integration, non-force push and PR creation for scoped paths, `gh pr merge --merge --match-head-commit` for qualifying PRs to `arch-freeze/rw-train` (product) or `development` (docs/tickets). Fail closed on review, exact SHA, approved write set, target, ticket or required CI mismatch.

Operator-only exclusions: main, FCS, G2/R/M3 ladder, IIAC, FRZ promotions, roadmap, protected settings, required-check bypass.

## Approved

Approved: operator @ 2026-11-22T09:04:11+01:00

Operator decision: OD-AF-INT A. Approval source: explicit operator instruction `Approved: OD-AF-INT A` in this conversation on 2026-11-22. The ISO8601 timestamp above is the recorded entry time, not a claim about the exact time the user typed the instruction. This is not an independently signed IIAC verdict.

## Implementation scope

`governance/agent-control-plane/role_registry.yaml` (additive Cursor-only role/map), `governance/TRUSTED_POLICY_PIN.yaml` (matching repin), `.cursor/agents/igx-integrator.md`, `.cursor/rules/od-af-int-a.mdc`, `.cursor/hooks.json`, `scripts/cursor/igx_integrator.py`, `scripts/hooks/igx_cursor_integration_guard.py`, `scripts/governance/verify_agent_control_plane.py`, `scripts/governance/sync_adapter_fragments.py` (Claude/Codex output must remain byte-identical), `docs/project/technical/devos/CURSOR_OD_AF_INT_A.md`, and only the relevant unit tests. No `.claude/**`, `CLAUDE.md`, `agent-env/**`, `intergrax/**`, `agents/**`, `applications/**`.

## Verification

`uv run python scripts/governance/verify_agent_control_plane.py`
`uv run python scripts/governance/verify_control_plane_consistency.py`
`uv run python scripts/governance/verify_trusted_policy_pin.py`
`uv run python scripts/governance/sync_agent_env.py --check`
`uv run pytest tests/unit/governance/test_cursor_integrator.py tests/unit/governance/test_agent_role_routing.py tests/unit/governance/test_dcp_control_plane_registry.py tests/unit/governance/test_agent_env_sync.py -q`

Cursor desktop IDE two-wave live E2E remains separately required to claim `CURSOR_ORCHESTRATION_LIVE_PASS`.

## Lifecycle state

in_progress
