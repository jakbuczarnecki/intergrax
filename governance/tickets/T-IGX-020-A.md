# T-IGX-020-A — Shared agent environment (`agent-env/`) for Cursor and Claude Code

## Scope

Move tool-neutral agent configuration out of `.cursor/` into one shared directory and make both tools consume it through generated thin adapters:

- `agent-env/rules/` (11 rules, `.mdc` -> `.md`, new frontmatter field `loading: always|explicit|auto`), `agent-env/skills/` (11 `igx-devos-*` skills), `agent-env/plans/`, `agent-env/README.md`.
- Generator and drift check `scripts/governance/sync_agent_env.py` (wired into `igx_gov sync-adapters` and `igx_gov doctor`), unit test `tests/unit/governance/test_agent_env_sync.py`.
- Generated adapters: `.cursor/rules/*.mdc`, `.cursor/skills/*`, `.claude/skills/*`, import block in `CLAUDE.md`; shared Claude settings in `.claude/settings.json`.
- Path updates in live artifacts: `skill_routing_registry.yaml`, `policy-registry.yaml`, `igx_gov/inventory.py`, `check_cursor_token_setup.py`, `AGENTS.md` (tool-neutral, <= 45 lines), bootstrap prompts and technical guides.

No product-tier (`intergrax/`, `agents/`, `applications/`) code is touched.

## Plan digest

(none — configuration relocation; scope approved by the operator in chat on 2026-11-07)

## Approved

(pending — scaffold only; operator adds the `Approved:` line)

## Test contract

```powershell
uv run python scripts/governance/sync_agent_env.py --check
uv run python scripts/ci/check_cursor_token_setup.py
uv run python scripts/governance/verify_agent_skills.py
uv run python scripts/governance/validate_policy_registry.py
uv run pytest tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_dcp_control_plane_registry.py tests/unit/governance/test_dcp_policy_registry.py -q
```

## Follow-ups (not part of this batch)

- After the commit: `uv run python scripts/governance/verify_trusted_policy_pin.py --write-pin` (pins hash HEAD blobs of `skill_routing_registry.yaml` and `policy-registry.yaml`), then a second commit.
- Regenerate `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` (lists former `.cursor/...` paths) under operator control.
- `tests/qualification/state_x/inventory.py` allowlist still names `.cursor/rules/intergrax-env-vendor-qualification.mdc`.
- Move the local, git-ignored `.cursor/plans/ebh-7-annexes/` to `agent-env/plans/`.

## Out of scope

Historical qualification evidence and approved plans that cite old paths (`docs/project/maintainers/qualification/**`, `governance/tickets/plans/**`), FRZ / IIAC / G6 status changes, Docker, vendor integrations, production configuration.

## Lifecycle state

draft
