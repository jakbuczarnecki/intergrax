# PLAN T-IGX-037-E — SCENARIO-FACTORY-X P4: proposal-only IDE toolkit v0 (agent-env declarative files only)

Parent: SFX master plan R1 (`governance/tickets/plans/SFX-MASTER-PLAN.md`), wave W2, units SFX-P4-A01..A02.

Authority:
- **OD-SFX-04** (pre-freeze scope).
- **OD-SFX-15** (pre-freeze sequencing; P4 is limited to a proposal-only toolkit with no change to protected platform mechanisms).
- **OD-SFX-16** (C-OD-01 resolved by the operator on 2026-11-12). The regenerated adapters under `.claude/skills/` and `.cursor/` may ship with the agent-env sources.

The units below stay **in W3**, after ARCH-FREEZE CLOSED and GATE-SFX-ENTRY:
- `scripts/governance/sync_agent_env.py` and its tests (SFX-P4-B01)
- `governance/agent-control-plane/role_registry.yaml` and its pinned test (SFX-P4-B02)

## Deliverables

1. **Rule** `agent-env/rules/intergrax-scenario-factory.md` (`loading: auto`, with a `description`).
   - Encodes the SFX invariants: no second engine, registry or governance authority; authority levels L0–L3, with only L0/L1 available in v0; outputs are proposal-only drafts under `.tmp/sfx/<slug>/`; no secrets; artifacts follow the SFX design schemas from T-IGX-037-C; SFX never claims admission or certification.
   - Because the rule loads on `auto`, the `CLAUDE.md` generated block does not change.
2. **Skills** `agent-env/skills/sfx-{session,discover,specify,resolve,blueprint,integration-setup,test-plan,present}/SKILL.md`.
   - Frontmatter `name` equals the directory name, and each skill has a `description`.
   - `sfx-session` runs the full lifecycle step by step and resumes from the draft folder. The other skills each run one step.
   - All of them write L1 drafts only. They never edit product paths, run scenarios or call a production runtime.
3. **Agent persona briefs** `agent-env/agents/sfx-{discovery-analyst,scenario-architect,capability-resolver,integration-advisor,test-designer,critic}.md` (declarative).
   - Skills use these files as subagent briefs.
   - `sync_agent_env.py` does not glob this directory, so the files are not synced in v0.
4. **Regenerated adapters** produced by running `uv run python scripts/governance/sync_agent_env.py` (generator unchanged): `.claude/skills/sfx-*/SKILL.md`, `.claude/skills/intergrax-scenario-factory/SKILL.md`, `.cursor/rules/intergrax-scenario-factory.mdc`, `.cursor/skills/sfx-*/SKILL.md`.
5. **Walkthrough record** `docs/project/maintainers/qualification/scenario_factory_x/SFX_IDE_TOOLKIT_V0_WALKTHROUGH.md`.
   - A dry run of `/sfx-session` on the REF-1 problem statement.
   - The produced drafts stay in `.tmp/`, which is gitignored. The record keeps their file list and digests only.

## Acceptance

- `sync_agent_env.py --check` is clean.
- `check_cursor_token_setup.py` passes; the forbidden phrase "full repository access" is absent.
- No file under `scripts/`, `intergrax/`, `agents/`, `applications/` or `governance/agent-control-plane/` changes.
- The skills never instruct the agent to write outside `.tmp/sfx/` or to run any runtime.

## Test contract

```text
uv run python scripts/governance/sync_agent_env.py --check
uv run pytest tests/unit/governance/test_agent_env_sync.py -q
uv run python scripts/ci/check_cursor_token_setup.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- SFX-P4-B01/B02 (W3).
- The CLI and MCP backed toolkit v1 (SFX-P10-08).
- Product code, FRZ changes and merges.

## Amendment 1 (2026-11-12): deliverable location

Operator decision in the planning chat, 2026-11-12. English rendering: "Move [the SFX workstream documents] to the dossier."

The SFX workstream documents are kept in the SFX qualification dossier `docs/project/maintainers/qualification/scenario_factory_x/` instead of `docs/project/capabilities/plan/satellites/scenario_factory/`. The reason is that the PDC-X closed-world scanner (`scripts/qualification/pdc_x_inventory.py`, `DOC_GLOBS` includes `docs/project/capabilities/**/*.md`) cannot classify `SFX_*.md` stems, which turns `test_pdc_x_inventory` red. Moving the files keeps `scripts/` unchanged (OD-SFX-04, OD-SFX-15).

Only the location changes. Content, acceptance criteria and test contract are unchanged. This amendment supersedes the earlier approval line; the ticket must be re-approved against the new digest.
