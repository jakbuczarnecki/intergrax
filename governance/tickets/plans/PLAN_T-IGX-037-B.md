# PLAN T-IGX-037-B — SCENARIO-FACTORY-X P1: research, competitive analysis and capability inventory (docs only)

Parent: SFX master plan R1 (`governance/tickets/plans/SFX-MASTER-PLAN.md`), wave W1, units SFX-P1-01..03. Authority: OD-SFX-04 and OD-SFX-15. OD-SFX-15 authorizes SFX-P0..P4 before ARCH-FREEZE, and execution stops before SFX-P5.

This plan changes documentation only. It touches no path under `intergrax/`, `agents/`, `applications/`, `scripts/` or `agent-env/`.

## Deliverables

All deliverables go under `docs/project/maintainers/qualification/scenario_factory_x/`.

1. **`SFX_COMPETITIVE_EVIDENCE_MATRIX.md`** (SFX-P1-01)
   - Cover these products: n8n, Zapier, Make, Microsoft Copilot Studio, Retool, OpenAI Agent Builder / AgentKit, Google Vertex AI Agent Builder, AWS Bedrock Agents, Dify, Flowise, Langflow, Relevance AI, Vellum, Gumloop and StackAI.
   - Score each product 0–3 on dimensions D-01..D-14 (master plan §4) and classify its interaction model.
   - Every cell cites a vendor primary source with URL and access date, or reads `UNKNOWN`.
   - Close with a design-implications register `CI-01..` that maps each borrowed pattern to an SFX requirement or task.
   - The work is split into three research lanes with disjoint vendor sets, merged into one file.
2. **`SFX_IDE_TOOLING_FACTS.md`** (SFX-P1-02)
   - The current Claude Code skill and subagent definition format (`.claude/skills`, `.claude/agents` frontmatter).
   - The current Cursor rules, skills and agents formats.
   - MCP client support for streamable HTTP in both tools.
   - Each fact is cited to vendor documentation with an access date, or marked `UNKNOWN`.
   - The result decides gap SFX-GAP-07 and the missing-evidence item ME for unit SFX-P4-B01.
3. **`SFX_CAPABILITY_INVENTORY.json`** and a short companion `SFX_CAPABILITY_INVENTORY.md` (SFX-P1-03)
   - The record schema is `CAPABILITY_ID | OWNER | CONTRACT | CODE_PATH | CONFIGURATION | MATURITY | QUALIFICATION | REUSE_CLASS | LIMITATIONS | EVIDENCE_SHA`.
   - Include every capability row SFX-CAP-01..37 from master plan §5.
   - Each path is verified at the reference SHA.
   - `MATURITY` and `QUALIFICATION` are taken from the PDC matrix and from closure records. When neither exists, write `MISSING_EVIDENCE`.
   - The qualification status of Collaborative Work is resolved, or recorded as `REQUALIFICATION_REQUIRED`.
   - Use no broad repository scan. Read only the owner paths that the master plan cites.

## Acceptance

- In both markdown deliverables, every claim is either cited or labelled `UNKNOWN` / `MISSING_EVIDENCE`. There are no invented competitor capabilities.
- The inventory JSON parses, and every row has all 10 fields.
- No path outside `docs/` or `governance/tickets/` is changed.

## Test contract

```text
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Architecture decisions and ADRs (T-IGX-037-C).
- UX work (T-IGX-037-D).
- The IDE toolkit (T-IGX-037-E).
- Product code, FRZ changes and merges.

## Amendment 1 (2026-11-12): deliverable location

Operator decision in the planning chat, 2026-11-12. English rendering: "Move [the SFX workstream documents] to the dossier."

The SFX workstream documents are kept in the SFX qualification dossier `docs/project/maintainers/qualification/scenario_factory_x/` instead of `docs/project/capabilities/plan/satellites/scenario_factory/`. The reason is that the PDC-X closed-world scanner (`scripts/qualification/pdc_x_inventory.py`, `DOC_GLOBS` includes `docs/project/capabilities/**/*.md`) cannot classify `SFX_*.md` stems, which turns `test_pdc_x_inventory` red. Moving the files keeps `scripts/` unchanged (OD-SFX-04, OD-SFX-15).

Only the location changes. Content, acceptance criteria and test contract are unchanged. This amendment supersedes the earlier approval line; the ticket must be re-approved against the new digest.
