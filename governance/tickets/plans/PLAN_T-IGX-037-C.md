# PLAN T-IGX-037-C — SCENARIO-FACTORY-X P2: gap analysis, proposed architecture, ADRs and contract designs (docs only)

Parent: SFX master plan R1 (`governance/tickets/plans/SFX-MASTER-PLAN.md`), wave W1, units SFX-P2-01..02. Authority: OD-SFX-04 and OD-SFX-15. Execution stops before SFX-P5. No ADR in this ticket may claim `Accepted`; acceptance is OD-SFX-06, a separate operator review.

Documentation only. No path under `intergrax/`, `agents/`, `applications/`, `scripts/` or `agent-env/` is touched.

## Deliverables

1. **ADRs, all with status `Proposed`**, in `docs/project/technical/adr/entries/2026-11-12/`:
   - `ADR-SFX-001.md`: placement and ownership (Option B). Include the ownership review for the dual `ToolRegistry` (SFX-GAP-09), the applicability of ECAP §16 (C-ECAP-01), and the post-freeze class of a new Tier-3 application (ME-1).
   - `ADR-SFX-002.md`: contracts, versioning, canonicalization, scenario workspace location, and the persistence mapping (master plan §7.5).
   - `ADR-SFX-003.md`: Studio stack, authentication and hosting (master plan §7.8, S-1..S-7, ME-2..ME-5).
   - `ADR-SFX-004.md`: IDE toolkit, CLI and MCP surface, and the parity contract PC-1..PC-6 (master plan §7.7).
   - `ADR-SFX-005.md`: catalog sidecar, with `ApplicationManifest` unchanged (master plan §7.6).
   - `ADR-SFX-006.md`:
     - authority levels L0–L3;
     - the two engineering modes and GATE-SEF-CONSUME;
     - the separation between product certification and scenario admission;
     - the semantic gate assessment (master plan §3, §14).
   - Each ADR follows the layout of `ADR-SEF-001.md`: status table, Context, Alternatives, Decision, Consequences, Related. It cites code paths at the reference SHA.
2. **Contract design drafts** in `docs/project/maintainers/qualification/scenario_factory_x/design/`:
   - `schemas/*.schema.json`: JSON Schema draft 2020-12, one file per SFX-CON-01..17. These are design documents, not code.
   - `SFX_CANONICALIZATION.md`: RFC 8785 canonical form, digest rules, and the determinism scope.
   - `SFX_METRICS.md`: definitions of G-1..G-5 and M-1..M-9 (master plan §8, §13), with data sources. Target values are left for OD-SFX-07.
3. **Gap analysis consolidation** `docs/project/maintainers/qualification/scenario_factory_x/SFX_GAP_ANALYSIS.md`:
   - Reconcile SFX-GAP-01..17 with the T-IGX-037-B findings (inventory, IDE facts).
   - Record each gap's final disposition and task mapping.
   - Finalize this file only after T-IGX-037-B lands. The drafting may run in parallel.

## Acceptance

- `check_harness_adr.py` reports no new findings for the six ADRs.
- Every schema is valid JSON with a `$schema` and an `$id` of the form `sfx.<name>.v1`.
- No ADR states `Accepted`.
- No path outside `docs/` or `governance/tickets/` is changed.

## Test contract

```text
uv run python scripts/maintenance/check_harness_adr.py
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- ADR acceptance (OD-SFX-06).
- Implementing any schema as code (SFX-P5+).
- Product code, FRZ changes and merges.

## Amendment 1 (2026-11-12): deliverable location

Operator decision in the planning chat, 2026-11-12. English rendering: "Move [the SFX workstream documents] to the dossier."

The SFX workstream documents are kept in the SFX qualification dossier `docs/project/maintainers/qualification/scenario_factory_x/` instead of `docs/project/capabilities/plan/satellites/scenario_factory/`. The reason is that the PDC-X closed-world scanner (`scripts/qualification/pdc_x_inventory.py`, `DOC_GLOBS` includes `docs/project/capabilities/**/*.md`) cannot classify `SFX_*.md` stems, which turns `test_pdc_x_inventory` red. Moving the files keeps `scripts/` unchanged (OD-SFX-04, OD-SFX-15).

Only the location changes. Content, acceptance criteria and test contract are unchanged. This amendment supersedes the earlier approval line; the ticket must be re-approved against the new digest.
