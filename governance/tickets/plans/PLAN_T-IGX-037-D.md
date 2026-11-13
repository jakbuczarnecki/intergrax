# PLAN T-IGX-037-D — SCENARIO-FACTORY-X P3: UX design, mock-ups, usability protocol and generalization-set seal (docs only)

Parent: SFX master plan R1 (`governance/tickets/plans/SFX-MASTER-PLAN.md`), wave W2, units SFX-P3-01..02. Authority: OD-SFX-04 and OD-SFX-15. Execution stops before SFX-P5.

Documentation only. The clickable prototype is a claude.ai Artifact; no prototype code goes into product paths. No path under `intergrax/`, `agents/`, `applications/`, `scripts/` or `agent-env/` is touched.

## Deliverables

1. **`docs/project/maintainers/qualification/scenario_factory_x/SFX_UX_DESIGN.md`** (SFX-P3-01) contains:
   - personas: business user, engineer, approver;
   - journeys across the full lifecycle: Describe → Discover → Understand → Design → Configure → Approve → Build → Test → Run → Present → Improve;
   - the information architecture and a screen inventory;
   - the progressive-disclosure rules;
   - how dynamic forms are generated from JSON Schema;
   - the conversational-refinement pattern;
   - the visual blueprint editor, driven by forms rather than free wiring;
   - the configuration inspector, which shows secret placeholders only;
   - approval inbox, version history and semantic diff;
   - interruption recovery in both front doors;
   - an authority-level badge (L0–L3) on every action;
   - IDE-journey parity notes;
   - accessibility requirements.

   It also links the published prototype Artifact and records its version.
2. **`SFX_USABILITY_PROTOCOL.md`** (SFX-P3-01):
   - 3 rounds of 5 participants each;
   - tasks per reference scenario (REF-1..3);
   - think-aloud procedure, SUS, task success and TTFS measurement;
   - a severity scale and recording rules.

   Round 0 runs on the prototype and records baseline TTFS. Only measured values are recorded; where nothing was measured, the field says `NOT_RUN`.
3. **`SFX_GEN_SET_SEAL.md`** (SFX-P3-02):
   - The seal holds only the digest and the metadata: count, domains and author session.
   - An independent session, not the implementer, architect or orchestrator, writes at least five business problems and computes the digest.
   - That session keeps the content outside the repository until the reveal at SFX-P12-04.

## Acceptance

- Every lifecycle step has a screen or IDE step and names the contract it touches (SFX-CON-*).
- The prototype Artifact link resolves.
- The seal digest is present, and its author session is distinct from the implementer.
- No path outside `docs/` or `governance/tickets/` is changed.

## Test contract

```text
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Implementing the Studio frontend (SFX-P10).
- Revealing the GEN set content.
- Product code, FRZ changes and merges.

## Amendment 1 (2026-11-12): deliverable location

Operator decision in the planning chat, 2026-11-12. English rendering: "Move [the SFX workstream documents] to the dossier."

The SFX workstream documents are kept in the SFX qualification dossier `docs/project/maintainers/qualification/scenario_factory_x/` instead of `docs/project/capabilities/plan/satellites/scenario_factory/`. The reason is that the PDC-X closed-world scanner (`scripts/qualification/pdc_x_inventory.py`, `DOC_GLOBS` includes `docs/project/capabilities/**/*.md`) cannot classify `SFX_*.md` stems, which turns `test_pdc_x_inventory` red. Moving the files keeps `scripts/` unchanged (OD-SFX-04, OD-SFX-15).

Only the location changes. Content, acceptance criteria and test contract are unchanged. This amendment supersedes the earlier approval line; the ticket must be re-approved against the new digest.
