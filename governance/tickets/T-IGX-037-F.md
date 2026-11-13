# T-IGX-037-F — SCENARIO-FACTORY-X pre-freeze documentation closeout (docs only)

## Scope

Reconcile the SFX dossier with the published pre-freeze work (PR #482 to #500), record the operator's scope correction (OD-SFX-21), publish the reader-facing Scenario Factory documentation with operator-supplied visuals, and run one independent review. Target status: `SFX_PRE_FREEZE_DOCS_CLOSED`. `SFX_PRE_IMPLEMENTATION_READY` is not granted. Documentation only; before ARCH-FREEZE under OD-SFX-04 and OD-SFX-15. Plan: `PLAN_T-IGX-037-F.md`.

## Plan digest

Plan digest: 717327c60f28af837165c25d2803e28d1bab4e8ec3c580bd1bcfc961436164a1

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-F.md`](plans/PLAN_T-IGX-037-F.md).

## Approved

_Not approved._ No `Approved:` line has been written. The implementer does not write one for its own ticket. The ticket waits for an operator or a commissioned delegate (see `governance/tickets/README.md`).

## Test contract

```text
uv run pytest tests/unit/docs/test_public_readme_contract.py tests/unit/docs/test_public_documentation_link_integrity.py tests/unit/docs/test_public_reader_documents_contract.py tests/unit/docs/test_public_roadmap_and_use_cases_contract.py tests/unit/docs/test_proof_library_public_contract.py tests/qualification/roadmap_replay_x -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- SFX code, Studio, services, `agent-env/`, `scripts/`, schemas, freeze contracts; ADR acceptance, FRZ or IIAC promotion, merges, `GATE-SFX-ENTRY`.
- Human usability sessions (OD-SFX-21), the operator walkthrough of IDE toolkit v0, GEN content confidentiality, the 5 known `roadmap_replay_x` group C failures.
- Any generated, redrawn or edited image.

## Lifecycle state

plan_pending

## Lease holder

Worktree `_wt/intergrax/t-igx-037-f` (detached at `364b3892544535e0969c28cfdcc0d308d565af3f`).

## Checkpoint

| Field | Value |
| --- | --- |
| Baseline | `origin/development` @ `364b3892544535e0969c28cfdcc0d308d565af3f` (PR #502). The plan assumed `dbc6102be`; the only change since is a PDC-X documentation merge. Stage states are unchanged. |
| Deviation from the closeout plan | The scope record is `OD-SFX-21`, not `OD-SFX-17`, because OD-SFX-17..20 are PROPOSED in `SFX_GAP_ANALYSIS.md`. |
| Visual check | See "Visual check result" below. Failed: visual publication blocked. |

## Visual check result (C4, implementer pre-check)

Source: the six PNG files supplied by the operator (`scenario-factory-{hero,journey,ownership}-{light,dark}.png`). They were viewed, compared with `docs/project/assets/public/readme/*.png`, and **not edited**. Result: **visual publication blocked**; the PNG files are kept out of the commits until corrected files arrive. The independent reviewer repeats this check (C5).

Corrections requested from the image author (no agent redraws anything):

| # | File(s) | Finding | Requested change |
| --- | --- | --- | --- |
| V-1 | all six | The brand wordmark reads **"Integrax"** (missing the "r"). Correct spelling: **"Intergrax"**. It appears in the logo lockup of every image, in the text "governed Integrax platform" (hero), in the title "Scenario Factory in the Integrax Ecosystem" and the "Integrax Platform" box (ownership). The ownership body text already says "Intergrax", so the image contradicts itself. | Respell every occurrence as "Intergrax". |
| V-2 | all six | The logo is an infinity-loop mark with a title-case wordmark. The existing public images use the X-lattice mark with an upper-case, letter-spaced wordmark "INTERGRAX". The set reads as a different brand system. | Use the repository's mark and wordmark treatment, as in `readme/intergrax-three-entry-points-*.png`. |
| V-3 | journey (light, dark) | No status marker inside the image. The other two images carry a "Planned" badge. | Add a visible "Planned architecture - not yet implemented" badge. |
| V-4 | hero (light, dark) | Badge text "Planned – Post-Arch-Freeze" uses an internal stage term. | Replace with "Planned architecture - not yet implemented". |
| V-5 | ownership (light, dark) | The factory box lists "Orchestration • Qualification". "Orchestration" can read as a second execution engine, and "Qualification" as certification of scenarios. | Replace with neutral terms, for example "Scenario Design • Composition • Handoff • Test planning". |
| V-6 | hero, journey | Wording is stronger than the overview: "design, build and qualify", "Validated Scenario", "Configure and build", "Test and validate". | Prefer "design, compose and check" and "Checked scenario"; keep the planned status. |
| V-7 | all six | Badges read "Planned Architecture" or "Planned - Post-Arch-Freeze" without "not yet implemented". | Use "Planned architecture - not yet implemented". |
| V-8 | journey and ownership | A thin edge artifact along the top edge (light) and a light stripe (journey dark). | Crop or regenerate cleanly. |

Notes (no change requested): the journey image shows seven steps; the complete ten-step journey is in the text, and the overview says so. Alt texts and captions carry the planned status for all three images.

## Independent review (C5) and campaign state

Reviewer: a separate read-only agent session on commit `8a937642f`. Verdict: **FAIL as a mergeable candidate**, one BLOCKER, no pixel edits involved. Text and governance content: PASS_WITH_FINDINGS.

| Finding | Disposition |
| --- | --- |
| BLOCKER: README and overview reference six PNG files that are not in the commit (they are held back by the visual check) | **Open by design.** The candidate must not be pushed or merged until corrected PNG files arrive and are committed. The plan forbids a docs-without-images wave and forbids replacement images. |
| HIGH: V-1..V-5 confirmed independently | Open. Corrections go to the image author (table above). |
| V-6 (MEDIUM): hero text says "design, build and qualify", output box says "Validated Scenario", journey says "Configure and build" / "Test and validate" (stronger than the overview) | Added to the correction list: prefer "design, compose and check" and "Checked scenario". |
| V-7, V-8 (LOW): badges lack "not yet implemented"; thin edge artifact on journey and ownership images | Added to the correction list. |
| MEDIUM: roadmap text pointed to the SHA ladder for merge SHAs | Fixed: it now points to `SFX_INDEX.md` and says the ladder is pinned after the merge. |
| MEDIUM: roadmap §3.0.7b edit was not listed in the plan scope | The plan scope item 1 and the original closeout plan name "roadmap §3.0.7b". The edit is descriptive, adds no status and the checkbox is unchanged. Recorded here. |
| LOW: SFX subsection sits under the evidence H2 in README | Kept. The text says "not a Scenario Proof" and the row is in the strategic table. Revisit when the images return. |
| LOW: noisy test diff | The test file had to be `ruff format`-clean for the pre-commit gate. Only the allowed visual root is substantive. |
| LOW: "before the platform architecture freeze" wording | Removed from the overview. |

Campaign state: `SFX_PRE_FREEZE_DOCS_CLOSED` is **not reached**. C6 (SHA pin, closure record) did not run because nothing is merged. `SFX_PRE_IMPLEMENTATION_READY` is not granted and `PRE_IMPLEMENTATION_READY_SHA` stays `_unset_`.

