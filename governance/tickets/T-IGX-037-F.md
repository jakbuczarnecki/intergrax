# T-IGX-037-F — SCENARIO-FACTORY-X pre-freeze documentation closeout (docs only)

## Scope

Reconcile the SFX dossier with the published pre-freeze work (PR #482 to #500), record the operator's scope correction (OD-SFX-21), publish the reader-facing Scenario Factory documentation with operator-supplied visuals, and run one independent review. Target status: `SFX_PRE_FREEZE_DOCS_CLOSED`. `SFX_PRE_IMPLEMENTATION_READY` is not granted. Documentation only; before ARCH-FREEZE under OD-SFX-04 and OD-SFX-15. Plan: `PLAN_T-IGX-037-F.md`.

## Plan digest

Plan digest: 2432783743996d1bb753e61fcfb00bdc4c319f87eb788d79934087889282b2d3

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-F.md`](plans/PLAN_T-IGX-037-F.md).

## Approved

Approved: operator-delegate @ 2026-11-13T08:54:39+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-CLOSEOUT.md@7d504964a14ca47713d3067b8de0987d99d0a2c3

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

plan_approved

## Lease holder

Worktree `_wt/intergrax/t-igx-037-f` (detached at `364b3892544535e0969c28cfdcc0d308d565af3f`).

## Checkpoint

| Field | Value |
| --- | --- |
| Baseline | `origin/development` @ `364b3892544535e0969c28cfdcc0d308d565af3f` (PR #502). The plan assumed `dbc6102be`; the only change since is a PDC-X documentation merge. Stage states are unchanged. |
| Deviation from the closeout plan | The scope record is `OD-SFX-21`, not `OD-SFX-17`, because OD-SFX-17..20 are PROPOSED in `SFX_GAP_ANALYSIS.md`. |
| Visual check | Round 1 and 2 failed (V-2, V-9). Round 3 on the final package: all V-items closed, independent audit PASS_WITH_FINDINGS, see "Visual check, round 3". |

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

## Visual check, round 2 (package `scenario_factory_assets_v2.zip`)

The six PNG files were extracted into an empty scratch folder and copied over the old ones under the same names (`docs/project/assets/public/scenario-factory/`). Their pixels were not edited. All six (light and dark, 1600 x 750) were viewed. The canonical mark was compared at 4x with `readme/intergrax-three-entry-points-light.png`.

| # | Result | Evidence |
| --- | --- | --- |
| V-1 | **Closed** | The wordmark reads "INTERGRAX" and the platform labels read "Intergrax" in all six images. |
| V-2 | **OPEN - finding, not approved** | The canonical mark is an outlined, interlocking X-lattice of rounded strokes in a blue to purple gradient, with a light-weight, letter-spaced wordmark. The v2 mark is a flat, solid, faceted chevron X in teal and navy (dark: cyan and blue) with a bold wordmark. It is a different shape and palette. The package notes call it a geometric approximation, not the canonical artwork. The files stay unapproved until the canonical artwork is substituted by the image author. |
| V-3 | Closed | Journey carries the badge "Planned architecture - not yet implemented". |
| V-4 | Closed | Hero badge no longer names the freeze. |
| V-5 | Closed | The factory box reads "Discovery / Specification / Composition / Handoff / Test planning" (ownership) and "Design / Composition / Test planning" (hero). |
| V-6 | Closed | Hero output reads "Checked scenario"; the journey step reads "Check evidence". |
| V-7 | Closed | All six carry "Planned architecture - not yet implemented" and the footer "CONCEPT - NOT A SHIPPED PRODUCT". |
| V-8 | Closed | No edge artifact or stripe is visible. |
| V-9 (new, MEDIUM) | **OPEN** | All six show the tagline "ENTERPRISE AI PLATFORM" under the logo. The public positioning contract defines the category descriptor as "a reusable foundation for governed AI applications" and does not use "Enterprise AI Platform". Remove the tagline or use the contract wording. |
| V-10 (LOW) | Note | Hero boxes have large empty areas. The journey step "Capability" is a shortened "Capability Resolution". The overview already says the picture compresses the journey. No change required. |

Light and dark sets are complete (six files). The hero alt text in README, the overview and the full-size page said "validated scenario" and listed content the image does not show; the independent audit of `54f0de279` found this (M2) and it was corrected in the next commit. Journey and ownership alt texts match their images. Local links of README and the overview were checked by the link-integrity tests.

Campaign state: images are committed on the candidate branch so the candidate is self-consistent, but **publication stays blocked** on V-2 and V-9. No PR is opened. The ticket still has no `Approved:` line (no delegation covers T-IGX-037-F); an operator or a commissioned delegate must write it before a PR.

## Independent audit, round 2 (commit `54f0de279`)

Verdict: **FAIL as a publishable candidate**; text, governance and scope content PASS_WITH_FINDINGS. The round-1 BLOCKER (PNG files missing from the commit) is fixed.

| Finding | Disposition |
| --- | --- |
| H1 = V-2: the X mark is not the canonical mark (confirmed independently; the repo has no official logo file, so the canonical reference is the lockup in `readme/intergrax-three-entry-points-*.png`) | **Open.** Needs the canonical artwork from the image author, or the operator's explicit written acceptance of the approximation. No agent edits pixels. |
| M1 = V-9: tagline "ENTERPRISE AI PLATFORM" conflicts with the positioning contract (category descriptor; "enterprise" needs qualification) | **Open.** Remove the tagline or use the contract wording, or an operator waiver. |
| M2: stale hero alt text in three places | Fixed (text only). |
| L1: ownership image says "READINESS-owned patterns" (internal track name) | Note for the image author, optional. |
| L2: minor rendering (skewed connector arrowhead on ownership, tight step numbers on journey, accent bars slightly past rounded corners on hero light) | Note for the image author, optional. |
| L3, L4 | Unchanged notes. |

Publication path: V-2 and V-9 resolved (new package or written operator decision) -> short re-check -> `Approved:` line for this ticket by the operator or a commissioned delegate -> PR -> merge by the operator -> pin commit (C6) -> closure record. None of these steps ran.

## Visual check, round 3 (package `scenario_factory_assets_final.zip`)

The six PNG files were extracted into an empty scratch folder, their SHA-256 values matched the list in the package's `ASSET_INTEGRATION.md`, and they replaced the previous files under the same names. Pixels were not edited. No text in README, the overview or the full-size pages needed a change; the corrected hero alt text already matches the images.

| # | Result | Evidence |
| --- | --- | --- |
| V-1 | Closed | "INTERGRAX" / "Intergrax" in all six. |
| V-2 | **Closed by removal** | The non-canonical X mark is gone from all six. The top-left lockup is the word "INTERGRAX" as plain bold text. It is not a reproduction of the canonical logo and no image claims to be the logo. |
| V-3..V-8 | Closed | Status badge and footer in all six; neutral labels; "Checked scenario"; no edge artifacts. |
| V-9 | **Closed** | The tagline "ENTERPRISE AI PLATFORM" is gone from all six. |
| V-10 | Notes only | Large empty areas in hero boxes; step "Capability" is short for "Capability Resolution"; ownership image says "READINESS-owned patterns" (internal track name); the ownership connector arrowhead is slightly skewed. Optional for the image author. |

### Independent audit, round 3 (commit `a95c87faa`)

Verdict: **PASS_WITH_FINDINGS**, no BLOCKER and no HIGH. The committed PNG blobs match the package checksums. V-1..V-9 pass on the committed pixels; V-10 stays optional notes (L1: "READINESS-owned patterns", short step label, empty hero areas; L2: skewed connector arrowhead, accent bars slightly past rounded corners, background arcs crossing the footer divider, tight step numbers). The two LOW findings on this ticket (stale checkpoint row, missing state line) are fixed here. The visuals may be published.

Campaign state: text, governance and visuals are audited. **Still required, and not done by the implementer:** an `Approved:` line for this ticket from the operator or a commissioned delegate (no delegation covers T-IGX-037-F), then a PR to `development`, merge by the operator, the pin commit (C6) and the closure record. `SFX_PRE_FREEZE_DOCS_CLOSED` is not reached. `SFX_PRE_IMPLEMENTATION_READY` is not granted. STOP before SFX-P4-B, SFX-P5 and ARCH-FREEZE.

Plan note: the plan text was clarified before approval (roadmap fact and the README test root were added to its scope); the digest above is the current one.
