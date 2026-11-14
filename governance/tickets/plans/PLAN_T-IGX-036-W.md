# PLAN T-IGX-036-W — PDC-P6 acceptance run and EBH-7 re-pin

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6-E**), P6 exit and the Operator decision of 2026-11-12 in `PDC_REG_DIAGNOSIS.md` ("re-pin once, after PDC-X P6 at `INTEGRATED_BASELINE_SHA`"). Dossier: `PDC_EBH7_REPIN_DOSSIER.md` (the re-pin dossier that authorization 8 of `DELEGATION-PDC-STANDING.md` refers to). Requirements: no new requirement rows.

## Problem

1. P6 has no acceptance record: the three repair tickets (T, U, V) closed the findings of the B-P6 report, but no single run shows XD-01..12, the DUP-X/EBH-4 gates and ECAP FULL on the integrated baseline `a156cbbd5`.
2. The EBH-7 certification pins `TEST_EXECUTION_SHA` `dc18e36d9`. The executable-tree identity gate (`verify_ebh7_p03_immutable_evidence.py`) fails whenever a path outside its allowlist changed after that pin, so 11 `ebh_7` tests have been red since `9f9cc8b57`. The Operator decided to re-pin once, after P6, as a fresh L3 replay.

## Deliverables

**Part A: P6 acceptance run** (record only; no product or test change)

1. One dedicated tester session, one process at a time, on a clean worktree of the current `development` whose product paths equal `INTEGRATED_BASELINE_SHA` (verified with `git diff --name-only a156cbbd5..HEAD`).
2. One `uv run pytest tests/qualification/pdc_x -q -rf` (hard wall-clock limit 120 minutes, no parallel pytest, no `-n`), the DUP-X and EBH-4 gate files named in the B-P6 report of T-IGX-036-S, and one `run_enterprise_audit.py run --mode FULL`. Logs under `.tmp/proof/pdc-x-p6/acceptance/`.
3. `PDC_P6_ACCEPTANCE_RECORD.md` states commands, SHAs, counts and every red test with its class (known: IIAC ACT-N11; `ent_audit_x` FREEZE launcher timeout; baseline-red tests listed in the T-IGX-036-T report). A red outside the known classes is a finding routed to the orchestrator; it is not repaired in this ticket. The record carries no PASS verdict.

**Part B: EBH-7 re-pin** (two PRs; paths in the write set only)

4. **Prep PR (W1).** `MANIFEST.json` gets the new `TEST_EXECUTION_SHA` (the `development` tip taken when W1 is cut) while the old pin and the last replay move to `historical_replays`; `tests/qualification/ebh_7/_constants.py` and the three defaults in `.github/workflows/unit-tests.yml` take the same value; the replay record and L3 record state the new roles. Signed historical EBH-7 documents are not edited (their text stays provenance).
5. **Replay.** After W1 merges, the implementer dispatches only `unit-tests.yml` with `profile=ebh7_p3_l3` and the SHA inputs of the dossier, records the run ID, waits for the run and downloads its artifact. Nothing else is sent to an external service.
6. **Record PR (W2).** Replay results (counts, run ID, artifact digests, F10 attestation, skip disposition) go into `MANIFEST.json` `last_replay`, the replay records, the re-rendered `ARTIFACT_DIGESTS.json` (with `render_ebh7_p03_artifact_digests.py`) and the closure evidence audit matrix. The records say "not B6 PASS".
7. After W2 the `ebh_7` suite is run once locally (one `uv run pytest tests/qualification/ebh_7 -q`); the former 11 failures are green or each remaining red is explained.

Design constraints: no change under `intergrax/`, `agents/`, `applications/`; no allowlist widening in `verify_ebh7_p03_immutable_evidence.py`; no weakening of any gate; no new authority; no edit of signed EBH-7 decision, signature or closure records.

## Ordering and risks (from the dossier)

- After the new pin, a change to any path outside the identity allowlist (including `governance/tickets/`, `uv.lock`, `pyproject.toml`, `scripts/` other than the verifier and the renderer, `tests/` other than the listed ebh_7 gate files) fails the gate again. So this ticket's own plan, signature and WBS records merge before the pin tip is taken; no ticket record edit follows, and the pin is taken right before W1 with no dependency-bot merge in between.
- Product repairs triggered by P7 or P8 findings change product paths and need another re-pin. The Operator is told this before signing (see Operator decisions below).

## Operator decisions (asked at signing)

1. Accept a second re-pin if P7 or P8 findings change product paths, or defer W Part B until after P8.
2. Pause dependency-bot merges (`uv.lock`) between the pin and the end of the replay.

## Write set

`docs/project/maintainers/qualification/ebh_7_p03_independent_replay/`, `docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md`, `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`, `docs/project/maintainers/qualification/platform_domain_completeness/PDC_P6_ACCEPTANCE_RECORD.md`, `docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md`, `docs/project/maintainers/qualification/platform_domain_completeness/PDC_REG_DIAGNOSIS.md`, `.github/workflows/unit-tests.yml`, `tests/qualification/ebh_7/_constants.py`

Nothing outside this set changes. A need to write elsewhere (for example another `ebh_7` test or the verifier allowlist) stops the batch and goes to the orchestrator and the Operator.

## Protocol

1. Part A first; the record is committed with W1 or W2, not separately after the pin.
2. One `uv run pytest` per verification step; ruff and `git diff --check` on touched files. Do not pass `-p no:cacheprovider`.
3. Commits carry exactly one `Ref: governance/tickets/T-IGX-036-W.md` trailer and no AI attribution. Push to a branch with upstream set to `origin/development` first, open a PR, never merge; the orchestrator merges docs PRs under `DELEGATION-PDC-STANDING.md` authorization 1 after green CI and a head match.
4. The implementer never certifies, never writes the `Approved:` line and never signs EBH-7 records.

## Acceptance

- `PDC_P6_ACCEPTANCE_RECORD.md` exists with logs and the classified list of reds.
- The replay run is green with a fresh manifest and digests; `ebh_7` P03-B5 identity, B6-F10 matrix and digest tests pass at the new pin.
- No product path, signed record or gate changed.

## Test contract

```text
uv run pytest tests/qualification/pdc_x -q
uv run pytest tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product repairs; `roadmap_replay_x` disposition rows (owner MCP-SERVER-X P7); `dist_scale_x` P4/P6 (Docker, C10/C15); widening the executable-identity allowlist; IIAC activation and header fix (PDC-P7-00); FRZ changes; OD signatures; certification.
