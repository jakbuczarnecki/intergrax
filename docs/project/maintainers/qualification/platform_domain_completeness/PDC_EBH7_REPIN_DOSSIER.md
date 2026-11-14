# PDC-X — EBH-7 re-pin dossier

Dossier named by authorization 8 of `operator_decisions/DELEGATION-PDC-STANDING.md`. Ticket: T-IGX-036-W (batch B-P6-E). Facts below were read at `origin/development` `282f6dc68`; the implementer re-verifies them before acting. This dossier changes nothing by itself.

## Why a re-pin

`PDC_REG_DIAGNOSIS.md` groups B1 and B2: the executable-tree identity gate compares the pinned `TEST_EXECUTION_SHA` `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` with the checkout and rejects any changed path outside its allowlist; `ARTIFACT_DIGESTS.json` still pins an older digest of the closure evidence audit matrix. Operator decision (2026-11-12, chat, English rendering): re-pin once, after PDC-X P6, as a fresh L3 replay with a new manifest, acceptance record and re-rendered digests.

## Where the pin is live

| Site | Role | Action in W |
| --- | --- | --- |
| `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json` | `TEST_EXECUTION_SHA`, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, `EVIDENCE_PUBLICATION_SHA`, `last_replay` | new pin; old values kept as history |
| `tests/qualification/ebh_7/_constants.py` | `EBH7_TEST_EXECUTION_SHA` | new pin |
| `.github/workflows/unit-tests.yml` (3 lines: concurrency key, `EBH7_DEFAULT_TEST_EXECUTION_SHA`, default input) and the evidence-publication default | CI defaults | new pin |
| `ebh_7_p03_independent_replay/` replay and L3 records, `ARTIFACT_DIGESTS.json`, closure evidence audit matrix | current-role tables, digests | updated and re-rendered |
| about 25 signed EBH-7, hist02/hist04 and `pre_ent` documents | historical references | **not edited** (provenance) |

## Allowlist and ordering

The gate allows only `docs/project/maintainers/qualification/`, `.github/`, the verifier script and nine named files (seven under `tests/qualification/ebh_7/`, the roadmap document and the renderer script). Observed at `282f6dc68`: since `a156cbbd5` only `scripts/qualification/pdc_x_inventory.py` (PR #527) and `uv.lock` (dependency bot, PR #526) changed outside docs. Both are on the wrong side of the allowlist, so the pin cannot be `a156cbbd5`; it is the `development` tip taken when W1 is cut, whose product paths equal `INTEGRATED_BASELINE_SHA` (check with `git diff --name-only a156cbbd5..<tip>`). After the pin only allowlisted paths may change until the replay is recorded; `governance/tickets/` is not allowlisted, so ticket and plan records of W are final before the pin.

## Sequence

1. Ticket W signed; its records merged. Part A acceptance run (record only).
2. Cut W1 from the then `development` tip `P`. `TEST_EXECUTION_SHA = P` (a docs-only descendant of `a156cbbd5` plus the two paths above). W1 edits only allowlisted paths. Merge W1; its merge commit is `EVIDENCE_PUBLICATION_SHA`.
3. Dispatch `gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3 -f ebh7_test_execution_sha=<P> -f ebh7_evidence_publication_sha=<W1 merge>`. This is the only external dispatch. The previous replay (run `41238620036`) used the same two inputs, with the replay checkout a later docs-only head.
4. Download the artifact (`junit.xml`, `meta.txt`, `junit-summary.json`, `skip-disposition.json`, `f10-attestation.json`, `pytest.log`) and record SHA-256 values.
5. W2 (docs only) records the run in `MANIFEST.json` `last_replay`, replay records, re-rendered `ARTIFACT_DIGESTS.json` and the matrix. Statements stay "not B6 PASS"; the verdict belongs to the certifier and the Operator.

## Risks

- A path outside the allowlist changed after `P` (dependency bot, ticket edit, script change) fails the identity gate; the run is then not valid and the pin moves.
- Product repairs from P7/P8 findings require another re-pin (Operator decision 1 in the plan).
- The replay needs `unit-tests.yml` on `development` to carry the new defaults; W1 provides them.

## Not covered

`roadmap_replay_x` disposition rows (MCP-SERVER-X P7), `dist_scale_x` (Docker), any change of the allowlist.
