# PLAN T-IGX-041-J — PDC-X batch B-P7-RP: the final EBH-7 re-pin after all P7 repairs (W1' prep, one replay dispatch, W2' record)

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P7-RP**, wave W15). Authority for the sequence: decision 6 of `operator_decisions/DELEGATION-PDC-WAVE2.md` ("one re-pin after all P7 repairs and one after P8 if product paths changed"), decision 7 of the same record (dependency-bot pull requests held until the final re-pin is recorded, then merged in one batch) and the dossier `PDC_EBH7_REPIN_DOSSIER.md`. Precedent: ticket T-IGX-036-W (batch B-P6-E, `PLAN_T-IGX-036-W.md`), whose prep step W1 is commit `0b62781a2` (PR 538 per `PLAN_T-IGX-036-Y.md`). This plan is not a certification verdict and issues no `Approved:` line. Every fact below was read or run in the planning session at `origin/development` `63005e290` (the merge of PR 576, T-IGX-041-I); the implementer re-verifies each before acting.

## Problem

1. **The pin is stale by design.** `TEST_EXECUTION_SHA` is `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` at `MANIFEST.json:2`, `tests/qualification/ebh_7/_constants.py:9`, and five lines of `.github/workflows/unit-tests.yml` (61, 804, 805, 808, 810). The executable-tree identity gate (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py`, allowlist at lines 20-37: the prefixes `docs/project/maintainers/qualification/`, `.github/`, the verifier itself, and nine exact files) rejects every changed path outside that allowlist between the pin and the replay head. Measured at `4a0f2e197` (the `development` tip one merge before PR 576), `git diff --name-only 146e7acf1..4a0f2e197` lists 960 paths, of which 326 lie outside the allowlist (179 under `tests/`, 81 under `intergrax/`, 41 under `governance/tickets/`, 11 under `docs/` outside the allowlisted prefix, 9 under `applications/`, 3 under `scripts/`, 1 under `agent-env/`, 1 under `platform_proofs/`). So the identity gate is red on `development` and stays red until the pin moves.
2. **The P6 replay was never recorded.** `MANIFEST.json` `pytest_profile.last_replay` still has `status` `PENDING` (lines 21-27). The one replay that ran at the pin (GitHub run `43576897070`, profile `ebh7_p3_l3`) ended `6 failed, 619 passed, 9 skipped`; the only place on `development` that names it is `governance/tickets/plans/PLAN_T-IGX-036-Y.md:27`. The six failures were the subject of T-IGX-036-Y (F10 classifier, closed-stage evidence, replay-profile coverage), merged as `d13cb7eb9`.
3. **Two checks the re-pin depends on are red today for reasons outside the allowlist** (measured at `63005e290`, one process each, logs under `.tmp/session/041j/`):
   - F10 inputs: `build_canonical_f10_inputs()` sets `accepted_baseline_sha` to `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92`; of the 3338 paths in `git diff --name-only 5a3a05347..63005e290` exactly 3 are classified UNKNOWN (the gate itself evaluates `accepted_baseline_sha..TEST_EXECUTION_SHA`, `_ebh7_f10_canonical_inputs.py:991`, which is 2488 paths against today's pin `146e7acf1` and becomes `..P` after the re-pin) by `classify_change` (`tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py:654`): `intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py`, `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py`. `PLAN_T-IGX-036-Y.md` ("Sequencing facts", last bullet) says the re-pin session must run `certify_f10()` first and that closing an unknown path needs one more directory entry in the classifier, never a default change. The classifier input file is `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` (C4 rules from line 236), which is **not** on the identity allowlist.
   - `tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py::test_cx_r3_r1_combined_gate_passes_on_current_head` fails with `raw_candidates 7417 != accepted current-head 7356` (and `exclusions 5357 != 5325` and further counters); `tests/qualification/compat_x/test_compat_x_deprecation_r5_p0_gates.py::test_r5_p0_closed_world_gate_passes_on_current_head` fails as well. These are item D1 of `PLAN_T-IGX-036-Z.md` (line 53), which says the counters are "re-pinned once, together with the final EBH-7 re-pin, after the last product change". One pin site is `tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py` (`ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES = 7356`, line 41); the shim reconciliation lives in the sibling `compat_x` modules and is determined when the work is done.

## What decides the order (three facts)

- The replay job checks out `P03_REPLAY_HEAD_SHA`, which defaults to `github.sha` (`unit-tests.yml:812`), i.e. the `development` tip at dispatch, and the identity step compares it with `TEST_EXECUTION_SHA`. So the pin `P` is a commit that contains **every** non-allowlisted change, and the commits after `P` up to the dispatch may change allowlisted paths only.
- `tests/qualification/compat_x/` and `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` are not allowlisted. Their re-pin edits therefore must be merged **before** `P` is taken. They cannot be part of W1' (which is cut from `P`) and cannot be part of W2' (after the replay, any such change makes the identity test of `ebh_7` red again on `development`).
- `governance/tickets/` and `scripts/` are not allowlisted either: this ticket, its plan, its WBS row and the generator change are merged before `P`, and no ticket or plan record is edited afterwards.

## Preconditions before `P` (not part of this ticket's write set)

| # | Precondition | State at `63005e290` | Owner |
| --- | --- | --- | --- |
| 1 | All P7 repair tickets T-IGX-041-A..I merged | open pull requests: none (`gh pr list --state open`); PR 576 (T-IGX-041-I, TENANT-X re-pin) is merged | orchestrator |
| 2 | F10 classifier covers the 3 unknown paths of Problem 3 (one directory or file entry each, a documentation-and-test-only change in `_ebh7_f10_canonical_inputs.py`, never a changed default) | open, ticket T-IGX-041-K | delegate-signed ticket T-IGX-041-K, see "Authority check" |
| 3 | COMPAT-X counters (Z-D1) re-pinned at the last product change | open, ticket T-IGX-041-L | delegate-signed ticket T-IGX-041-L, see "Authority check" |
| 4 | This ticket signed and its records merged | this plan | delegate, see "Authority check" |
| 5 | No dependency-bot or `uv.lock` change between the cut of W1' and the recording of W2' | no open dependency-bot pull request at planning time; hold per decision 7 of `DELEGATION-PDC-WAVE2.md` | orchestrator |
| 6 | Any re-audit loop correction of a P7 domain merged | not yet known | orchestrator |

## How `P` is chosen

`P` is `git rev-parse origin/development` taken at the moment W1' is cut, and only when all of these hold (the implementer records the commands and outputs in the ticket checkpoint):

1. Preconditions 1-6 hold; `gh pr list --state open` shows no pull request that would change a non-allowlisted path and no dependency-bot pull request that is not held.
2. `git diff --name-only <last merged P7 repair merge commit>..P` contains only paths that are allowlisted or that the signed tickets of the preconditions account for (a path list in the checkpoint, each path assigned to a ticket). Any other path stops the batch.
3. The F10 inputs are evaluated once at `P` (the planning session built them with `build_canonical_f10_inputs()` and ran `classify_change` over `git diff --name-only <accepted baseline> HEAD`, from a probe script under `.tmp/session/`, not committed): 0 UNKNOWN paths in C4.
4. The two `compat_x` tests of Problem 3 pass at `P` (one `uv run pytest` over both).
5. Between the cut and the merge of W1' nothing else is merged (freeze); if anything lands, `P` moves and W1' is redone.

## Deliverables

**W1' — prep pull request (allowlisted paths only), cut from `P`**

1. `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json`: `TEST_EXECUTION_SHA` = `P`; `EVIDENCE_PUBLICATION_SHA` = `P` (the W1' merge commit cannot be pinned inside its own manifest, the reason recorded in decision 1 of `DELEGATION-PDC-WAVE2.md` for the previous pin); `TEST_EXECUTION_NOTE` rewritten; the current `last_replay` (role `PDC_P6_REPIN_REPLAY`, status `PENDING`, pin `146e7acf1`) moves to `historical_replays` together with the run `43576897070` result `6 failed, 619 passed, 9 skipped` as history; the new `last_replay` is `PENDING` without counts. `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` stays `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (the integrity test requires it to equal or be an ancestor of the pin).
2. `tests/qualification/ebh_7/_constants.py` lines 9-10 take `P`.
3. `.github/workflows/unit-tests.yml` lines 61, 804, 805, 808, 810 take `P` (concurrency key, the two `EBH7_DEFAULT_*` values and the two input fallbacks).
4. Re-rendered or edited current-role records, as in the W1 precedent `0b62781a2`: `EBH_7_SHA_LADDER.md`, `EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` and `EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md` (through `scripts/qualification/render_ebh7_f10_normative_matrix.py`, never by hand), `ebh_7_p03_independent_replay/L3_REPLAY_RECORD.md`, `EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`, `B6_EVIDENCE_CONSISTENCY_MATRIX.md`, `EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`, `ARTIFACT_DIGESTS.json` (with `scripts/qualification/render_ebh7_p03_artifact_digests.py`, in a separate commit after the content commit, the pattern of `d018208c7`, `08c398e76`, `9fd8cbe0d`), `platform_domain_completeness/PDC_SHA_LADDER.md` and `PDC_REG_DIAGNOSIS.md`. Signed historical EBH-7 documents (about 25) keep their text as provenance.
5. No edit of `_ebh7_p03_evidence_integrity.py`, `test_ebh_7_b6_f10_f11_corrective_gates.py` or any other test. (PR 538 changed those two files with a one-time ratification in decision 2 of `DELEGATION-PDC-WAVE2.md`; that ratification does not extend to this ticket. If a test change proves necessary the batch stops and the need goes to the orchestrator and the Operator.)

**Dispatch — exactly one external action, after W1' merges**

6. `gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3 -f ebh7_test_execution_sha=<P> -f ebh7_evidence_publication_sha=<P>`. Both inputs are `P`, as decision 1 of `DELEGATION-PDC-WAVE2.md` did for the previous pin and as the manifest of item 1 states; the replay head is the W1' merge commit through `github.sha`. (The brief of this ticket named the W1' merge as the publication input; that would disagree with the manifest. The plan follows the Operator-accepted form and the point is answered in `DELEGATION-PDC-P7-REPIN.md`, see "Authority check".) The implementer records the run id, waits for the run and downloads the artifact (`junit.xml`, `meta.txt`, `junit-summary.json`, `skip-disposition.json`, `f10-attestation.json`, `pytest.log`) with SHA-256 values. Nothing else is sent to an external service.

**W2' — record pull request (docs only)**

7. `MANIFEST.json` `last_replay` with run id, counts, artifact digests; the replay records of item 4; a re-rendered `ARTIFACT_DIGESTS.json` and closure matrix; `platform_domain_completeness/PDC_EBH7_FINAL_REPIN_RECORD.md` (new) stating `P`, the W1' merge, the run id and result, the artifact SHA-256 values, the previous run `43576897070` `6 failed, 619 passed, 9 skipped` as history, the list of paths changed since `146e7acf1` outside the allowlist with the ticket that owns each, and the explicit wording "not B6 PASS": the verdict belongs to the certifier and the Operator.
8. After W2' one local `uv run pytest tests/qualification/ebh_7 -q`; each remaining red is explained. The F10 numbers the T-IGX-036-Y plan expects at the final re-pin (`PLAN_T-IGX-036-Y.md` line 64: C1 32/30/2/0, C2 43/43/0/0, C3 105/99/6/0, C4 10/10/0/0, C5 3 PASS, disposition QUALIFIED) are an expectation to compare with, not a promise.
9. The dependency-bot and `uv.lock` pull requests held under decision 7 of `DELEGATION-PDC-WAVE2.md` are listed by the orchestrator (`gh pr list` shows none open at planning time) and merged in one batch **after** W2' is recorded; they are not part of this ticket's write set.

Design constraints: no change under `intergrax/`, `agents/`, `applications/`, `scripts/` (except the one renderer run), no allowlist widening in `verify_ebh7_p03_immutable_evidence.py`, no weakened gate, no new registry, scheduler, engine, trust, policy or audit authority, no edit of signed EBH-7 decision, signature or closure records, no edit of `governance/tickets/` after `P`.

## Write set

```text
docs/project/maintainers/qualification/ebh_7_p03_independent_replay/
docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md
docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md
docs/project/maintainers/qualification/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md
docs/project/maintainers/qualification/platform_domain_completeness/PDC_SHA_LADDER.md
docs/project/maintainers/qualification/platform_domain_completeness/PDC_REG_DIAGNOSIS.md
docs/project/maintainers/qualification/platform_domain_completeness/PDC_EBH7_FINAL_REPIN_RECORD.md
.github/workflows/unit-tests.yml
tests/qualification/ebh_7/_constants.py
```

Batch documents of the planning commit (merged before `P`, not part of W1'/W2'): `governance/tickets/T-IGX-041-J.md`, `governance/tickets/plans/PLAN_T-IGX-041-J.md`, `scripts/qualification/pdc_x_wbs.py` and the regenerated `PDC_IMPLEMENTATION_WBS.md` and `PDC_DEPENDENCY_DAG.md`. Nothing else changes. A need to write elsewhere (another `ebh_7` test, the verifier allowlist, a signed record, a product path, `compat_x`, the F10 classifier) stops the batch and goes to the orchestrator and the Operator.

## Protocol

1. Token-budget preflight (read scope, edit scope, tests) before the first edit; one `uv run pytest` per verification step; logs under `.tmp/proof/pdc-x-041j/`; ruff and `git diff --check` on touched files.
2. W1' and W2' are separate branches from the then `development` tip, each with its own pull request; commits carry exactly one `Ref: governance/tickets/T-IGX-041-J.md` trailer and no AI attribution; upstream set to `origin/development` before the first push; explicit refspec; never merge (the orchestrator merges docs pull requests under authorization 1 of `DELEGATION-PDC-STANDING.md` after green CI and a head match).
3. The implementer never writes `Approved:` lines, never certifies and never signs EBH-7 records.
4. Expected red at W1' merge: nothing new on the identity class (the pin now equals the tree); the IIAC activation staleness (ACT-N11) stays red until the P8-04 rebind.

## Acceptance

- `MANIFEST.json`, `_constants.py` and the five workflow lines carry the same `P`; the identity step and `ebh_7` tests at the W1' head report no violation.
- The replay run exists, its artifact SHA-256 values are recorded, and the record states the result without a PASS verdict. The result may be green or red; a red result is recorded as it is and routed, not repaired in this ticket.
- The previous run `43576897070` and the P6 pin `146e7acf1` appear only as history.
- No path outside the write set changed in W1' or W2'; no gate weakened.

## Test contract

```text
uv run pytest tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Plus the one `gh workflow run unit-tests.yml ...` dispatch of Deliverable 6 and its recorded result.

## Authority check

The table states what the delegation records on `origin/development` covered when this plan was written (`63005e290`).

| Question | Reading of the records | Covered? |
| --- | --- | --- |
| (a) A delegate session signs this ticket | `DELEGATION-PDC-P6-REPIN.md` authorizes delegate signing of **T-IGX-036-W only** ("Scope of the delegation"); `DELEGATION-PDC-P7-REPAIR.md` authorizes T-IGX-041-A..I only; decision 6 of `DELEGATION-PDC-WAVE2.md` lets a delegate sign plans of **repair tickets of the owning domain**; `DELEGATION-PDC-STANDING.md` answer 4 names plans M and S and write-set amendments. None names T-IGX-041-J. | **No.** Operator decision needed. |
| (b) Editing `.github/workflows/unit-tests.yml` for the pins | The path is on the identity allowlist (`.github/`), and the previous re-pin edited it under T-IGX-036-W. The delegation that signed that ticket does not extend to a new ticket. | **Only through the signature of (a).** |
| (c) A second re-pin | Decision 6 of `DELEGATION-PDC-WAVE2.md` pre-authorizes "one re-pin after all P7 repairs and one after P8 if product paths changed"; decision 1 of `DELEGATION-PDC-P6-REPIN.md` accepts further re-pins. This ticket is the one after all P7 repairs. | **Yes** for the re-pin itself. |
| Dispatch of the replay at `P` | Answer 8 of `DELEGATION-PDC-STANDING.md` allows dispatching "the replay workflows named in the EBH-7 re-pin dossier, at the integrated baseline SHA" after P6; decision 1 of `DELEGATION-PDC-WAVE2.md` names the pin `146e7acf1`. `P` is a different SHA. | **Not stated.** Operator decision needed. |
| Evidence-publication input equal to `P` | Decision 1 of `DELEGATION-PDC-WAVE2.md` for the previous pin; the dossier says the W1 merge commit. | **Confirm** which form applies to `P`. |
| F10 classifier entries for the 3 unknown paths and the COMPAT-X counter re-pin as separate tickets | Decision 3 of `DELEGATION-PDC-WAVE2.md` covers F10 classifier work (plan signed by a delegate); `PLAN_T-IGX-036-Z.md` D1 says the counters are re-pinned together with the final re-pin but lists reopen conditions (closed accepted stage, the `roadmap_replay_x` shim delta owned by MCP-SERVER-X P7, `mcp_host_mount.py` under the `_shared` hotspot, a new `RuntimeEvent` writer needing a fix or a classification with new evidence). | **Classifier: by analogy to decision 3. Counters: not clearly covered.** Operator decision needed. |

Superseded by the Operator record `operator_decisions/DELEGATION-PDC-P7-REPIN.md` (branch `pdc-x-p7-repin-record`, tip `664b054ec`, PR 577; not yet on `development` when this paragraph was written). The Operator answered "Yes, I approve (a)-(c)": (a) a delegate session may sign T-IGX-041-J, T-IGX-041-K (classifier entries for the three UNKNOWN paths) and T-IGX-041-L (COMPAT-X counter re-pin); (b) the replay is dispatched at `P`; (c) both SHA inputs equal `P`. The rows above that read "No", "Not stated" or "Confirm" are therefore answered; the `Delegation:` line of the signature cites the commit of that record. The ordering rule of that record (tickets K and L and the signed records of J merge before `P`) is the one of "Preconditions before `P`".

## Out of scope

Everything listed under "Preconditions" as owned elsewhere; product repairs; `roadmap_replay_x` dispositions; `dist_scale_x` and Docker (C10/C15); widening the executable-identity allowlist; IIAC activation and rebind (PDC-P7-00, PDC-P8-04); the P8 re-pin; FRZ changes; OD signatures; certification and any B6 verdict. This plan is not a certification verdict.
