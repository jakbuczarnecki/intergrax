# T-IGX-041-J — PDC-X batch B-P7-RP: the final EBH-7 re-pin after all P7 repairs

## Scope

One fresh EBH-7 L3 replay pin at the `development` tip `P` taken after the last test and product merge of the P7 repair loop, as decision 6 of `DELEGATION-PDC-WAVE2.md` provides ("one re-pin after all P7 repairs"). Two pull requests and one external dispatch: W1' (allowlisted paths only: `MANIFEST.json` pins, `tests/qualification/ebh_7/_constants.py`, the five pin lines of `.github/workflows/unit-tests.yml`, replay and L3 records, the re-rendered F10 matrix and ECAP report block, `ARTIFACT_DIGESTS.json`, SHA ladders), the single `gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3 -f ebh7_test_execution_sha=<P> -f ebh7_evidence_publication_sha=<P>` dispatch, and W2' (docs only: run id and result, artifact SHA-256 values, the earlier run `43576897070` `6 failed, 619 passed, 9 skipped` kept as history, status "not B6 PASS"). The plan also states how `P` is chosen and names two preconditions that are outside this ticket's write set and must merge before `P`: the F10 classifier entries for three unknown changed paths and the COMPAT-X counter re-pin (D1 of `PLAN_T-IGX-036-Z.md`). Product code: no. This ticket is not a certification verdict and not a B6 PASS. Plan: `PLAN_T-IGX-041-J.md`.

## Plan digest

Plan digest: 70b23e4fa18f7dbb41b161f0d7496111a885169ee1c95750cc6c17bc0c261a81

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-J.md`](plans/PLAN_T-IGX-041-J.md).

## Approved

Approved: operator-delegate @ 2026-11-15T13:40:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-REPIN.md@664b054ec20c8964edb8b3b5c786e1769fa67d70

Delegate checks (scope of `DELEGATION-PDC-P7-REPIN.md`, all passed before this line was written; the record is on `origin/development` since PR 577, last changed by `664b054ec`; plan read at commit `78099ed1c` of the batch branch). A first signing attempt at `54c879103` was refused because Problem 3 stated "3338 paths in `git diff --name-only dc18e36d9..HEAD`" (actual 3124 for that range; 3338 belongs to `5a3a05347..63005e290`) and referred to a non-existent section "Operator decisions needed"; the implementer amended the plan and the checks were redone from scratch:

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`70b23e4fa18f7dbb41b161f0d7496111a885169ee1c95750cc6c17bc0c261a81`).
- The write set of the plan (9 entries) equals the write set of batch row `B-P7-RP` in `PDC_IMPLEMENTATION_WBS.md` (compared as lists, equal). It contains nothing under `intergrax/`, `agents/`, `applications/` or `scripts/governance/`; under `.github/` only `.github/workflows/unit-tests.yml`, and the plan names its five pin lines (61, 804, 805, 808, 810), which I confirmed hold `146e7acf1` (concurrency key, the two `EBH7_DEFAULT_*` values, the two input fallbacks); `P03_REPLAY_HEAD_SHA` defaults to `github.sha` at line 812.
- Code and history claims re-verified: `MANIFEST.json:2` and `last_replay` (lines 21-27, `PENDING`, role `PDC_P6_REPIN_REPLAY`); `_constants.py:9-10`; the allowlist of `verify_ebh7_p03_immutable_evidence.py` lines 20-37 (3 prefixes, 9 exact files); `git diff --name-only 146e7acf1..4a0f2e197` = 960 paths, 326 outside the allowlist, with the per-directory breakdown of the plan; the F10 range figures of the amended sentence: `accepted_baseline_sha` is `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` (`_constants.py:15`, `_ebh7_f10_canonical_inputs.py:942`), `5a3a05347..63005e290` = 3338 paths of which exactly the 3 named paths are UNKNOWN under `classify_change` (re-run by the delegate with `build_canonical_f10_inputs()`), `5a3a05347..146e7acf1` = 2488 paths, the gate's range is `accepted..test_sha` (`_ebh7_f10_canonical_inputs.py:991`); `classify_change` at `_ebh7_f10_ecap_normative_matrix.py:654`, C4 rules from line 236; the two `compat_x` gate tests fail on `raw_candidates 7417 != 7356` and `exclusions 5357 != 5325` (delegate run), pin at `_compat_x_semantic_identity_reconciliation.py:41`; run `43576897070` is named on `development` only in `PLAN_T-IGX-036-Y.md:27`; W1 precedent `0b62781a2`, digest re-render commits `d018208c7`, `08c398e76`, `9fd8cbe0d`, F10 commit `d13cb7eb9`; `PLAN_T-IGX-036-Y.md` sequencing bullet (line 30) and expected numbers (line 64); `PLAN_T-IGX-036-Z.md` D1 (line 53).
- Delegation text read against the plan: decisions 1, 2, 3, 6, 7 of `DELEGATION-PDC-WAVE2.md` and answers 4 and 8 of `DELEGATION-PDC-STANDING.md` read as the plan quotes them; the dispatch is `gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3 -f ebh7_test_execution_sha=<P> -f ebh7_evidence_publication_sha=<P>`, both inputs equal `P`, as part (c) of the record; `P` is the `development` tip taken after the last test or product merge (plan "How `P` is chosen"). Ordering: the preconditions T-IGX-041-K (F10 classifier entries for the three UNKNOWN paths) and T-IGX-041-L (COMPAT-X counter re-pin) and the signed records of J merge before `P` (plan "What decides the order" and "Preconditions before `P`"), as the record's ordering rule requires, because `governance/tickets/` and `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` are outside the identity allowlist.
- No gate is weakened or deleted: the plan changes pin values only, edits no test, no verifier allowlist and no signed record, and records the earlier run `43576897070` (`6 failed, 619 passed, 9 skipped`) and the pin `146e7acf1` as history. No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict and "not B6 PASS".
- The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. Not checked by the delegate: the open-pull-request and dependency-bot state at the cut of `P` (the plan makes it a precondition for the implementer to record).
- Merge gate (as the record states): the pull requests merge under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes, head equals the verified SHA. The replay is dispatched once; run id and artifact SHA-256 values are recorded in a docs-only pull request stating "not B6 PASS". Write-set amendments are not delegated.

## Test contract

```text
uv run pytest tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Plus the one `unit-tests.yml` dispatch (`profile=ebh7_p3_l3`) described in the plan and its recorded result.

## Out of scope

- The two preconditions (F10 classifier entries, COMPAT-X counters), product repairs, `roadmap_replay_x` dispositions, `dist_scale_x` (Docker, C10/C15), widening the executable-identity allowlist, IIAC activation and rebind, the P8 re-pin, FRZ changes, OD signatures, certification and any B6 verdict.

## Lifecycle state

plan_approved
