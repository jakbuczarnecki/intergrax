# T-IGX-041-J — PDC-X batch B-P7-RP: the final EBH-7 re-pin after all P7 repairs

## Scope

One fresh EBH-7 L3 replay pin at the `development` tip `P` taken after the last test and product merge of the P7 repair loop, as decision 6 of `DELEGATION-PDC-WAVE2.md` provides ("one re-pin after all P7 repairs"). Two pull requests and one external dispatch: W1' (allowlisted paths only: `MANIFEST.json` pins, `tests/qualification/ebh_7/_constants.py`, the five pin lines of `.github/workflows/unit-tests.yml`, replay and L3 records, the re-rendered F10 matrix and ECAP report block, `ARTIFACT_DIGESTS.json`, SHA ladders), the single `gh workflow run unit-tests.yml --ref development -f profile=ebh7_p3_l3 -f ebh7_test_execution_sha=<P> -f ebh7_evidence_publication_sha=<P>` dispatch, and W2' (docs only: run id and result, artifact SHA-256 values, the earlier run `43576897070` `6 failed, 619 passed, 9 skipped` kept as history, status "not B6 PASS"). The plan also states how `P` is chosen and names two preconditions that are outside this ticket's write set and must merge before `P`: the F10 classifier entries for three unknown changed paths and the COMPAT-X counter re-pin (D1 of `PLAN_T-IGX-036-Z.md`). Product code: no. This ticket is not a certification verdict and not a B6 PASS. Plan: `PLAN_T-IGX-041-J.md`.

## Plan digest

Plan digest: 9e41a6e7e1b86cb6205aeab5f6548c12b018c9fb89c7862638ab8210bd97b06f

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-J.md`](plans/PLAN_T-IGX-041-J.md).

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

plan_pending
