# T-IGX-040-C — PDC-X finding F-56: accept a net-zero scoped path in the `roadmap_replay_x` scoped production-delta check

## Scope

Test-infrastructure repair, by Operator decision `DECISION-PDC-F56.md` (2026-11-14). `test_b6_03_scoped_disposition_valid_at_head` fails on `development` (`83d939095`) with `scoped_path_not_in_prod_diff:intergrax/integrations/providers/embedding_provider/ollama/manifest.py`: the file was added after the DUP-X pin and deleted in T-IGX-036-Z, so its net diff is empty, but the frozen PRE-ENT list still names it. The check `scoped_prod_delta_disposition_violations` in `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py` is loosened minimally: a scoped path that is not in the prod diff is accepted only when it exists neither at HEAD nor at the pin (net-zero add and delete); every other case stays reported. New pinned tests (written first) use throwaway git repositories. The frozen PRE-ENT list data and all existing tests stay byte-identical. Product code: no. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-040-C.md`.

## Plan digest

Plan digest: dea4766d879b36d7f16a71bed8dd54cdb2737078bd32336442e4fde1769e5085

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-C.md`](plans/PLAN_T-IGX-040-C.md).

## Approved

Approved: operator-delegate @ 2026-11-15T06:53:04+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-F56-F35.md@88ee973db2f84a4be63493a40edf1ca0000a745b

Delegate checks (scope of `DELEGATION-PDC-F56-F35.md`, all passed before this line was written; read at the batch branch, base `61531828f`, commits `4ffdff976`, `9ca336a8f`, `88ee973db`):

- The plan text states "signing returns to the Operator"; that sentence predates the delegation. The Operator later answered "Tak, delegat podpisze" (2026-11-15, recorded in `DELEGATION-PDC-F56-F35.md`) after approving the loosening in `DECISION-PDC-F56.md`. The delegate treats the delegation record as the Operator's later, specific instruction; the plan text is not edited (it would change the digest).
- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`dea4766d879b36d7f16a71bed8dd54cdb2737078bd32336442e4fde1769e5085`).
- The write set of the plan equals the diff of commit `4ffdff976` (5 files: `_prod_delta_disposition.py`, the new `test_roadmap_replay_x_scoped_path_net_zero.py`, this ticket, the plan, `DECISION-PDC-F56.md`); it lies only in `tests/qualification/roadmap_replay_x/` and batch documents; no path under `intergrax/`, `agents/`, `applications/`, `scripts/governance/`, `.github/`, `intergrax/contracts/` or `intergrax/applications/_shared/`. The frozen PRE-ENT list `_pre_ent_l3_prod_delta.py` and `test_roadmap_replay_x_closure_counter_integrity.py` are byte-identical to `61531828f`.
- Loosening read in code: only the last `elif` of `scoped_prod_delta_disposition_violations` gains `and not _is_net_zero_absent(repo, row.path)`; that helper is true only when `git ls-tree` finds the path neither at `HEAD` nor at `_DUP_X_PIN`, and a bad revision raises (fail closed). The `wrong_qualification_head` and `missing_disposition` checks stay in the unchanged first branch. Negative tests exist for: path present at HEAD but not in the diff, path existing at the pin and deleted (not net zero), rename-away, wrong head, missing disposition, unknown pin; the positive net-zero case uses throwaway git repositories.
- One `uv run pytest` over the 040-C and 040-D test files (log `.tmp/proof/pdc-x-f56/delegate-verify.log`): `test_b6_03_scoped_disposition_valid_at_head` and all `test_roadmap_replay_x_scoped_path_net_zero.py` tests pass. Five other `roadmap_replay_x` tests are red on `historical_evidence_invalidated_by_later_changes` / `unscoped_prod_delta` (prod delta since DUP-X from other batches): a class this change cannot influence, since it only removes reports of `scoped_path_not_in_prod_diff`.
- No registry, scheduler, engine, trust, policy or audit authority is introduced; the plan states it is not a certification verdict.
- Merge gate (stated as the delegation record requires): the test-only pull request that carries this batch merges under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`, as `DELEGATION-PDC-F56-F35.md` names them: green CI, only the known red classes, `headRefOid` equal to the verified SHA. The EBH-7 identity gate stays red until the campaign's final re-pin (accepted by the Operator). Not decided here: the delegate did not verify that the numbering used by the record matches the numbered items of `DELEGATION-PDC-STANDING.md`.
- The implementer session (`impl-f56`) and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. Write-set amendments are not delegated and return to the Operator. This signature is not a certification verdict.

## Test contract

```text
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_b6_r2_reconciliation.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py -q
uv run ruff check tests/qualification/roadmap_replay_x/_prod_delta_disposition.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py
uv run pyright tests/qualification/roadmap_replay_x/_prod_delta_disposition.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py
```

## Out of scope

- The frozen PRE-ENT list data, every existing test, any product path, `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`.
- Re-pinning EBH-7, the IIAC audit, activation and certification, and every other finding.

## Lifecycle state

plan_approved
