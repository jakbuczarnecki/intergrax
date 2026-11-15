# PLAN T-IGX-040-C — PDC-X finding F-56: accept a net-zero scoped path in the `roadmap_replay_x` scoped production-delta check

Parent: PDC-X campaign, closing wave. Origin: finding F-56 of the independent audit (addendum 4), assigned to PDC-X for repair; Operator decision `DECISION-PDC-F56.md` (2026-11-14, "loosen the check minimally"). This plan is written by the implementer lane and is **unsigned** (`plan_pending`). It is not a certification verdict and certifies nothing.

## Problem

On `development` tip `83d939095`, `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py::test_b6_03_scoped_disposition_valid_at_head` fails with `scoped_path_not_in_prod_diff:intergrax/integrations/providers/embedding_provider/ollama/manifest.py`.

- `scoped_prod_delta_disposition_violations` in `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py` reports every row of the frozen PRE-ENT list (`_pre_ent_l3_prod_delta.py`) whose path is not in `git diff --name-only <_DUP_X_PIN>..HEAD -- intergrax agents applications`.
- `intergrax/integrations/providers/embedding_provider/ollama/manifest.py` was added after the DUP-X pin (`a7d83925b375b5f08eb0f175ced2c17fa026bba9`; the pin tree has no such file) and deleted in T-IGX-036-Z (commit `df436b7d1`). Its net diff against the pin is empty, so it is not in the diff, while the frozen list still names it. The check therefore reports a path that has no delta at all.

## Deliverable

One minimal loosening of the `elif` branch that reports `scoped_path_not_in_prod_diff`, in `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py`:

- Two private helpers: `_path_in_tree(repo, rev, path)` (`git ls-tree --name-only <rev> -- <path>` with `check_output`, so a bad revision raises and the check fails closed) and `_is_net_zero_absent(repo, path)` (true only when the path is absent at `HEAD` **and** absent at `_DUP_X_PIN`).
- The branch becomes `elif row.path in REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS and not _is_net_zero_absent(repo, row.path):` — a scoped path that is not in the prod diff is accepted only when it is net-zero (added after the pin and deleted again).
- Every other case stays reported: the path exists at HEAD but not in the diff; the path existed at the pin and is absent at HEAD but not listed in the diff (for example a rename away, where `git diff --name-only` lists only the target); and, for paths that are in the diff, `wrong_qualification_head` and `missing_disposition` are untouched. An unknown pin still raises (fail closed).
- New test file `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py`, written before the code change (red run recorded under `.tmp/proof/pdc-x-f56/`), using throwaway git repositories (no real history): positive net-zero deleted path; negative: present at HEAD but not in diff, deleted-after-pin helper result, renamed-away path, wrong head, missing disposition, unknown pin.

## Write set

Exactly: `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py` (the loosening), `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py` (new), plus the governance documents of this batch (`governance/tickets/T-IGX-040-C.md`, this plan, `docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DECISION-PDC-F56.md`).

Not edited: the frozen PRE-ENT list `_pre_ent_l3_prod_delta.py` (data), `test_roadmap_replay_x_closure_counter_integrity.py` and every other existing test (byte-identical), any product path (`intergrax/`, `agents/`, `applications/`), `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`, the IIAC activation record, the EBH-7 pins and allowlist.

## Gate-semantics note and signing

The change narrows a qualification check (a reported class becomes accepted for one precisely defined case). It is a gate-semantics change decided by the Operator in `DECISION-PDC-F56.md`. **Signing of this plan is not covered by any delegation** (`DELEGATION-PDC-*`) and returns to the Operator via the orchestrator. The implementer never writes an approval or delegation line and never merges.

## Ordering and constraints

1. Local commits only until `development` is unfrozen: the MCP-RC session merges its joint IIAC rebind PR and reports the merge SHA; only then is this branch rebased onto that tip, re-verified, pushed and opened as a PR.
2. The change lies under `tests/`, which is not on the allowlist of the EBH-7 executable-identity gate: the gate goes red until the campaign's final EBH-7 re-pin (accepted by the orchestrator; `DELEGATION-PDC-WAVE2.md` decision 6 covers the re-pin). It touches no path that the IIAC `MATERIAL_GOVERNANCE_PATHS` lists; the implementer confirms this before the PR.
3. No new registry, scheduler, engine, trust, policy or audit authority is introduced; no gate beyond the one branch is changed.

## Protocol

1. One `uv run pytest` per verification step, several paths in one command; log under `.tmp/proof/pdc-x-f56/`.
2. Verification: `uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_b6_r2_reconciliation.py tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py -q`; `uv run ruff check` and `ruff format --check` and `uv run pyright` on the two touched Python files.
3. Commit with exactly one `Ref: governance/tickets/T-IGX-040-C.md` trailer and no AI attribution.

## Acceptance

- `test_b6_03_scoped_disposition_valid_at_head` passes at `development` HEAD; all pre-existing `roadmap_replay_x` tests pass byte-identical.
- The new tests pass, including the negative cases; before the loosening the positive test is red.
- The frozen PRE-ENT list and every path in "Not edited" are byte-for-byte unchanged (`git diff --stat` lists only the write set).

## Out of scope

- Editing the frozen PRE-ENT list data; any broader loosening (for example accepting paths that exist at HEAD without a diff); re-pinning EBH-7; the IIAC audit, activation or certification; any other finding of the audit.
