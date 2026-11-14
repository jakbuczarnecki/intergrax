# T-IGX-040-C — PDC-X finding F-56: accept a net-zero scoped path in the `roadmap_replay_x` scoped production-delta check

## Scope

Test-infrastructure repair, by Operator decision `DECISION-PDC-F56.md` (2026-11-14). `test_b6_03_scoped_disposition_valid_at_head` fails on `development` (`83d939095`) with `scoped_path_not_in_prod_diff:intergrax/integrations/providers/embedding_provider/ollama/manifest.py`: the file was added after the DUP-X pin and deleted in T-IGX-036-Z, so its net diff is empty, but the frozen PRE-ENT list still names it. The check `scoped_prod_delta_disposition_violations` in `tests/qualification/roadmap_replay_x/_prod_delta_disposition.py` is loosened minimally: a scoped path that is not in the prod diff is accepted only when it exists neither at HEAD nor at the pin (net-zero add and delete); every other case stays reported. New pinned tests (written first) use throwaway git repositories. The frozen PRE-ENT list data and all existing tests stay byte-identical. Product code: no. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-040-C.md`.

## Plan digest

Plan digest: dea4766d879b36d7f16a71bed8dd54cdb2737078bd32336442e4fde1769e5085

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-C.md`](plans/PLAN_T-IGX-040-C.md).

## Approval

Pending. The plan is `plan_pending`. The change is a gate-semantics decision of the Operator (`DECISION-PDC-F56.md`); signing it is **not** covered by any `DELEGATION-PDC-*` record and returns to the Operator via the orchestrator. The implementer does not write the approval line.

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

plan_pending
