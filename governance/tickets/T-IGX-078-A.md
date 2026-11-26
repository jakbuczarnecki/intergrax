# T-IGX-078-A - AF-CLOSE K0.3 ladder amendment and bookkeeping repair

## Scope

Docs and ticket bookkeeping only. Record the 2026-11-26 AF-CLOSE decisions D-1..D-4 and MEP-AF2 Appendix V6 / AD-AF-DEVTRAIN. Record, as relayed by the orchestrator session, the Operator signature of AD-AF-DEVTRAIN and of the D-3 merge-only override, and OD-AF-23. Repair the SHA ladder, the node ledger, the WP-12 DAG, the retired train documents and the `SOURCES.sha256` digests. Assign the AF-CLOSE owner to T-IGX-053-I, T-IGX-065-A and T-IGX-073-E. No gate is set by this ticket.

## Plan digest

Plan digest: 31767bf783e9a40c819c2a9367ee4b708b6f645ffac590ea209faf3acb72bc1e

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-078-A.md`](plans/PLAN_T-IGX-078-A.md).

## Test contract

Run one command at a time.

- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` (also with `--self-test`)
- `uv run python scripts/governance/verify_rw_plan_phase_check3.py`
- `sha256sum -c SOURCES.sha256` in `docs/project/maintainers/qualification/arch_freeze_program/`
- `uv run python -m scripts.governance.igx_gov ticket --profile smoke`

## Out of scope

Product trees, `scripts/`, `tests/`, `.github/`, `governance/agent-control-plane/role_registry.yaml` (proposal only), roadmap, FRZ checklist, signing anything or writing an `Approved:` line, any gate record, the OD-AF-16a delegation file (K4).

## Approved

_(pending: Operator or a commissioned delegate; the implementing session does not write this line)_

## Lifecycle state

plan_pending
