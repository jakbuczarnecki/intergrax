# T-IGX-078-A - AF-CLOSE K0.3 ladder amendment and bookkeeping repair

## Scope

Docs and ticket bookkeeping only. Record the 2026-11-26 AF-CLOSE decisions D-1..D-4 and draft MEP-AF2 Appendix V6 / AD-AF-DEVTRAIN (not signed). Repair the SHA ladder, the node ledger, the WP-12 DAG, the retired train documents and the `SOURCES.sha256` digests. Assign the AF-CLOSE owner to T-IGX-053-I, T-IGX-065-A and T-IGX-073-E. No gate is set by this ticket.

## Plan digest

Plan digest: ea2c4ca59a5277293d065324f038fc6129bd8337f0a0f7404424e36beadfdeec

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-078-A.md`](plans/PLAN_T-IGX-078-A.md).

## Test contract

Run one command at a time.

- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py` (also with `--self-test`)
- `uv run python scripts/governance/verify_rw_plan_phase_check3.py`
- `sha256sum -c SOURCES.sha256` in `docs/project/maintainers/qualification/arch_freeze_program/`
- `uv run python -m scripts.governance.igx_gov ticket --profile smoke`

## Out of scope

Product trees, `scripts/`, `tests/`, `.github/`, `governance/agent-control-plane/role_registry.yaml` (proposal only), roadmap, FRZ checklist, any signature, any gate record.

## Approved

_(pending: Operator or a commissioned delegate; the implementing session does not write this line)_

## Lifecycle state

plan_pending
