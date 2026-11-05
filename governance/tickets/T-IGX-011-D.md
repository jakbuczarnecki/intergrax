# T-IGX-011-D — EVOL-X local B6 attestation automation (custody-separated)

## Scope

Implement local ECAP-P4-aligned B6 attestation **mechanics** (read-only attestor process, custodian signer, replay ledger guards, Windows custody setup docs). Run consolidated validation without re-executing 290 qualification tests. **No** governance `IN_FORCE`, **no** formal B6 PASS, **no** FRZ promotion, **no** EVOL-X CLOSED.

**Artifacts:** [`EVOL_X_LOCAL_B6_AUTOMATION_TERMINAL_REPORT.md`](../../docs/project/maintainers/qualification/evol_x/EVOL_X_LOCAL_B6_AUTOMATION_TERMINAL_REPORT.md) · [`EVOL_X_LOCAL_B6_ACTIVATION_AND_CUSTODY_PACKAGE.md`](../../docs/project/maintainers/qualification/evol_x/EVOL_X_LOCAL_B6_ACTIVATION_AND_CUSTODY_PACKAGE.md)

## Approved

Approved: operator @ 2026-11-05 (EVOL-X local B6 automation campaign)

## Test contract

```powershell
uv run pytest tests/unit/qualification/test_evol_x_b6_attestation_contract.py tests/unit/qualification/test_evol_x_b6_local_custody_guards.py tests/unit/qualification/test_evol_x_frz_evol_canonical_matrix.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Lifecycle state

`quality_gate`

**Commit trailer:** `Ref: governance/tickets/T-IGX-011-D.md`
