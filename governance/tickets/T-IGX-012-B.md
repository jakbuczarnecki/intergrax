# T-IGX-012-B — EVOL-X local B6 unattended execution (RunAs bootstrap)

## Scope

Campaign `EVOL-X-LOCAL-B6-UNATTENDED-EXECUTION`: elevated RunAs launcher, unattended bootstrap (custody, principals, ACLs, scheduled-task principal runs), local key ceremony module, terminal report. **No** ECAP `IN_FORCE`, **no** forged maintainer/security signatures, **no** Protocol 03 PASS / FRZ / W7 closure without authority.

**Artifact:** [`EVOL_X_LOCAL_B6_UNATTENDED_EXECUTION_FINAL_REPORT.md`](../../docs/project/maintainers/qualification/evol_x/EVOL_X_LOCAL_B6_UNATTENDED_EXECUTION_FINAL_REPORT.md)

## Approved

Approved: operator @ 2026-11-05 (EVOL-X LOCAL B6 UNATTENDED EXECUTION)

## Test contract

```powershell
uv run pytest tests/unit/qualification/test_evol_x_b6_attestation_contract.py tests/unit/qualification/test_evol_x_b6_local_custody_guards.py tests/unit/qualification/test_evol_x_frz_evol_canonical_matrix.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

290-test replay (TES unchanged), ENT-AUDIT-X, ARCH-FREEZE, organizational OD/ECAP signatures.

## Lifecycle state

`delivered`

## Checkpoint (@ 2026-11-05)

- Pipeline disposition: **`TECHNICAL_PIPELINE_OK_PENDING_AUTHORITY`** (`elevated-result.json` @ `2026-11-05T08:00:08Z`)
- Publication SHA: `42363732d9f450bfe4329d76a96c646d2ab730b2`
- Tests: 35/35 B6 suite; `igx_gov ticket` PASS (PYTHONPATH unset)
- Superseded: launcher EXIT=5 timeout run (~07:35 UTC)

**Commit trailer:** `Ref: governance/tickets/T-IGX-012-B.md`
