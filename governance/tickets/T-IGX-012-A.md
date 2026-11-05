# T-IGX-012-A — EVOL-X local B6 end-to-end automation campaign (authority fail-closed)

## Scope

Execute campaign `EVOL-X-LOCAL-B6-END-TO-END-AUTOMATION` @ HEAD `e8de0cb034505ad045c336da435fcaccaec1a49c`: Windows custody provisioning attempt, OS independence verification, attestor/custodian/evaluator flows, consolidated qualification tests, terminal report publication. **No** forged ECAP `IN_FORCE`, **no** production custodian key ceremony under implementer, **no** Protocol 03 PASS, **no** FRZ promotion, **no** EVOL-X formal CLOSED.

**Artifacts:** [`EVOL_X_LOCAL_B6_END_TO_END_EXECUTION_TERMINAL_REPORT.md`](../../docs/project/maintainers/qualification/evol_x/EVOL_X_LOCAL_B6_END_TO_END_EXECUTION_TERMINAL_REPORT.md)

## Approved

Approved: operator @ 2026-11-05 (EVOL-X LOCAL B6 END-TO-END AUTOMATION campaign)

## Test contract

```powershell
uv run pytest tests/unit/qualification/test_evol_x_b6_attestation_contract.py tests/unit/qualification/test_evol_x_b6_local_custody_guards.py tests/unit/qualification/test_evol_x_frz_evol_canonical_matrix.py tests/qualification/evol_x/gates/test_evol_x_p03_integrity.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

ECAP maintainer signature, independent security review signature, Windows user creation without UAC, ENT-AUDIT-X B6, ARCH-FREEZE certification, 290-test replay (pinned TES unchanged).

## Lifecycle state

`quality_gate`

**Commit trailer:** `Ref: governance/tickets/T-IGX-012-A.md`
