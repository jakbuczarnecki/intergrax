# T-IGX-036-Y — PDC-X P6 batch B-P6-H: EBH-7 B6-F10 computed certification (closed-stage family evidence, C4 classification, replay-profile coverage)

## Scope

Repair of the B6-F10 computed ECAP certification that fails at the P6 re-pin (`146e7acf1`): executable evidence for the two closed-stage families `FRZ-AUD` and `FRZ-EVOL`, directory classification with a stated reason for the 491 unclassified changed paths (unknown stays a blocker), and replay-profile coverage of the contract regression and of the new evidence files. Six ebh_7 tests are red through this one cause. Decision 3 of `DELEGATION-PDC-WAVE2.md`. Product code: no. Plan: `PLAN_T-IGX-036-Y.md`. The real gate and the six tests turn green only at the final re-pin (executable-identity gate; wave-2 decision 6).

## Plan digest

Plan digest: 0c0a54b73c534fd9276e2d7afbb5cb4213c16caadb2d3f51674b552f4dd2610c

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-Y.md`](plans/PLAN_T-IGX-036-Y.md).

## Approved

_Pending: delegate signing per `DELEGATION-PDC-WAVE2.md`, decision 3. The implementer never writes this line._

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_certification_adversarial.py -q
uv run pytest tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py tests/qualification/ent_audit_x/unit/test_frz_coverage_matrix_complete.py tests/qualification/ent_audit_x/unit/test_frz_aud_03_unmapped_neg.py tests/qualification/ent_audit_x/unit/test_frz_aud_w402_neg.py tests/qualification/evol_x/frz -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product code under `intergrax/`, `agents/`, `applications/`; any change of the unknown default or of the N/A rules; edits of signed EBH-7 records.
- Rendering of the F10 matrix and the ECAP report block, the pin, the replay record and the SHA ladders (final re-pin); IIAC activation (PDC-P7-00); FRZ changes; OD signatures and certification.

## Lifecycle state

plan_pending
