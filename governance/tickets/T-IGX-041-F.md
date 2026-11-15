# T-IGX-041-F — PDC-X batch B-P7-D6: repair of the independent D6 (RUNTIME-SECURITY) audit findings, loop 1

## Scope

Product and test repair after the independent audit IA-D6 (`PDC_P7_D6_AUDIT_REPORT.md`, verdict PARTIAL, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`), under the Operator's option B (`DELEGATION-PDC-P7-REPAIR.md`): real defects in existing code and test hygiene only. In scope: F-D6-01 (the evidence-plane security audit trail normalizes its persistence port with `as_evidence_persistence_port`, so its append goes through the validating port that the F-35 COMPAT-X classification assumes), the relabel part of F-D6-02 (the "immutable, replicated" wording of the multi-region trail is in-process bookkeeping), the binding part of F-D6-05 (plugin provenance requires the attested distribution to equal the entry point's distribution), F-D6-07 (the sandbox governance-deny unit test gets the governance identity it needs; a negative test pins the fail-closed contract), and by-product F-D6-12 (no provider text in the credential-unavailable error chain). Excluded with reason and reopen condition (plan, table "Excluded findings"): tamper evidence of the audit trail, F-D6-04 (production composition, F-04 family), the installed-file digest of F-D6-05, F-D6-06 (ticket T-IGX-041-I), and all other LOW/INFO findings. Product code: yes (`intergrax/runtime/security/`, `intergrax/core/plugins/`, `intergrax/integrations/credentials/`). No hotspot path is written. This ticket is not a certification verdict; IIAC activation stays stale until the P8-04 rebind and the EBH-7 identity gate stays red until the final re-pin. Plan: `PLAN_T-IGX-041-F.md`.

## Plan digest

Plan digest: 65cb0343e67e9193f424d0362e3a26b7e44ac51df6e524f6ae0fae5534a57574

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-F.md`](plans/PLAN_T-IGX-041-F.md).

## Approved

(pending; the plan is signed by a delegate session under `DELEGATION-PDC-P7-REPAIR.md` or by the Operator; the implementer never writes the approval line)

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d6_repair tests/qualification/pdc_x/d6 tests/qualification/pdc_x/xd/test_xd_02_agent_context.py tests/qualification/pdc_x/xd/test_xd_06_background_security.py tests/qualification/pdc_x/xd/test_xd_10_artifacts_security.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py tests/unit/runtime/sandbox tests/integration/runtime/sandbox/reference_substrate tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py tests/unit/integrations/credentials -q -p no:randomly
uv run python scripts/maintenance/check_immutable_security_audit_trail.py
uv run ruff check <touched files>
uv run pyright <touched files>
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

Known reds that may remain and are not repaired here: the two `test_audit_ideal_depth_gate.py` failures (`29_1`, `30_4`), the physical egress tests skipped on Windows, EBH-7 identity, IIAC activation.

## Out of scope

- The excluded findings (plan table) and the TENANT-X carrier re-pin (T-IGX-041-I).
- `intergrax/applications/_shared/` (including `security_audit_trail_wiring.py`), `intergrax/contracts/`, `intergrax/runtime/nexus/`, `testing_support/`, the COMPAT-X inventory and its tests, `scripts/governance/`, `.github/`, FRZ files and P8 originals.
- IIAC activation and rebind, the EBH-7 re-pin, OD signatures, certification and the re-audit.

## Lifecycle state

plan_pending
