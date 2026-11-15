# T-IGX-040-D — PDC-X finding F-35: classify the evidence-plane security audit trail as a COMPAT-X persistence site

## Scope

Test-infrastructure repair for finding F-35 (`OD-MCP-RC-04` section 2). Ten COMPAT-X event-evolution gates (four in `test_compat_x_event_evolution_r3_p0_gates.py`, six in `test_compat_x_event_evolution_r3_r1_gates.py`) are red on `development` because `EvidencePlaneSecurityAuditTrail` (`intergrax/runtime/security/security_audit_trail.py`, added by T-IGX-036-P) appends a canonical `RuntimeEvent` through the injected `EvidencePersistencePort` and constructs it at line 198; the gates' classifiers do not know the site (`canonical production persistence bypass count 1 (expected 0)`). The plan diagnoses it as a legitimate classified site (canonical validating evidence plane, registered typed payload, same pattern as the classified governance evidence persistence), not a bypass and not a product defect. The repair adds two narrow, typed classification rules on the test side (exact module + function + callee; no module-wide trust) and new pinned positive and negative tests written first; no gate is weakened or deleted, the two gate test files stay byte-identical. Product code: no. This ticket is not a certification verdict. Plan: `PLAN_T-IGX-040-D.md`.

## Plan digest

Plan digest: 01717feb557c0cb983859fc539e4e0c6f3d0013157bb31a7e481bdfd4259d584

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-D.md`](plans/PLAN_T-IGX-040-D.md).

## Approval

Pending. The plan is `plan_pending`. Signing is **not** covered by any `DELEGATION-PDC-*` record (the change adds classifications to qualification gates) and returns to the Operator via the orchestrator. The implementer does not write the approval line.

## Test contract

```text
uv run pytest tests/qualification/compat_x/test_compat_x_event_evolution_r3_p0_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_r3_r1_gates.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py tests/qualification/pdc_x/d6/test_d6_audit_trail_durable.py -q
uv run ruff check tests/qualification/compat_x/_compat_x_event_evolution_inventory.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py
uv run pyright tests/qualification/compat_x/_compat_x_event_evolution_inventory.py tests/qualification/compat_x/_compat_x_event_evolution_r3_r1.py tests/qualification/compat_x/test_compat_x_event_evolution_f35_security_audit_trail_site.py
```

## Out of scope

- Any product path (including a runtime marker assertion in the audit trail), the two gate test files, the seven COMPAT-X gates that are red at the pin, `scripts/governance/`, `.github/`, `intergrax/contracts/`, `intergrax/applications/_shared/`.
- Re-pinning EBH-7, the IIAC audit, activation and certification, and every other finding.

## Lifecycle state

plan_pending
