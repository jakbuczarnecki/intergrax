# T-IGX-041-A — PDC-X batch B-P7-D1: repair of the IA-D1 audit findings for domain D1 (AGENT-LIFECYCLE), loop 1

## Scope

Repair ticket for the independent audit `PDC_P7_D1_AUDIT_REPORT.md` (IA-D1, audited SHA `c08efe46a74a20595468d8d8bf06eab2dca5f573`, verdict PARTIAL), under Operator option B of `DELEGATION-PDC-P7-REPAIR.md`. In scope: F-D1-02 (activation and rollback re-check revocation of the target packages), F-D1-03 authorization part (principal, tenant scope and policy on `respond_to_emergency_revocation`, no caller-supplied revocation snapshot), F-D1-05 (application-ownership guard on install into an occupied slot and on upgrade), F-D1-06 (no leftover `VERIFIED` candidate after a failed upgrade; optional inert event sink so failure-path transitions have an event), F-D1-11 (all 18 red D1 suites root-caused and repaired as test hygiene, classification or one conditional product fix), and by-product test pins for F-D1-12 (mutants M2, M7, M10) and F-D1-14. Excluded with reason and reopen condition (plan section "Excluded findings"): F-D1-01 (attestation never verified on install/upgrade; the trust record has no signature material and no production composition supplies a verifier), F-D1-03 trigger, F-D1-04 (Operator decision), F-D1-07, F-D1-08 (no production caller, F-04 family), the `deploy_activate_pipeline` activation path of F-D1-02, and the LOW/INFO findings. Product paths: yes (`intergrax/agent_distribution/admin_service.py`; one conditional hotspot file `intergrax/applications/_shared/durable_registry_projection_input_store.py`, named exactly). This ticket is not a certification verdict and not a closure of D1; the product-path changes keep the IIAC activation stale (ACT-N11) until the P8-04 rebind and the EBH-7 identity gate red until the final re-pin. Plan: `PLAN_T-IGX-041-A.md`.

## Plan digest

Plan digest: 49f00deee2f8448c878b56233ac8ae8d8c1b8a732169026c21d5a546680ce063

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-A.md`](plans/PLAN_T-IGX-041-A.md).

## Test contract

```text
uv run pytest tests/qualification/pdc_x/d1_repair tests/qualification/pdc_x/d1 tests/unit/agents/authoring tests/unit/agent_distribution tests/integration/agent_distribution tests/unit/runtime/registry tests/unit/skills tests/qualification/pdc_x/xd/test_xd_04_agent_host.py -q
uv run ruff check intergrax/agent_distribution/admin_service.py intergrax/applications/_shared/durable_registry_projection_input_store.py tests/qualification/pdc_x/d1_repair tests/unit/agents/authoring/conftest.py
uv run pyright intergrax/agent_distribution/admin_service.py tests/qualification/pdc_x/d1_repair
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- All excluded findings listed in the plan; the fail-closed rule for a revoked revision without a trusted rollback target (F-D1-04, Operator decision); verifier wiring, production composition of the lifecycle verbs, the revocation-feed trigger.
- `scripts/governance/`, `.github/`, `intergrax/contracts/`, every `intergrax/applications/_shared/` file except the one conditional file, `intergrax/agent_distribution/activation.py`, `emergency_revocation_response.py`, `package_trust.py`, `testing_support/`, the TENANT-X carrier re-pin (T-IGX-041-I), the EBH-7 re-pin, the IIAC activation record and rebind, FRZ promotion, OD signatures, certification.

## Lifecycle state

plan_pending
