# TRACE-X-P5-R2-P1 — Typed Contracts & Validation Foundation

**Status:** READY FOR AUDIT  
**START_HEAD:** `94099f9980a1bc15c95e779e426eb5a308bb290d`  
**Scope:** P1 contracts and validators only — no persistence, wiring, or reconstruction.

## Contract inventory

| Module | Symbols |
|--------|---------|
| `intergrax/integrations/contracts/existing_capability_configuration_opportunity.py` | `ConfigurationOpportunityRef`, facts, final opportunity, lookup failures, `ReadPort`, `Provider` SPI, `MutationRiskPolicy` |
| `intergrax/integrations/contracts/execution_integration_configuration.py` | `IntegrationMaterializationKind`, `EffectiveIntegrationIdentity`, `ExecutionIntegrationConfigurationAdoption`, match validator |
| `intergrax/contracts/execution_integration_configuration_provenance.py` | subject, configured slice, mode, provenance DTO, read status, `Reader` Protocol |

Reused without duplication: `ConfiguredCapabilityBinding`, `IntegrationConfigurationPayload`, `ExecutionId` / `validate_execution_id`, `ControlPlaneMutationRisk`, `IntegrationCategory`.

## Dependency graph (after)

```text
intergrax.integrations.contracts.existing_capability_configuration_opportunity
  → intergrax.integrations.contracts.existing_capability_configuration
  → intergrax.integrations.contracts.base

intergrax.integrations.contracts.execution_integration_configuration
  → existing_capability_configuration, base

intergrax.contracts.execution_integration_configuration_provenance
  → execution_identity
  → integrations.contracts.execution_integration_configuration (EffectiveIntegrationIdentity)
  → integrations.contracts.base (IntegrationCategory)
```

## Tenant audit

**Verdict:** PASS — contract validators enforce tenant on opportunity, adoption match (`expected_tenant_id`), and provenance record validation.

## Applicable FRZ

- **FRZ-TRC-11:** OPEN (P1 contributes contract evidence only)
- Supporting: FRZ-CTR-*, FRZ-TYP-*, FRZ-PLG-*, FRZ-CFG-*, FRZ-GOV-* (risk ≠ permission), local FRZ-TEN-* invariants

## Tests

- `tests/unit/integrations/test_existing_capability_configuration_opportunity_contracts.py`
- `tests/unit/integrations/test_execution_integration_configuration_contracts.py`
- `tests/unit/contracts/test_execution_integration_configuration_provenance.py`
- `tests/qualification/trace_x/test_trace_x_p5_r2_p1_contract_gates.py`

Baseline regression: P0+R1 gates (`134 passed` at START_HEAD lineage), INT-CONFIG qualification, full P1 bundle (`242 passed` with P1 tests included).

## Enterprise matrix (summary)

| Area | Grade |
|------|-------|
| Contracts over implementations | PASS |
| Semantic / opportunity ownership | PASS (Integrations-owned opportunity; plugin facts only) |
| Risk vs Governance | PASS (policy returns `ControlPlaneMutationRisk` only) |
| Configured vs effective | PASS |
| Neutral provenance layer boundaries | PASS |
| Strong typing | PASS (pyright 0 errors on contract modules) |
| Production wiring | PASS (none) |
| Persistence | N/A — P2 |
