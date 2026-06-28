# Salesforce (salesforce)

Category: `crm`

## Legacy facade

- `create_salesforce_crm()` remains backward-compatible.

## Contract-based integration

- `SalesforceCrmIntegration` derives from the category-specific contract.
- Factory: `create_salesforce_crm_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
