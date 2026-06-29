# Salesforce (salesforce)

Category: `crm`

## Single public entrypoint

- **`SalesforceCrmIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SalesforceCrmIntegration`.
- Contract factory: `create_salesforce_crm_integration()`.
