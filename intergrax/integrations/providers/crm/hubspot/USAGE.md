# Hubspot (hubspot)

Category: `crm`

## Single public entrypoint

- **`HubspotCrmIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `HubspotCrmIntegration`.
- Contract factory: `create_hubspot_crm_integration()`.
