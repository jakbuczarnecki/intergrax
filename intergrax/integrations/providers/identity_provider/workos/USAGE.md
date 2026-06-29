# Workos (workos)

Category: `identity_provider`

## Single public entrypoint

- **`WorkosIdentityProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `WorkosIdentityProviderIntegration`.
- Contract factory: `create_workos_identity_provider_integration()`.
