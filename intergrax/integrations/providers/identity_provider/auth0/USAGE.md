# Auth0 (auth0)

Category: `identity_provider`

## Single public entrypoint

- **`Auth0IdentityProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `Auth0IdentityProviderIntegration`.
- Contract factory: `create_auth0_identity_provider_integration()`.
