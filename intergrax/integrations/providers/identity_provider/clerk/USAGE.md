# Clerk (clerk)

Category: `identity_provider`

## Single public entrypoint

- **`ClerkIdentityProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ClerkIdentityProviderIntegration`.
- Contract factory: `create_clerk_identity_provider_integration()`.
