# Okta (okta)

Category: `identity_provider`

## Single public entrypoint

- **`OktaIdentityProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OktaIdentityProviderIntegration`.
- Contract factory: `create_okta_identity_provider_integration()`.
