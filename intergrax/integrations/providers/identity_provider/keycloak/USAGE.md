# Keycloak (keycloak)

Category: `identity_provider`

## Single public entrypoint

- **`KeycloakIdentityProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `KeycloakIdentityProviderIntegration`.
- Contract factory: `create_keycloak_identity_provider_integration()`.
