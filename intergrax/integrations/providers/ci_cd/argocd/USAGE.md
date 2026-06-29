# Argocd (argocd)

Category: `ci_cd`

## Single public entrypoint

- **`ArgocdCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ArgocdCiCdIntegration`.
- Contract factory: `create_argocd_ci_cd_integration()`.
