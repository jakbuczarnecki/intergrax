# Circleci (circleci)

Category: `ci_cd`

## Single public entrypoint

- **`CircleciCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CircleciCiCdIntegration`.
- Contract factory: `create_circleci_ci_cd_integration()`.
