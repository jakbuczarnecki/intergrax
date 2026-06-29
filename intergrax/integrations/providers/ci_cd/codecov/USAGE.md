# Codecov (codecov)

Category: `ci_cd`

## Single public entrypoint

- **`CodecovCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CodecovCiCdIntegration`.
- Contract factory: `create_codecov_ci_cd_integration()`.
