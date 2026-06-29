# Buildkite (buildkite)

Category: `ci_cd`

## Single public entrypoint

- **`BuildkiteCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BuildkiteCiCdIntegration`.
- Contract factory: `create_buildkite_ci_cd_integration()`.
