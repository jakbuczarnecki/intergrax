# Jenkins (jenkins)

Category: `ci_cd`

## Single public entrypoint

- **`JenkinsCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `JenkinsCiCdIntegration`.
- Contract factory: `create_jenkins_ci_cd_integration()`.
