# Gitlab Ci (gitlab_ci)

Category: `ci_cd`

## Single public entrypoint

- **`GitlabCiCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GitlabCiCiCdIntegration`.
- Contract factory: `create_gitlab_ci_ci_cd_integration()`.
