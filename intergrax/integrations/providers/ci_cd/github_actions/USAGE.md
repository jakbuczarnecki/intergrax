# Github Actions (github_actions)

Category: `ci_cd`

## Single public entrypoint

- **`GithubActionsCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GithubActionsCiCdIntegration`.
- Contract factory: `create_github_actions_ci_cd_integration()`.
