# Gitlab Ci (gitlab_ci)

Category: `ci_cd`

## Legacy facade

- `create_gitlab_ci_ci_cd()` remains backward-compatible.

## Contract-based integration

- `GitlabCiCiCdIntegration` derives from the category-specific contract.
- Factory: `create_gitlab_ci_ci_cd_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
