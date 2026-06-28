# Codecov (codecov)

Category: `ci_cd`

## Legacy facade

- `create_codecov_ci_cd()` remains backward-compatible.

## Contract-based integration

- `CodecovCiCdIntegration` derives from the category-specific contract.
- Factory: `create_codecov_ci_cd_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
