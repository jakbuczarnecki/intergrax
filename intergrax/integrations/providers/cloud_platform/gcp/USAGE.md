# GCP (gcp)

Category: `cloud_platform`

## Legacy facade

- `create_gcp_integration()` remains backward-compatible.

## Contract-based integration

- `GcpCloudPlatformIntegration` derives from the category-specific contract.
- Factory: `create_gcp_cloud_platform_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
