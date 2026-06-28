# Replicate (replicate)

Category: `ml_inference_host`

## Legacy facade

- `create_replicate_ml_inference_host()` remains backward-compatible.

## Contract-based integration

- `ReplicateMlInferenceHostIntegration` derives from the category-specific contract.
- Factory: `create_replicate_ml_inference_host_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
