# Semgrep (semgrep)

Category: `security_scanner`

## Legacy facade

- `create_semgrep_security_scanner()` remains backward-compatible.

## Contract-based integration

- `SemgrepSecurityScannerIntegration` derives from the category-specific contract.
- Factory: `create_semgrep_security_scanner_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
