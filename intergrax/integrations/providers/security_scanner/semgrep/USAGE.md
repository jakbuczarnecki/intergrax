# Semgrep (semgrep)

Category: `security_scanner`

## Single public entrypoint

- **`SemgrepSecurityScannerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SemgrepSecurityScannerIntegration`.
- Contract factory: `create_semgrep_security_scanner_integration()`.
