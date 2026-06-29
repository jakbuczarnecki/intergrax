# Trivy (trivy)

Category: `security_scanner`

## Single public entrypoint

- **`TrivySecurityScannerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TrivySecurityScannerIntegration`.
- Contract factory: `create_trivy_security_scanner_integration()`.
