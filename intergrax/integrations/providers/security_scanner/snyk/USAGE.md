# Snyk (snyk)

Category: `security_scanner`

## Single public entrypoint

- **`SnykSecurityScannerIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SnykSecurityScannerIntegration`.
- Contract factory: `create_snyk_security_scanner_integration()`.
