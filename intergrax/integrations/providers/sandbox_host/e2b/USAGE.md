# E2B (e2b)

Category: `sandbox_host`

## Single public entrypoint

- **`E2bSandboxHostIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `E2bSandboxHostIntegration`.
- Contract factory: `create_e2b_sandbox_host_integration()`.
