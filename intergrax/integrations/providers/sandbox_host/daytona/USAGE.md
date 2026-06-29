# Daytona (daytona)

Category: `sandbox_host`

## Single public entrypoint

- **`DaytonaSandboxHostIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DaytonaSandboxHostIntegration`.
- Contract factory: `create_daytona_sandbox_host_integration()`.
