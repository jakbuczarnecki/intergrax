# Modal (modal)

Category: `sandbox_host`

## Single public entrypoint

- **`ModalSandboxHostIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ModalSandboxHostIntegration`.
- Contract factory: `create_modal_sandbox_host_integration()`.
