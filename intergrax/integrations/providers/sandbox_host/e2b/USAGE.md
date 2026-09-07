# E2B (e2b)

Category: `sandbox_host`

## Single public entrypoint

- **`E2bSandboxHostIntegration`** in `integration.py` is the only public provider class.
- Catalog factory: `create_e2b_sandbox_host()` resolves **`E2bSandboxHostBackend`** (not the generic HTTP shim).
- Contract factory: `create_e2b_sandbox_host_integration()`.

## Security-qualified egress (AW-7C P0-3A)

- Adapter: `E2bSandboxHostBackend` (`backend.py`)
- Transport: `E2bSandboxApiClient` / `SdkE2bSandboxApiClient` (`client.py`)
- Optional SDK extra: `uv sync --extra integrations-e2b`
- Credentials: `INTERGRAX_E2B_API_KEY` or `E2B_API_KEY`
- Template: `INTERGRAX_E2B_TEMPLATE_ID` (default `base`)
- V1 scope: exact `https://host:443` only → E2B `network.allowOut` domains + `network.denyOut: ["0.0.0.0/0"]` at create time
- Attestation: provider sandbox info `network.allowOut` (not requested scope)
- Physical qualification: `tests/integration/providers/sandbox_host/e2b/test_e2b_physical_egress_qualification.py`
