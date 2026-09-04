# Proof Data Package (`intergrax.proof_data`)

Reusable distribution primitives for large external proof assets.

See [architecture documentation](../../../docs/project/capabilities/architecture/PROOF_DATA_PACKAGE_DISTRIBUTION.md).

## Public API

- `ProofDataPackageDescriptor` / `load_proof_data_package_descriptor`
- `DataPackageTransportPort` — `HttpDataPackageTransport`, `LocalFileDataPackageTransport`
- `DataPackageCache` — SHA256-addressed object cache
- `DataPackageInstaller` — orchestrated install with typed `DataPackageInstallReport`

## Example

```python
from pathlib import Path
from intergrax.proof_data import (
    DataPackageCache,
    DataPackageInstaller,
    DataPackageInstallRequest,
    HttpDataPackageTransport,
    load_proof_data_package_descriptor,
)

descriptor = load_proof_data_package_descriptor(Path("package.json"))
report = DataPackageInstaller().install(
    DataPackageInstallRequest(
        descriptor=descriptor,
        install_root=Path("installed"),
        cache=DataPackageCache(Path("~/.cache/intergrax/proof-data")),
        transport=HttpDataPackageTransport(),
        base_uri="https://example.invalid/vpi/v1/",
    )
)
```
