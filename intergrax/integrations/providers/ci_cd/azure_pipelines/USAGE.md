# Azure Pipelines (azure_pipelines)

Category: `ci_cd`

## Single public entrypoint

- **`AzurePipelinesCiCdIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AzurePipelinesCiCdIntegration`.
- Contract factory: `create_azure_pipelines_ci_cd_integration()`.
