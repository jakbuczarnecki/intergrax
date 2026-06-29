# Lab Json (lab_json)

Category: `interaction_surface`

## Single public entrypoint

- **`LabJsonInteractionSurfaceIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LabJsonInteractionSurfaceIntegration`.
- Contract factory: `create_lab_json_interaction_surface_integration()`.
