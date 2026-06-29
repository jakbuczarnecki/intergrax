# Slash Command (slash_command)

Category: `interaction_surface`

## Single public entrypoint

- **`SlashCommandInteractionSurfaceIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SlashCommandInteractionSurfaceIntegration`.
- Contract factory: `create_slash_command_interaction_surface_integration()`.
