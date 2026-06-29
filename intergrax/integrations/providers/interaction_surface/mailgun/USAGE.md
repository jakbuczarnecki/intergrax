# Mailgun (mailgun)

Category: `interaction_surface`

## Single public entrypoint

- **`MailgunInteractionSurfaceIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MailgunInteractionSurfaceIntegration`.
- Contract factory: `create_mailgun_interaction_surface_integration()`.
