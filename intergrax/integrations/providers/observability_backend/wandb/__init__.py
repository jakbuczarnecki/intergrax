# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_wandb_observability_backend", "register_wandb_integration"]

def __getattr__(name: str):
    if name == "register_wandb_integration":
        from intergrax.integrations.providers.observability_backend.wandb.register import register_wandb_integration
        return register_wandb_integration
    if name == "create_wandb_observability_backend":
        from intergrax.integrations.providers.observability_backend.wandb.bundle import create_wandb_observability_backend
        return create_wandb_observability_backend
    raise AttributeError(name)
