# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register wandb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.wandb.bundle import create_wandb_observability_backend
from intergrax.integrations.providers.observability_backend.wandb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_wandb_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_wandb_observability_backend, override=override)
