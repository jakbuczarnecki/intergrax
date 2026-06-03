# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register ms365_graph in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.collaboration_suite.ms365_graph.bundle import create_ms365_graph_collaboration_suite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_ms365_graph_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_ms365_graph_collaboration_suite, override=override)
