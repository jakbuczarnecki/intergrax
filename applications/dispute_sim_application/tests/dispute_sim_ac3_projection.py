# © Artur Czarnecki. All rights reserved.

"""Dispute sim test registry projection helper (AC-3)."""

from __future__ import annotations

from dispute_sim_application.host.agent_builders import DISPUTE_SIM_AGENT_BUILDERS
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.manifest import build_dispute_sim_manifest
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection


def build_dispute_sim_test_registry_projection(
    *,
    revision_id: str = "dispute-sim-test-revision",
) -> MaterializedRegistryProjection:
    manifest = build_dispute_sim_manifest()
    env = manifest.environment or build_dispute_sim_environment_profile()
    return build_test_registry_projection(
        manifest,
        env,
        builders=DISPUTE_SIM_AGENT_BUILDERS,
        revision_id=revision_id,
    )
