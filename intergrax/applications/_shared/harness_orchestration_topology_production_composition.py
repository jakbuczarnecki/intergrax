# © Artur Czarnecki. All rights reserved.

"""Harness-host strict production orchestration topology composition (GR-10-R13-R2)."""

from intergrax.runtime.execution.orchestration_topology_production_composition import (
    build_strict_production_orchestration_topology_submission_port,
)

build_harness_host_production_orchestration_topology_submission_port = (
    build_strict_production_orchestration_topology_submission_port
)

__all__ = [
    "build_harness_host_production_orchestration_topology_submission_port",
]
