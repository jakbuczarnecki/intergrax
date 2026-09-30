# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: ``runtime.execution.application_graph_spec_to_plan``."""

from intergrax.runtime.execution.application_graph_spec_to_plan import (
    application_graph_spec_to_nexus_plan,
    should_seed_plan_from_graph_spec,
)

__all__ = [
    "application_graph_spec_to_nexus_plan",
    "should_seed_plan_from_graph_spec",
]
