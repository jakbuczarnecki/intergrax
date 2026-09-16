# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared invocation wiring requirement presets for catalog providers."""

from intergrax.tools.invocation_wiring_requirements import ToolInvocationWiringRequirements

REQUIRE_SHADOW_WORKSPACE = ToolInvocationWiringRequirements(shadow_workspace=True)
REQUIRE_MEMORY_VIEW = ToolInvocationWiringRequirements(memory_view=True)
REQUIRE_TRACE_READER = ToolInvocationWiringRequirements(trace_reader=True)
REQUIRE_RUN_BUDGET = ToolInvocationWiringRequirements(run_budget=True)
REQUIRE_SANDBOX_SESSION = ToolInvocationWiringRequirements(sandbox_session=True)
