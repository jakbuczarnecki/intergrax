# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Legacy/internal Nexus access for HarnessHostRuntime migration (NPSC-1).

Tier-3 authors MUST use ``HarnessHostRuntime.execution``. This module is for
internal platform wiring and NPSC-2/NPSC-3 consumer migration only.
"""

from __future__ import annotations

from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def resolve_harness_host_nexus_loop_legacy(runtime: HarnessHostRuntime) -> NexusLoop:
    """INTERNAL LEGACY: resolve orchestration backend from host runtime composition."""
    return runtime.execution.nexus_loop
