# © Artur Czarnecki. All rights reserved.

"""Compose canonical Memory Control Plane for Tier-3 hosts (MEM-ENT-3)."""

from __future__ import annotations

from intergrax.memory.contracts.memory_control import (
    EpisodicMemoryCapability,
    MemoryControlPlane,
    TaskMemoryCapability,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.contracts.memory_observability import MemoryObservabilitySink
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
)
from intergrax.applications._shared.memory_observability_wiring import (
    resolve_memory_diagnostic_emitter,
)
from intergrax.applications._shared.memory_security_governance_wiring import (
    resolve_memory_security_governance_service,
)

__all__ = ["build_default_memory_control_plane"]


def build_default_memory_control_plane(
    *,
    user_profile_manager: UserProfileManager | None = None,
    task_memory: TaskMemoryCapability | None = None,
    episodic: EpisodicMemoryCapability | None = None,
    security_governance: MemorySecurityGovernanceService | None = None,
    memory_observability_sink: MemoryObservabilitySink | None = None,
    memory_diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
) -> MemoryControlPlane:
    emitter = resolve_memory_diagnostic_emitter(
        sink=memory_observability_sink,
        emitter=memory_diagnostic_emitter,
    )
    governance = resolve_memory_security_governance_service(
        security_governance=security_governance,
        memory_diagnostic_emitter=emitter,
    )
    user_capability = (
        UserProfileManagerMemoryCapability(_manager=user_profile_manager)
        if user_profile_manager is not None
        else None
    )
    return DefaultMemoryControlPlane(
        user_profile=user_capability,
        task_memory=task_memory,
        episodic=episodic,
        security_governance=governance,
        diagnostic_emitter=emitter,
    )
