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

__all__ = ["build_default_memory_control_plane"]


def build_default_memory_control_plane(
    *,
    user_profile_manager: UserProfileManager | None = None,
    task_memory: TaskMemoryCapability | None = None,
    episodic: EpisodicMemoryCapability | None = None,
) -> MemoryControlPlane:
    user_capability = (
        UserProfileManagerMemoryCapability(_manager=user_profile_manager)
        if user_profile_manager is not None
        else None
    )
    return DefaultMemoryControlPlane(
        user_profile=user_capability,
        task_memory=task_memory,
        episodic=episodic,
    )
