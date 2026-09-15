# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Public re-export surface for user-profile memory domain models."""

from intergrax.memory.contracts.memory_models import (
    EnterpriseMemoryRecord,
    MemoryImportance,
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
    UserProfileMemoryEntryNotFoundError,
)

__all__ = [
    "EnterpriseMemoryRecord",
    "MemoryImportance",
    "MemoryKind",
    "UserIdentity",
    "UserPreferences",
    "UserProfile",
    "UserProfileMemoryEntry",
    "UserProfileMemoryEntryNotFoundError",
]
