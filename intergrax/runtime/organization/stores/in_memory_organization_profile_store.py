# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict

from intergrax.runtime.organization.organization_profile import OrganizationIdentity, OrganizationPreferences, OrganizationProfile
from intergrax.runtime.organization.organization_profile_store import OrganizationProfileStore


class InMemoryOrganizationProfileStore(OrganizationProfileStore):
    """
    In-memory implementation of OrganizationProfileStore.

    Use cases:
      - unit tests,
      - local development,
      - experiments and notebooks.

    This implementation does NOT provide durability or cross-process sharing.
    """

    def __init__(self) -> None:
        # organization_id -> OrganizationProfile
        self._profiles: Dict[str, OrganizationProfile] = {}

    async def get_profile(self, organization_id: str) -> OrganizationProfile:
        """
        Return an existing profile or a default one if not present.
        """
        if organization_id in self._profiles:
            return self._profiles[organization_id]

        # Create a default, mostly empty profile for a new organization.
        identity = OrganizationIdentity(organization_id=organization_id, name=organization_id)
        preferences = OrganizationPreferences()
        profile = OrganizationProfile(identity=identity, preferences=preferences)

        # Optionally store the default profile to make subsequent calls cheaper.
        self._profiles[organization_id] = profile
        return profile

    async def save_profile(self, profile: OrganizationProfile) -> None:
        """
        Persist or update the profile in memory.
        """
        self._profiles[profile.identity.organization_id] = profile

    async def delete_profile(self, organization_id: str) -> None:
        """
        Remove a stored profile, if present. Ignore unknown IDs.
        """
        self._profiles.pop(organization_id, None)

    # Optional helper for debugging / tests
    def list_organization_ids(self):
        return list(self._profiles.keys())
