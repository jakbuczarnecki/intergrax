# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.memory.organization_profile_memory import (
    OrganizationProfile,
    OrganizationProfilePromptBundle,
    build_organization_profile_prompt_bundle,
)
from intergrax.memory.organization_profile_store import OrganizationProfileStore


class OrganizationProfileManager:
    """
    High-level facade for working with organization profiles.

    Responsibilities:
      - provide convenient methods to:
          * load or create an OrganizationProfile for a given organization_id,
          * persist profile changes,
          * build a prompt-ready bundle for the LLM/runtime;
      - hide direct interaction with the underlying OrganizationProfileStore.

    It intentionally does NOT:
      - call LLMs directly,
      - perform RAG over organizational knowledge bases,
      - decide *when* the profile should be updated (this is a policy concern
        for higher-level components such as the runtime or application logic).
    """

    def __init__(self, store: OrganizationProfileStore) -> None:
        self._store = store

    # ---------------------------------------------------------------------
    # Core profile APIs
    # ---------------------------------------------------------------------

    async def get_profile(self, organization_id: str) -> OrganizationProfile:
        """
        Load the organization profile for the given organization_id.

        Implementations of OrganizationProfileStore are expected to return
        an initialized profile even if no data exists yet for that organization.
        """
        return await self._store.get_profile(organization_id)

    async def save_profile(self, profile: OrganizationProfile) -> None:
        """
        Persist the given OrganizationProfile aggregate.

        This MUST overwrite any previously stored profile for the same organization.
        """
        await self._store.save_profile(profile)

    async def delete_profile(self, organization_id: str) -> None:
        """
        Remove any stored profile data for the given organization_id.

        This operation is typically used for cleanup or tenant deletion flows.
        """
        await self._store.delete_profile(organization_id)

    # ---------------------------------------------------------------------
    # Prompt bundle API
    # ---------------------------------------------------------------------

    async def get_prompt_bundle(
        self,
        organization_id: str,
        *,
        override_profile: Optional[OrganizationProfile] = None,
        max_summary_length: int = 1200,
    ) -> OrganizationProfilePromptBundle:
        """
        Build a prompt-ready bundle for the given organization.

        Behaviour:
          - if `override_profile` is provided, it is used directly;
          - otherwise the profile is loaded from the store for `organization_id`.

        The resulting OrganizationProfilePromptBundle is suitable to be injected
        into system instructions or used by the runtime to configure
        organization-level constraints, guidelines and defaults.
        """
        profile: OrganizationProfile
        if override_profile is not None:
            profile = override_profile
        else:
            profile = await self._store.get_profile(organization_id)

        return build_organization_profile_prompt_bundle(
            profile=profile,
            max_summary_length=max_summary_length,
        )
