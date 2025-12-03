# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol

from intergrax.memory.organization_profile_memory import OrganizationProfile


class OrganizationProfileStore(Protocol):
    """
    Persistent storage interface for organization profiles.

    This store is responsible for:
    - loading and saving the `OrganizationProfile` aggregate,
    - providing sane defaults for new organizations,
    - hiding backend-specific concerns (JSON files, SQL DB, etc.).

    It MUST NOT:
    - implement LLM prompt logic,
    - perform RAG operations,
    - decide how the profile is injected into prompts.
    """

    async def get_profile(self, organization_id: str) -> OrganizationProfile:
        """
        Load organization profile for the given organization_id.

        Implementations SHOULD:
        - return an initialized profile even if no data exists yet
          (e.g. with default identity/preferences),
        - never return None.
        """
        ...

    async def save_profile(self, profile: OrganizationProfile) -> None:
        """
        Persist the given profile aggregate for the associated organization_id.

        This MUST overwrite any previously stored profile for that organization.
        """
        ...

    async def delete_profile(self, organization_id: str) -> None:
        """
        Remove any stored profile data for the given organization_id.

        Implementations MUST tolerate unknown organization_ids without error.
        """
        ...
