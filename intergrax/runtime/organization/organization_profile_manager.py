# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.runtime.organization.organization_profile import OrganizationProfile
from intergrax.runtime.organization.organization_profile_store import OrganizationProfileStore

class OrganizationProfileManager:
    """
    High-level facade for working with organization profiles.

    Responsibilities:
      - provide convenient methods to:
          * load an OrganizationProfile for a given organization_id,
          * persist profile changes,
          * resolve organization-level system instructions string
            for use in the runtime;
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
    # System instructions management
    # ---------------------------------------------------------------------

    async def get_system_instructions_for_organization(
        self,
        organization_id: str,
    ) -> str:
        """
        Return a compact system-level instruction string for the given organization.

        Behavior:
          - loads the organization's profile from the store,
          - uses the profile's `system_instructions` if set,
          - otherwise builds a deterministic fallback based on identity,
            preferences and high-level summary fields.

        This method does NOT call any LLM. It is pure, deterministic logic
        that always returns a non-empty string.
        """
        profile = await self._store.get_profile(organization_id)
        return self._build_default_system_instructions(profile)

    async def update_system_instructions(
        self,
        organization_id: str,
        instructions: str,
    ) -> OrganizationProfile:
        """
        Update the `system_instructions` field of the organization's profile.

        This method assumes that some higher-level component (e.g. the runtime
        or a batch job) has already decided *what* the new instructions should be,
        possibly by calling an LLM over organization knowledge and other data.

        The manager is responsible only for:
          - loading the profile,
          - normalizing and setting `system_instructions`,
          - marking the profile as modified,
          - persisting it via the store,
          - and clearing the `modified` flag after a successful save.

        Returns the updated OrganizationProfile for convenience.
        """
        profile = await self._store.get_profile(organization_id)

        normalized = instructions.strip()
        profile.system_instructions = normalized or None
        profile.modified = True

        await self._store.save_profile(profile)

        # Lack of exception is interpreted as success.
        profile.modified = False
        return profile

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _build_default_system_instructions(
        self,
        profile: OrganizationProfile,
    ) -> str:
        """
        Deterministic, non-LLM helper that builds system instructions
        from the given organization profile when `system_instructions`
        is not explicitly set.

        This mirrors `_build_default_system_instructions` from UserProfileManager,
        but uses organization-specific fields.
        """
        # If explicit instructions exist, respect them.
        if profile.system_instructions:
            return profile.system_instructions.strip()

        identity = profile.identity
        prefs = profile.preferences

        parts: list[str] = []

        # Identity
        parts.append(
            f"You are working in the context of the organization '{identity.name}'."
        )

        if identity.legal_name and identity.legal_name != identity.name:
            parts.append(
                f"The legal name of the organization is: {identity.legal_name}."
            )

        if identity.industry:
            parts.append(
                f"The organization operates in the '{identity.industry}' industry."
            )

        # High-level domain description and knowledge
        if profile.domain_summary:
            parts.append(
                "High-level description of the organization:\n"
                f"{profile.domain_summary}"
            )

        if profile.knowledge_summary:
            parts.append(
                "Summary of key organizational knowledge, principles "
                "and domain rules:\n"
                f"{profile.knowledge_summary}"
            )

        # Default behaviour
        parts.append(
            "When generating answers for this organization, use the following "
            "default behaviour unless the user explicitly asks otherwise:"
        )
        parts.append(
            f"- Default language: {prefs.default_language}\n"
            f"- Default output format: {prefs.default_output_format}\n"
            f"- Tone of voice: {prefs.tone_of_voice}"
        )

        # Capabilities
        capability_lines: list[str] = []
        capability_lines.append(
            f"- Web search is {'allowed' if prefs.allow_web_search else 'NOT allowed'} "
            "by default."
        )
        capability_lines.append(
            f"- Tools are {'allowed' if prefs.allow_tools else 'NOT allowed'} "
            "by default."
        )
        parts.append("Capabilities and constraints:\n" + "\n".join(capability_lines))

        # Sensitive topics
        if prefs.sensitive_topics:
            parts.append(
                "Treat the following topics as sensitive or restricted; "
                "handle them carefully and prefer concise, non-revealing answers "
                "unless explicitly authorized:\n"
                + ", ".join(prefs.sensitive_topics)
            )

        # Knowledge sources (traceability only, may be useful for the model)
        if profile.knowledge_sources:
            parts.append(
                "Main knowledge sources used to construct this organizational profile:\n"
                + ", ".join(profile.knowledge_sources)
            )

        summary = "\n\n".join(parts).strip()

        if not summary:
            # Extreme fallback – should not normally happen, but we want
            # to guarantee a non-empty instruction string.
            return (
                "You are working in the context of an organization. "
                "Use a professional, concise tone and respect any "
                "organization-specific constraints if they are provided."
            )

        return summary
