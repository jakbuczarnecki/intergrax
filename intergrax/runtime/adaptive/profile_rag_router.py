# © Artur Czarnecki. All rights reserved.

"""RAG profile overrides from adaptive profile versions (Phase W-ADAPT-3.7)."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.routing.query_router import QueryRouter, RouteTier
from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionRecord

RouteMode = Literal["off", "auto"]

_ARM_DEEP_QUERY_MIN_WORDS: dict[str, int] = {
    "rag_tier_default": 12,
    "rag_tier_deep": 8,
    "llm_route_balanced": 12,
}


def apply_rag_profile_version(
    base_profile: RagProfile,
    version: ProfileVersionRecord | None,
) -> RagProfile:
    """Merge candidate/active RAG profile version payload into a base RagProfile."""
    if version is None or version.artifact_type != ProfileArtifactType.RAG:
        return base_profile

    payload = version.artifact_payload
    updates: dict[str, object] = {}

    selected_arm = payload.get("selected_arm")
    if isinstance(selected_arm, str) and selected_arm in _ARM_DEEP_QUERY_MIN_WORDS:
        updates["deep_query_min_words"] = _ARM_DEEP_QUERY_MIN_WORDS[selected_arm]

    if "deep_query_min_words" in payload:
        raw = payload["deep_query_min_words"]
        if isinstance(raw, int) and raw >= 1:
            updates["deep_query_min_words"] = raw

    route_mode = payload.get("route_mode")
    if route_mode in ("off", "auto"):
        updates["route_mode"] = route_mode

    if not updates:
        return base_profile
    return replace(base_profile, **updates)


class ProfileAwareQueryRouter:
    """
    QueryRouter that prefers a candidate profile version over the active one.

    Candidate wins for shadow runs; active is used when no candidate is supplied.
    """

    def __init__(
        self,
        base_profile: RagProfile,
        *,
        active_version: ProfileVersionRecord | None = None,
        candidate_version: ProfileVersionRecord | None = None,
    ) -> None:
        selected = candidate_version or active_version
        merged = apply_rag_profile_version(base_profile, selected)
        self._router = QueryRouter(merged)
        self._candidate_version_id = candidate_version.version_id if candidate_version else None
        self._active_version_id = active_version.version_id if active_version else None

    @property
    def candidate_profile_version_id(self) -> str | None:
        return self._candidate_version_id

    @property
    def active_profile_version_id(self) -> str | None:
        return self._active_version_id

    def route(self, query_text: str) -> RouteTier:
        return self._router.route(query_text)
