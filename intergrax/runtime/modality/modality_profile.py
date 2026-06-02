# © Artur Czarnecki. All rights reserved.

"""Modality profile contracts and tool policy intersection (Phase W-ML.6)."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from pydantic import BaseModel, Field


class ModalityPlane(str, Enum):
    GENERATIVE_LLM = "generative_llm"
    MEDIA_INGEST = "media_ingest"
    DEDICATED_INFERENCE = "dedicated_inference"


_MODALITY_TOOL_PREFIXES: dict[ModalityPlane, tuple[str, ...]] = {
    ModalityPlane.MEDIA_INGEST: ("rag.",),
    ModalityPlane.DEDICATED_INFERENCE: ("vision.", "ml.", "speech."),
    ModalityPlane.GENERATIVE_LLM: ("websearch.",),
}


class ModalityProfile(BaseModel):
    profile_id: str
    allowed_planes: set[ModalityPlane] = Field(default_factory=set)
    allowed_tool_ids: tuple[str, ...] = Field(default_factory=tuple)


def filter_tool_ids_by_modality_profile(
    tool_ids: Sequence[str],
    profile: ModalityProfile,
) -> tuple[str, ...]:
    """Return tool IDs allowed by explicit list and modality plane policy."""
    allowed_explicit = set(profile.allowed_tool_ids)
    allowed_prefixes: list[str] = []
    for plane in profile.allowed_planes:
        allowed_prefixes.extend(_MODALITY_TOOL_PREFIXES.get(plane, ()))

    filtered: list[str] = []
    for tool_id in tool_ids:
        if allowed_explicit and tool_id not in allowed_explicit:
            continue
        if allowed_prefixes and not any(tool_id.startswith(prefix) for prefix in allowed_prefixes):
            if tool_id not in allowed_explicit:
                continue
        filtered.append(tool_id)
    return tuple(filtered)


MODALITY_PROFILE_EXTRA_KEY = "modality_profile"


def lab_default_modality_profile() -> ModalityProfile:
    """Harness lab profile enabling ingest + dedicated inference tools."""
    return ModalityProfile(
        profile_id="lab.default",
        allowed_planes={
            ModalityPlane.MEDIA_INGEST,
            ModalityPlane.DEDICATED_INFERENCE,
            ModalityPlane.GENERATIVE_LLM,
        },
        allowed_tool_ids=(),
    )
