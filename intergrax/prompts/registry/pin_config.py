# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptPinConfig:
    """
    Mapping prompt_id -> pinned version and optional content digests for resume checks.
    """
    pins: dict[str, int]
    content_digests: dict[str, str] = field(default_factory=dict)

    def get(self, prompt_id: str) -> int | None:
        return self.pins.get(prompt_id)

    def get_content_digest(self, prompt_id: str) -> str | None:
        return self.content_digests.get(prompt_id)
