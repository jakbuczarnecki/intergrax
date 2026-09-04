# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Provider execution tuning — separate from semantic embedding identity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingProviderExecutionConfig:
    """Runtime execution settings for embedding providers (not artifact identity)."""

    device: str | None = None
    batch_size: int | None = None
    max_length: int | None = None

    def hf_init_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if self.device is not None:
            kwargs["device"] = self.device
        if self.batch_size is not None:
            kwargs["batch_size"] = self.batch_size
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        return kwargs
