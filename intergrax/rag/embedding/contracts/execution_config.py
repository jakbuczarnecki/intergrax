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
