# © Artur Czarnecki. All rights reserved.

"""Canonical revision identity for the host-global integration catalog."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

CANONICAL_INTEGRATION_CATALOG_ID: Final = "host-global-integration-catalog"


class CatalogRevision(BaseModel):
    """Monotonic generation plus deterministic digest of canonical catalog logical state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: int = Field(ge=0)
    state_digest: str = Field(min_length=64, max_length=64)

    def as_mutation_revision_token(self) -> str:
        return f"gen:{self.generation}:sha256:{self.state_digest}"

    @classmethod
    def from_mutation_revision_token(cls, token: str) -> CatalogRevision:
        normalized = token.strip()
        prefix = "gen:"
        mid = ":sha256:"
        if not normalized.startswith(prefix) or mid not in normalized:
            raise ValueError("invalid_catalog_revision_token")
        gen_part, digest = normalized[len(prefix) :].split(mid, 1)
        generation = int(gen_part)
        if generation < 0:
            raise ValueError("invalid_catalog_revision_generation")
        digest = digest.strip()
        if len(digest) != 64:
            raise ValueError("invalid_catalog_revision_digest")
        return cls(generation=generation, state_digest=digest)
