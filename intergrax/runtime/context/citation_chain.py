# © Artur Czarnecki. All rights reserved.

"""Citation chain output→fragment→source (IDEAL-16.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CitationChainLink(BaseModel):
    output_ref: str
    fragment_id: str
    source_id: str
    excerpt: str = ""


class CitationChain(BaseModel):
    links: list[CitationChainLink] = Field(default_factory=list)

    def add(self, *, output_ref: str, fragment_id: str, source_id: str, excerpt: str = "") -> None:
        self.links.append(
            CitationChainLink(
                output_ref=output_ref,
                fragment_id=fragment_id,
                source_id=source_id,
                excerpt=excerpt,
            )
        )
