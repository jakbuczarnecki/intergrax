# © Artur Czarnecki. All rights reserved.

"""Neutral capability graph catalog entries for runtime edge building."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ApplicationCapabilityCatalogEntry(BaseModel):
    """One application host and its mounted agent contract ids."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    agent_contract_ids: list[str] = Field(default_factory=list)
