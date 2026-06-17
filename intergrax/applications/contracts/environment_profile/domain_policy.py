# © Artur Czarnecki. All rights reserved.

"""Typed domain policy fragment container (APP-EVOL-8 · P1-ARCH-01 proposal 5)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainPolicyFragment(BaseModel):
    """One named policy slice merged into ``RuntimePolicyBundle.domain_fragments``."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fragment_id")
    @classmethod
    def _normalize_fragment_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("fragment_id must be non-empty")
        return stripped


class DomainPolicyFragments(BaseModel):
    """Typed wrapper for product-specific policy slices on a Tier-3 host."""

    model_config = ConfigDict(extra="forbid")

    fragments: dict[str, DomainPolicyFragment] = Field(default_factory=dict)

    @classmethod
    def from_runtime_dict(cls, raw: dict[str, Any] | None) -> DomainPolicyFragments:
        if not raw:
            return cls()
        fragments: dict[str, DomainPolicyFragment] = {}
        for key, payload in raw.items():
            fragment_id = str(key).strip()
            if not fragment_id:
                continue
            if isinstance(payload, DomainPolicyFragment):
                fragments[fragment_id] = payload
            elif isinstance(payload, dict):
                fragments[fragment_id] = DomainPolicyFragment(
                    fragment_id=fragment_id,
                    payload=dict(payload),
                )
            else:
                fragments[fragment_id] = DomainPolicyFragment(
                    fragment_id=fragment_id,
                    payload={"value": payload},
                )
        return cls(fragments=fragments)

    def to_runtime_dict(self) -> dict[str, Any]:
        return {key: fragment.payload for key, fragment in self.fragments.items()}

    def merge(self, other: DomainPolicyFragments) -> DomainPolicyFragments:
        merged = dict(self.fragments)
        merged.update(other.fragments)
        return DomainPolicyFragments(fragments=merged)
