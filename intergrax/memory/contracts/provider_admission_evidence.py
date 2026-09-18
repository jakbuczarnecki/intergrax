# © Artur Czarnecki. All rights reserved.

"""Typed admission evidence context (MEM-FINAL-AUDIT-5A-R3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderDurabilityEvidenceRegistry,
)
from intergrax.memory.contracts.provider_qualification_evidence import (
    MemoryProviderQualificationEvidenceRegistry,
)

__all__ = ["MemoryProviderAdmissionEvidenceContext"]


@dataclass(frozen=True, slots=True)
class MemoryProviderAdmissionEvidenceContext:
    qualification_registry: MemoryProviderQualificationEvidenceRegistry
    durability_registry: MemoryProviderDurabilityEvidenceRegistry
