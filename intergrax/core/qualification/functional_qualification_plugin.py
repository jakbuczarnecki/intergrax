# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Plugin contract and descriptor for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.core.qualification.functional_qualification_identity import FunctionalQualificationPluginId
from intergrax.core.qualification.functional_qualification_result import QualificationPluginResult


@dataclass(frozen=True, slots=True)
class QualificationPluginDescriptor:
    plugin_id: FunctionalQualificationPluginId
    domain: str
    version: str
    display_name: str
    contract_version: str
    qualification_level: str
    required_capabilities: tuple[str, ...] = ()


class FunctionalQualificationPlugin(Protocol):
    @property
    def descriptor(self) -> QualificationPluginDescriptor: ...

    def execute(self) -> QualificationPluginResult: ...


__all__ = [
    "FunctionalQualificationPlugin",
    "QualificationPluginDescriptor",
]
