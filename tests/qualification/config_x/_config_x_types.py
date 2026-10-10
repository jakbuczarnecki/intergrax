# © Artur Czarnecki. All rights reserved.

"""CONFIG-X classification model (A–L)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConfigClassification(StrEnum):
    A_CANONICAL_CONFIGURATION_CONTRACT = "A"
    B_CANONICAL_COMPOSITION_OWNER = "B"
    C_CANONICAL_PROVIDER_IMPLEMENTATION = "C"
    D_SANCTIONED_EXPLICIT_DEFAULT = "D"
    E_PROTOCOL_FORMAT_CONSTANT = "E"
    F_ENVIRONMENT_DEPLOYMENT_CONSTANT = "F"
    G_COMPATIBILITY_ADAPTER = "G"
    H_REFERENCE_LAB_TEST_ONLY = "H"
    I_HARD_CODED_PRODUCTION_SELECTION = "I"
    J_SILENT_FALLBACK = "J"
    K_DUPLICATE_CONFIGURATION_AUTHORITY = "K"
    L_UNCLEAR = "L"


BLOCKER_CLASSIFICATIONS: frozenset[ConfigClassification] = frozenset(
    {
        ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION,
        ConfigClassification.J_SILENT_FALLBACK,
        ConfigClassification.K_DUPLICATE_CONFIGURATION_AUTHORITY,
        ConfigClassification.L_UNCLEAR,
    },
)


@dataclass(frozen=True, slots=True)
class ConfigBlockerRecord:
    blocker_id: str
    classification: ConfigClassification
    paths: tuple[str, ...]
    summary: str
    child_stage: str
    remediation_lineage: str
