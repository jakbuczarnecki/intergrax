# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 typed verdicts and matrix rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class R2ChainVerdict(StrEnum):
    PROVEN_CURRENT_HEAD = "PROVEN_CURRENT_HEAD"
    MISSING = "MISSING"
    INVALID = "INVALID"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class R2PluginRewriteClass(StrEnum):
    LEGAL = "LEGAL"
    ILLEGAL = "ILLEGAL"
    UNKNOWN = "UNKNOWN"


class R2SubstitutionClass(StrEnum):
    LEGAL_NON_TENANT = "LEGAL_NON_TENANT"
    ILLEGAL = "ILLEGAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class R2ChainTransition:
    transition_id: str
    source_symbol: str
    target_symbol: str
    tenant_source: str
    credential_or_provider_key: str
    mismatch_behavior: str
    fail_closed: bool
    verdict: R2ChainVerdict
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class R2MechanismRow:
    mechanism_id: str
    family_label: str
    continuity_verdict: R2ChainVerdict
    consumption_verdict: R2ChainVerdict
    evidence: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class R2MatrixRow:
    row_id: str
    surface_hint: str
    verdict: R2ChainVerdict | R2PluginRewriteClass | R2SubstitutionClass
    evidence: tuple[str, ...]


R2_EXPECTED_BASELINE_HEAD: Final[str] = "220fd6e9c5c59fb7bae6c789ebee0484d8bc7c0e"
