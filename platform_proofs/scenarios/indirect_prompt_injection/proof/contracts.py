"""Proof-owned evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProofVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EXERCISED = "NOT_EXERCISED"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    verdict: ProofVerdict
    checks: tuple[str, ...]
    failures: tuple[str, ...]
