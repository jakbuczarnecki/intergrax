"""Typed finalist qualification contracts — model-neutral at evaluation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FinalistRole(str, Enum):
    BASELINE = "BASELINE"
    CHALLENGER = "CHALLENGER"


@dataclass(frozen=True, slots=True)
class FinalistQualificationSelection:
    baseline_candidate_id: str
    challenger_candidate_id: str

    def __post_init__(self) -> None:
        if not self.baseline_candidate_id:
            msg = "baseline_candidate_id must not be empty"
            raise ValueError(msg)
        if not self.challenger_candidate_id:
            msg = "challenger_candidate_id must not be empty"
            raise ValueError(msg)
        if self.baseline_candidate_id == self.challenger_candidate_id:
            msg = "baseline and challenger candidate ids must differ"
            raise ValueError(msg)
