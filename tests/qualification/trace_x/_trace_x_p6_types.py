# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P6 restart/resume and terminal causality qualification types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RestartResumePathClass(str, Enum):
    A_CANONICAL_RESUME = "A"
    B_CANONICAL_RETRY = "B"
    C_NEW_EXECUTION_AFTER_FAILURE = "C"
    D_SANCTIONED_NON_RESUMABLE = "D"
    E_TEST_LAB_REFERENCE = "E"
    F_PRODUCTION_BYPASS = "F"
    G_UNCLEAR = "G"


class TerminalProducerRole(str, Enum):
    CANONICAL_TERMINAL_TRUTH = "canonical_truth"
    CANONICAL_TERMINAL_DELEGATE = "canonical_delegate"
    RUNTIME_EVENT_EVIDENCE = "runtime_event_evidence"
    DIAGNOSTIC_PROJECTION = "diagnostic_projection"
    OBSERVABILITY_PROJECTION = "observability_projection"
    COMPATIBILITY_ADAPTER = "compatibility_adapter"
    FORBIDDEN_BYPASS = "forbidden_bypass"
    UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class RegisteredRestartResumePath:
    path: str
    surface_id: str
    classification: RestartResumePathClass
    summary: str


@dataclass(frozen=True, slots=True)
class RegisteredTerminalProducer:
    path: str
    surface_id: str
    role: TerminalProducerRole
    summary: str


@dataclass(frozen=True, slots=True)
class RestartResumeParityResult:
    unknown: frozenset[tuple[str, str]]
    orphan: frozenset[tuple[str, str]]
    production_bypass: frozenset[tuple[str, str]]
    unclassified: frozenset[tuple[str, str]]
    ok: bool


@dataclass(frozen=True, slots=True)
class TerminalProducerParityResult:
    unknown: frozenset[tuple[str, str]]
    orphan: frozenset[tuple[str, str]]
    forbidden_bypass: frozenset[tuple[str, str]]
    unclassified: frozenset[tuple[str, str]]
    ok: bool


@dataclass(frozen=True, slots=True)
class AdversarialBundleRow:
    bundle_id: str
    scenario: str
    test_module: str
    test_id: str
    status: str
