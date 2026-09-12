# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed failures for the Runtime Intelligence contract plane (W6-B)."""

from __future__ import annotations


class RuntimeIntelligenceError(Exception):
    """Base error for runtime intelligence contracts — never execution authority."""


class AnalyzerExecutionError(RuntimeIntelligenceError):
    """Analyzer implementation failed during analysis — containment at orchestration boundary."""


class InvalidIntelligenceContextError(RuntimeIntelligenceError):
    """Context snapshot failed validation before analysis."""


__all__ = [
    "AnalyzerExecutionError",
    "InvalidIntelligenceContextError",
    "RuntimeIntelligenceError",
]
