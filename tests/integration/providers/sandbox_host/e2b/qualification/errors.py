# © Artur Czarnecki. All rights reserved.

"""Qualification harness errors — causal-proof boundary only."""

from __future__ import annotations


class QualificationError(Exception):
    """Base qualification harness error."""


class QualificationCredentialUnavailable(QualificationError):
    """E2B credentials are not available for physical qualification."""


class QualificationSessionError(QualificationError):
    """Hosted sandbox session could not be admitted or executed."""


class QualificationBaselineError(QualificationError):
    """Control phase baseline connectivity does not support causal proof."""


class QualificationAssertionError(QualificationError):
    """Observed provider behavior does not satisfy qualification expectations."""
