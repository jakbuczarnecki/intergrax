"""Explicit bootstrap failure hierarchy for VPI storage preparation."""

from __future__ import annotations


class VpiBootstrapError(Exception):
    """Base error for VPI bootstrap orchestration."""


class VpiBootstrapConfigurationError(VpiBootstrapError):
    """Invalid or incomplete bootstrap configuration."""


class VpiBootstrapCompatibilityError(VpiBootstrapError):
    """Existing environment identity does not match active configuration."""


class VpiBootstrapProviderError(VpiBootstrapError):
    """Provider readiness or operation failure during bootstrap."""


class VpiBootstrapDataError(VpiBootstrapError):
    """Dataset parsing, derivation, or ingest data failure."""
