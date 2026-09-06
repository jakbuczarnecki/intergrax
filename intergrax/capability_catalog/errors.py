# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated capability catalog read-model errors (CAPABILITY-CATALOG-1 Stage 2)."""

from __future__ import annotations


class CapabilityCatalogError(ValueError):
    """Base error for federated capability catalog read model."""


class CapabilityCatalogConfigurationError(CapabilityCatalogError):
    """Invalid federation composition or source configuration."""


class CapabilityCatalogIdentityConflict(CapabilityCatalogError):
    """Conflicting entries for the same source-qualified discovery identity."""


class CapabilityCatalogSourceFailure(CapabilityCatalogError):
    """A catalog source failed during read — federation aborts fail-closed."""


class CapabilityCatalogDiscoveryError(CapabilityCatalogError):
    """Invalid discovery query inputs or missing enterprise availability evidence."""


class CapabilityRankingError(CapabilityCatalogError):
    """Ranker contract violation — output integrity or invalid ranking metadata."""


class CapabilityGovernanceError(CapabilityCatalogError):
    """Governance evaluator contract violation — partition or elevation failure."""
