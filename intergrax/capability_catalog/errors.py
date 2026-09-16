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


class CapabilitySearchError(CapabilityCatalogError):
    """Search strategy contract violation — output integrity or invalid search metadata."""


class CapabilityRecommendationError(CapabilityCatalogError):
    """Recommendation strategy contract violation — output integrity or invalid metadata."""


class CapabilityGovernanceError(CapabilityCatalogError):
    """Governance evaluator contract violation — partition or elevation failure."""


class CapabilityGovernanceEvaluatorUnavailableError(CapabilityCatalogError):
    """Expected operational inability to produce a governance decision.

    Plugins raise this when a dependency is down or evidence cannot be
    obtained for a known, contractual reason — not for programming defects.
    """


# Stable alias for cross-evaluator governance plugin contracts (ME-6-C1).
CapabilityGovernanceExpectedEvaluatorFailure = CapabilityGovernanceEvaluatorUnavailableError
