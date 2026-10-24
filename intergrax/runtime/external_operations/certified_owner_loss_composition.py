# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition binding for certified owner-loss issuers (fail-closed default)."""

from __future__ import annotations

from intergrax.contracts.certified_owner_loss_evidence import (
    CompositionCertifiedOwnerLossIssuerBinding,
)

_DEFAULT_BINDING = CompositionCertifiedOwnerLossIssuerBinding()


def current_certified_owner_loss_issuer_binding() -> CompositionCertifiedOwnerLossIssuerBinding:
    """Return deployment issuer allowlist; empty until operator composition wires issuers."""
    return _DEFAULT_BINDING
