# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition binding for certified owner-loss issuers (fail-closed default)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.autonomous_work.repository import WorkerInstanceRepository
from intergrax.contracts.certified_owner_loss_evidence import (
    CertifiedOwnerLossEvidenceIssuerPort,
    CompositionCertifiedOwnerLossIssuerBinding,
)

_DEFAULT_BINDING = CompositionCertifiedOwnerLossIssuerBinding()


@dataclass(frozen=True, slots=True)
class CertifiedOwnerLossComposition:
    issuer_binding: CompositionCertifiedOwnerLossIssuerBinding
    issuer: CertifiedOwnerLossEvidenceIssuerPort | None = None
    worker_instance_repository: WorkerInstanceRepository | None = None


_COMPOSITION = CertifiedOwnerLossComposition(issuer_binding=_DEFAULT_BINDING)


def current_certified_owner_loss_issuer_binding() -> CompositionCertifiedOwnerLossIssuerBinding:
    """Return deployment issuer allowlist; empty until operator composition wires issuers."""
    return _COMPOSITION.issuer_binding


def require_certified_owner_loss_composition() -> CertifiedOwnerLossComposition:
    return _COMPOSITION


def _replace_certified_owner_loss_composition(
    composition: CertifiedOwnerLossComposition,
) -> None:
    global _COMPOSITION
    _COMPOSITION = composition


def reset_certified_owner_loss_composition_for_tests() -> None:
    """Test-only reset to fail-closed default."""
    _replace_certified_owner_loss_composition(
        CertifiedOwnerLossComposition(issuer_binding=_DEFAULT_BINDING),
    )
