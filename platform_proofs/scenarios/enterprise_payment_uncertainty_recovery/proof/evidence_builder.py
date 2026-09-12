"""Proof-owned evidence projection — consumes application/runtime artifacts."""

from __future__ import annotations


def build_platform_proof_evidence(
    domain_result: object,
    *,
    source_revision: str,
) -> object:
    """Project runtime evidence into PlatformProofEvidence v3."""
    raise NotImplementedError("Implement evidence projection.")
