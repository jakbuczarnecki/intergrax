# © Artur Czarnecki. All rights reserved.

"""Execution Evidence runtime: compose, attest, verify (host-owned)."""

from intergrax.runtime.execution_evidence.attestor import (
    Ed25519HostAttestor,
    build_deterministic_test_attestor,
)
from intergrax.runtime.execution_evidence.compose import (
    AttestationOutcome,
    attest_after_governed_side_effect,
    compose_execution_boundary_event,
    produce_proof_receipt,
)
from intergrax.runtime.execution_evidence.verify import (
    VerificationKeyResolver,
    verify_proof_receipt,
)

__all__ = [
    "AttestationOutcome",
    "Ed25519HostAttestor",
    "VerificationKeyResolver",
    "attest_after_governed_side_effect",
    "build_deterministic_test_attestor",
    "compose_execution_boundary_event",
    "produce_proof_receipt",
    "verify_proof_receipt",
]
