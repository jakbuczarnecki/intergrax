# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical cross-boundary causal evidence (DIAG-1) — re-export from contracts."""

from intergrax.contracts.platform_causal_evidence import (
    PLATFORM_CAUSAL_EVIDENCE_SCHEMA,
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)

__all__ = [
    "CausalRelationKind",
    "MessageBusTaskRef",
    "PLATFORM_CAUSAL_EVIDENCE_SCHEMA",
    "PlatformCausalEvidence",
    "RuntimeExecutionRef",
]
