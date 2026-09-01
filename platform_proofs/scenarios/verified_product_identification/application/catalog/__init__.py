"""Catalog application helpers for provider-neutral candidate handling."""

from platform_proofs.scenarios.verified_product_identification.application.catalog.candidate_handoff import (
    collect_channel_candidates,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.source_resolution import (
    SourceTruthResolutionError,
    resolve_source_record,
)

__all__ = (
    "SourceTruthResolutionError",
    "collect_channel_candidates",
    "resolve_source_record",
)
