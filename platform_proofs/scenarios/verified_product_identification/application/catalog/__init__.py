"""Catalog application helpers for provider-neutral candidate handling."""

from platform_proofs.scenarios.verified_product_identification.application.catalog.candidate_handoff import (
    collect_channel_candidates,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
    flatten_lexical_text,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.source_resolution import (
    SourceTruthResolutionError,
    resolve_source_record,
)

__all__ = (
    "SourceTruthResolutionError",
    "build_source_record_ref",
    "collect_channel_candidates",
    "derive_search_representation",
    "flatten_lexical_text",
    "resolve_source_record",
)
