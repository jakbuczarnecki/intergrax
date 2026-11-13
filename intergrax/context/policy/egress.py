# © Artur Czarnecki. All rights reserved.

"""Provider-egress sensitivity gate for context fragments (PDC-D2-R014)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragment
from intergrax.contracts.data_classification import DataClassification

PROVIDER_EGRESS_BLOCKED_REASON = "sensitivity.provider_egress_blocked"

_RANK: dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}


def filter_fragments_by_provider_egress(
    fragments: list[ContextFragment],
    *,
    ceiling: DataClassification = DataClassification.INTERNAL,
) -> tuple[list[ContextFragment], list[tuple[ContextFragment, str]]]:
    """Withhold fragments classified above ``ceiling`` from the model-facing context.

    The default ceiling matches ``DataClassification.allows_export`` (PUBLIC and INTERNAL only):
    CONFIDENTIAL and RESTRICTED fragments never reach an external LLM provider, mandatory or not.
    """
    kept: list[ContextFragment] = []
    excluded: list[tuple[ContextFragment, str]] = []
    for fragment in fragments:
        if _RANK[fragment.sensitivity] > _RANK[ceiling]:
            excluded.append((fragment, PROVIDER_EGRESS_BLOCKED_REASON))
            continue
        kept.append(fragment)
    return kept, excluded
