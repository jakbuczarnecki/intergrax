# © Artur Czarnecki. All rights reserved.

"""Example documentation-host preference policy — extension proof for source selection."""

from __future__ import annotations

from dataclasses import dataclass

from web_search_qualifier.source_selection.contracts import (
    SourceSelectionContext,
    SourceSelectionOutcome,
    SourceSelectionPolicyDecision,
    SourceSelectionPolicyDescriptor,
    SourceSelectionPolicyId,
)
from web_search_qualifier.source_selection.url_normalization import normalize_url_identity
from web_search_qualifier.web_search import WebSearchCandidate

_POLICY_ID = SourceSelectionPolicyId("source.example_docs_preference")
_PREFERRED_HOST = "docs.example.com"


def _host_matches(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(normalize_url_identity(url))
    return (parsed.hostname or "").lower() == _PREFERRED_HOST


@dataclass(frozen=True, slots=True)
class ExampleDocsSourceSelectionPolicy:
    @property
    def descriptor(self) -> SourceSelectionPolicyDescriptor:
        return SourceSelectionPolicyDescriptor(
            policy_id=_POLICY_ID,
            display_name="Example docs host preference",
        )

    def evaluate(self, context: SourceSelectionContext) -> SourceSelectionPolicyDecision:
        preferred = [
            candidate for candidate in context.candidates if _host_matches(candidate.url)
        ]
        if not preferred:
            return SourceSelectionPolicyDecision(
                outcome=SourceSelectionOutcome.ABSTAIN,
                reason_code="no_preferred_host_candidate",
            )
        selected = min(preferred, key=lambda candidate: candidate.rank)
        return SourceSelectionPolicyDecision(
            outcome=SourceSelectionOutcome.SELECT,
            selected_url=selected.url,
            reason_code="preferred_docs_host",
        )


__all__ = ["ExampleDocsSourceSelectionPolicy"]
