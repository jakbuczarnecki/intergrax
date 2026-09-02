# © Artur Czarnecki. All rights reserved.

"""Python final-release source selection policy for official python.org downloads."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from web_search_qualifier.source_selection.contracts import (
    SourceSelectionContext,
    SourceSelectionOutcome,
    SourceSelectionPolicyDecision,
    SourceSelectionPolicyDescriptor,
    SourceSelectionPolicyId,
)
from web_search_qualifier.url_identity import normalize_url_identity
from web_search_qualifier.web_search import WebSearchCandidate

_POLICY_ID = SourceSelectionPolicyId("source.python_final_release")
_OFFICIAL_HOST_SUFFIX = "python.org"
_RELEASE_PATH_FRAGMENT = "/downloads/release/"
_PRERELEASE_PATH_MARKERS: tuple[str, ...] = (
    "rc",
    "beta",
    "alpha",
    "a1",
    "a2",
    "b1",
    "b2",
    "b3",
    "b4",
)


def _task_targets_python_release(task_message: str) -> bool:
    lowered = task_message.lower()
    if "python" not in lowered:
        return False
    return any(token in lowered for token in ("release", "version", "3.12", "download"))


def _is_final_release_url(url: str) -> bool:
    normalized = normalize_url_identity(url)
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host.endswith(_OFFICIAL_HOST_SUFFIX):
        return False
    path = parsed.path.lower()
    if _RELEASE_PATH_FRAGMENT not in path:
        return False
    segment = path.split(_RELEASE_PATH_FRAGMENT, 1)[-1]
    return not any(marker in segment for marker in _PRERELEASE_PATH_MARKERS)


def _policy_applies(context: SourceSelectionContext) -> bool:
    if _task_targets_python_release(context.task_message):
        return True
    return any(_is_final_release_url(candidate.url) for candidate in context.candidates)


@dataclass(frozen=True, slots=True)
class PythonFinalReleaseSourcePolicy:
    @property
    def descriptor(self) -> SourceSelectionPolicyDescriptor:
        return SourceSelectionPolicyDescriptor(
            policy_id=_POLICY_ID,
            display_name="Python official final release preference",
        )

    def evaluate(self, context: SourceSelectionContext) -> SourceSelectionPolicyDecision:
        if not _policy_applies(context):
            return SourceSelectionPolicyDecision(
                outcome=SourceSelectionOutcome.ABSTAIN,
                reason_code="not_applicable",
            )
        finals = [
            candidate
            for candidate in context.candidates
            if _is_final_release_url(candidate.url)
        ]
        if not finals:
            return SourceSelectionPolicyDecision(
                outcome=SourceSelectionOutcome.ABSTAIN,
                reason_code="no_final_release_candidate",
            )
        selected = min(finals, key=lambda candidate: candidate.rank)
        return SourceSelectionPolicyDecision(
            outcome=SourceSelectionOutcome.SELECT,
            selected_url=selected.url,
            reason_code="official_final_release",
        )


__all__ = ["PythonFinalReleaseSourcePolicy"]
