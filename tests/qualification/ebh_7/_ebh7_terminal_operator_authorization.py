# © Artur Czarnecki. All rights reserved.

"""EBH-7 terminal operator authorization @ 2026-10-31 — normative classification SSOT."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification


class TerminalRplDisposition(StrEnum):
    INTERNAL_RATIFIED = "INTERNAL_RATIFIED"
    FIXED_SUBSTRATE_RATIFIED = "FIXED_SUBSTRATE_RATIFIED"
    REPLACEABLE_DEMONSTRATED = "REPLACEABLE_DEMONSTRATED"


EBH7_OPERATOR_AUTHORIZATION_SHA: Final[str] = "fd3bc0104c9273f30262946afb16c15c04c8493a"

_INTERNAL_CONCERN_IDS: Final[frozenset[str]] = frozenset(
    {
        "application.composition_root",
        "prodq.colocated_runtime_adapter_isinstance",
    }
)

_FIXED_SUBSTRATE_CONCERN_IDS: Final[frozenset[str]] = frozenset(
    {
        "memory.sqlite.conversational",
        "integration.billing_meter",
        "integration.managed_retrieval",
        "integration.ml_inference_host",
        "integration.model_serving_runtime",
        "integration.vision_serving",
    }
)

_TERMINAL_DISPOSITION: Final[dict[str, TerminalRplDisposition]] = {
    **{cid: TerminalRplDisposition.INTERNAL_RATIFIED for cid in _INTERNAL_CONCERN_IDS},
    **{cid: TerminalRplDisposition.FIXED_SUBSTRATE_RATIFIED for cid in _FIXED_SUBSTRATE_CONCERN_IDS},
    "integration.external_work": TerminalRplDisposition.REPLACEABLE_DEMONSTRATED,
    "agent.effective_profile": TerminalRplDisposition.REPLACEABLE_DEMONSTRATED,
    "tool.marketplace_activation": TerminalRplDisposition.REPLACEABLE_DEMONSTRATED,
}


def terminal_rpl_disposition(concern_id: str) -> TerminalRplDisposition | None:
    return _TERMINAL_DISPOSITION.get(concern_id)


def terminal_extensibility_classification(concern_id: str) -> ExtensibilityClassification | None:
    disp = terminal_rpl_disposition(concern_id)
    if disp is TerminalRplDisposition.INTERNAL_RATIFIED:
        return ExtensibilityClassification.INTERNAL
    if disp is TerminalRplDisposition.FIXED_SUBSTRATE_RATIFIED:
        return ExtensibilityClassification.FIXED_SUBSTRATE
    return None


def internal_rpl_concern_ids() -> frozenset[str]:
    return _INTERNAL_CONCERN_IDS


def fixed_substrate_rpl_concern_ids() -> frozenset[str]:
    return _FIXED_SUBSTRATE_CONCERN_IDS


def assert_ebh7_terminal_authorization_integrity() -> None:
    assert len(_TERMINAL_DISPOSITION) == 11
    assert not (_INTERNAL_CONCERN_IDS & _FIXED_SUBSTRATE_CONCERN_IDS)
