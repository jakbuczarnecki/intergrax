# © Artur Czarnecki. All rights reserved.

"""Apply EBH-7 terminal operator classification to vendor inventory rows."""

from __future__ import annotations

from dataclasses import replace

from tests.qualification.ebh_5._vendor_backend_types import (
    ConcernConclusion,
    ExtensibilityClassification,
    ExtensibilityPromised,
    VendorBackendConcernRecord,
)
from tests.qualification.ebh_7._ebh7_terminal_operator_authorization import (
    TerminalRplDisposition,
    terminal_extensibility_classification,
    terminal_rpl_disposition,
)


def apply_ebh7_terminal_inventory_overlay(row: VendorBackendConcernRecord) -> VendorBackendConcernRecord:
    disp = terminal_rpl_disposition(row.concern_id)
    if disp is None:
        return row
    rpl_h = terminal_extensibility_classification(row.concern_id)
    if rpl_h is not None:
        promised = (
            ExtensibilityPromised.PARTIAL
            if rpl_h is ExtensibilityClassification.FIXED_SUBSTRATE
            else ExtensibilityPromised.YES
        )
        return replace(
            row,
            extensibility_promised=promised,
            extensibility_classification=rpl_h,
            outstanding_debt="",
            conclusion=ConcernConclusion.ACCEPTED,
            behavioral_qualification_evidence_id=(
                row.behavioral_qualification_evidence_id
                if disp is TerminalRplDisposition.REPLACEABLE_DEMONSTRATED
                else f"EBH-7 terminal {disp.value}"
            ),
        )
    return row
