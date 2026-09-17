# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Map read-model truncation flags to inspection section completeness."""

from __future__ import annotations

from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness


def completeness_for_read_result(is_truncated: bool) -> RuntimeInspectionCompleteness:
    if is_truncated:
        return RuntimeInspectionCompleteness.PARTIAL
    return RuntimeInspectionCompleteness.COMPLETE


__all__ = ["completeness_for_read_result"]
