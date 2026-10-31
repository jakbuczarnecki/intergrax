# © Artur Czarnecki. All rights reserved.

"""Documented non-PROHIBITED RPL residuals permitted only after OD-EBH7-SCOPE-RESIDUAL."""

from __future__ import annotations

from typing import Final

# Populated when operator approves scoped residuals (B6 prep). Empty = fail-closed @ normative gate.
OD_EBH7_SCOPE_RESIDUAL_APPROVED: Final[bool] = False

_SCOPE_RESIDUAL_CONCERN_IDS: Final[frozenset[str]] = frozenset(
    {
        # Example placeholders — operator must approve before non-empty allowance at B6.
    }
)


def allowed_non_demonstrated_concern_ids() -> frozenset[str]:
    if not OD_EBH7_SCOPE_RESIDUAL_APPROVED:
        return frozenset()
    return _SCOPE_RESIDUAL_CONCERN_IDS
