# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federation completeness vocabulary for catalog discovery projections."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class CapabilityCatalogFederationCompleteness(StrEnum):
    """Canonical federation completeness — owned by Capability Catalog facts."""

    COMPLETE = "complete"
    PARTIAL = "partial"


NORMATIVE_FEDERATION_COMPLETENESS: Final[
    frozenset[CapabilityCatalogFederationCompleteness]
] = frozenset(CapabilityCatalogFederationCompleteness)
