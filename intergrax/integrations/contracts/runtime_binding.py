# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic runtime-binding extension metadata for integration contract specs."""

from __future__ import annotations

from abc import ABC


class IntegrationRuntimeBindingSpec(ABC):
    """Category-specific runtime binding descriptor attached to IntegrationContractSpec."""


__all__ = ["IntegrationRuntimeBindingSpec"]
