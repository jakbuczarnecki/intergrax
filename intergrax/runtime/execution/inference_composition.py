# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Re-export governed inference composition (implementation lives under governance/)."""

from __future__ import annotations

from intergrax.runtime.governance.governed_inference_composition import (
    build_governed_inference_executor,
)

__all__ = ["build_governed_inference_executor"]
