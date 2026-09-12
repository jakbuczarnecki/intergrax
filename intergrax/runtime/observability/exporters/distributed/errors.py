# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Distributed observability transport errors (W5-F)."""

from __future__ import annotations

from intergrax.contracts.observability_export import OtlpTransportError


class DistributedTransportError(OtlpTransportError):
    """Collector / broker boundary delivery failure — isolated from execution plane."""
