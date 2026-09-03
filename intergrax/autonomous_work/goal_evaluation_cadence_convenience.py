# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional cadence-ref convenience helpers outside canonical AW semantics."""

from __future__ import annotations

import re

from intergrax.contracts.autonomous_work.references import EvaluationCadenceRef

_CADENCE_SUFFIX_PATTERN = re.compile(r"^(\d+)([smhd])$")


def parse_cadence_interval_seconds(ref: EvaluationCadenceRef) -> int | None:
    """Parse human-readable cadence suffixes such as ``goal-eval-5m`` for config only."""
    suffix = ref.rsplit("-", 1)[-1]
    match = _CADENCE_SUFFIX_PATTERN.fullmatch(suffix)
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * multiplier
