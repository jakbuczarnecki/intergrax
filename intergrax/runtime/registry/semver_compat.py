# © Artur Czarnecki. All rights reserved.

"""SemVer compatibility checks for registry resolution (IDEAL-19.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.semver import SemVer

__all__ = ["CompatResult", "SemVer", "is_compatible_runtime"]


@dataclass(frozen=True, slots=True)
class CompatResult:
    compatible: bool
    reason: str = ""


def is_compatible_runtime(requested: str, available: str) -> CompatResult:
    """
    Runtime compatibility: same major, available minor/patch >= requested.

    Pre-1.0 artifacts use minor as breaking boundary.
    """
    req = SemVer.parse(requested)
    avail = SemVer.parse(available)
    if req.major == 0 and avail.major == 0:
        if avail.minor < req.minor:
            return CompatResult(False, "minor below requested pre-1.0 boundary")
        return CompatResult(True)
    if avail.major != req.major:
        return CompatResult(False, "major version mismatch")
    if avail.minor < req.minor:
        return CompatResult(False, "minor below requested")
    if avail.minor == req.minor and avail.patch < req.patch:
        return CompatResult(False, "patch below requested")
    return CompatResult(True)
