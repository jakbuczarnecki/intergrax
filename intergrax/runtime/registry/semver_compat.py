# © Artur Czarnecki. All rights reserved.

"""SemVer compatibility checks for registry resolution (IDEAL-19.2)."""

from __future__ import annotations

from dataclasses import dataclass
import re

_SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:[-+].*)?$"
)


@dataclass(frozen=True, slots=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> SemVer:
        match = _SEMVER_RE.match(value.strip())
        if match is None:
            raise ValueError(f"invalid semver: {value!r}")
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )


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
