# © Artur Czarnecki. All rights reserved.

"""Canonical SemVer value object for public contracts (parse/validate only)."""

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


__all__ = ["SemVer"]
