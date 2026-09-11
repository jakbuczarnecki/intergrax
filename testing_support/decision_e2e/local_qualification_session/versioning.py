# © Artur Czarnecki. All rights reserved.

"""Normalized provider runtime version parsing (no stringly matching)."""

from __future__ import annotations

import re

from testing_support.decision_e2e.local_qualification_session.contracts import (
    ProviderRuntimeVersion,
    QualificationIdentityStatus,
    VersionMatchPolicy,
)

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def parse_provider_runtime_version(raw: str) -> ProviderRuntimeVersion | None:
    text = raw.strip()
    match = _VERSION_RE.match(text)
    if match is None:
        return None
    return ProviderRuntimeVersion(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
    )


def compare_runtime_versions(
    expected: ProviderRuntimeVersion | None,
    actual: ProviderRuntimeVersion | None,
    *,
    policy: VersionMatchPolicy,
) -> QualificationIdentityStatus:
    if policy is VersionMatchPolicy.IGNORE:
        return QualificationIdentityStatus.MATCH
    if expected is None:
        return QualificationIdentityStatus.UNVERIFIABLE
    if actual is None:
        return QualificationIdentityStatus.UNVERIFIABLE
    if (
        expected.major == actual.major
        and expected.minor == actual.minor
        and expected.patch == actual.patch
    ):
        return QualificationIdentityStatus.MATCH
    return QualificationIdentityStatus.MISMATCH


def compare_digest_values(
    expected: str | None,
    actual: str | None,
    *,
    policy: VersionMatchPolicy,
) -> QualificationIdentityStatus:
    if policy is VersionMatchPolicy.IGNORE:
        return QualificationIdentityStatus.MATCH
    if expected is None:
        return QualificationIdentityStatus.UNVERIFIABLE
    if actual is None:
        return QualificationIdentityStatus.UNVERIFIABLE
    if expected == actual:
        return QualificationIdentityStatus.MATCH
    return QualificationIdentityStatus.MISMATCH
