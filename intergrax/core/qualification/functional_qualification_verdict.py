# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Verdict aggregation for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.core.qualification.functional_qualification_identity import (
    FunctionalQualificationPluginId,
)


class QualificationVerdict(StrEnum):
    PASS = "PASS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


_VERDICT_PRECEDENCE: tuple[QualificationVerdict, ...] = (
    QualificationVerdict.FAILED,
    QualificationVerdict.BLOCKED,
    QualificationVerdict.PASS,
)


def aggregate_qualification_verdicts(
    plugin_verdicts: tuple[QualificationVerdict, ...],
) -> QualificationVerdict:
    """FAILED > BLOCKED > PASS when combining required plugin outcomes."""
    if not plugin_verdicts:
        return QualificationVerdict.BLOCKED
    for verdict in _VERDICT_PRECEDENCE:
        if verdict in plugin_verdicts:
            return verdict
    return QualificationVerdict.PASS


def aggregate_plugin_verdict(
    *,
    domain_verdict: QualificationVerdict,
    gate_failures: int,
) -> QualificationVerdict:
    if domain_verdict is QualificationVerdict.BLOCKED:
        return QualificationVerdict.BLOCKED
    if domain_verdict is QualificationVerdict.FAILED or gate_failures > 0:
        return QualificationVerdict.FAILED
    return QualificationVerdict.PASS


def verdict_from_domain_string(value: str) -> QualificationVerdict:
    normalized = value.strip().upper()
    if normalized == QualificationVerdict.PASS.value:
        return QualificationVerdict.PASS
    if normalized == QualificationVerdict.FAILED.value:
        return QualificationVerdict.FAILED
    if normalized == QualificationVerdict.BLOCKED.value:
        return QualificationVerdict.BLOCKED
    raise ValueError(f"qualification_verdict_invalid:{value}")


def required_plugin_ids_from_plan(
    plugin_ids: tuple[FunctionalQualificationPluginId, ...],
) -> frozenset[FunctionalQualificationPluginId]:
    return frozenset(plugin_ids)


__all__ = [
    "QualificationVerdict",
    "aggregate_plugin_verdict",
    "aggregate_qualification_verdicts",
    "required_plugin_ids_from_plan",
    "verdict_from_domain_string",
]
