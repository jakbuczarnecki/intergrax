# © Artur Czarnecki. All rights reserved.

"""Retention policy per data classification (IDEAL-23.5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.data_classification import DataClassification


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    classification: DataClassification
    retention_days: int


_DEFAULT_RETENTION: dict[DataClassification, int] = {
    DataClassification.PUBLIC: 365,
    DataClassification.INTERNAL: 180,
    DataClassification.CONFIDENTIAL: 90,
    DataClassification.RESTRICTED: 30,
}


def retention_days_for(classification: DataClassification) -> int:
    return _DEFAULT_RETENTION[classification]
