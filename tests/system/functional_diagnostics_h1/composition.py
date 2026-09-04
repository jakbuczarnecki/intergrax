# © Artur Czarnecki. All rights reserved.

"""Explicit composition root for H1 qualification family enumeration."""

from __future__ import annotations

from dataclasses import dataclass

from tests.system.functional_diagnostics_h1.inventory import QUALIFICATION_RUNNERS
from tests.system.functional_diagnostics_h1.models import QualificationFamily


@dataclass(frozen=True, slots=True)
class H1QualificationFamilyDescriptor:
    family: QualificationFamily
    display_name: str
    runner_path: str
    historical_status: str


def build_h1_qualification_families() -> tuple[H1QualificationFamilyDescriptor, ...]:
    historical = {
        QualificationFamily.Q1: "PASS (historical)",
        QualificationFamily.Q2: "PASS (historical)",
        QualificationFamily.Q3: "PASS (historical)",
        QualificationFamily.Q4: "PASS (historical)",
        QualificationFamily.Q5: "PASS (historical)",
        QualificationFamily.D1: "PASS (historical)",
        QualificationFamily.S1: "PASS (historical)",
        QualificationFamily.R1: "PASS (historical)",
        QualificationFamily.R1_R1: "PASS (historical)",
        QualificationFamily.R1_R2: "PASS (historical)",
        QualificationFamily.R1_R3: "PASS (historical)",
    }
    descriptors: list[H1QualificationFamilyDescriptor] = []
    for runner in QUALIFICATION_RUNNERS:
        descriptors.append(
            H1QualificationFamilyDescriptor(
                family=runner.family,
                display_name=runner.family.value,
                runner_path=runner.runner_path,
                historical_status=historical.get(runner.family, "UNKNOWN"),
            )
        )
    descriptors.append(
        H1QualificationFamilyDescriptor(
            family=QualificationFamily.H1,
            display_name="H1 synthetic extension proof",
            runner_path="tests/system/functional_diagnostics_h1/composition.py",
            historical_status="NOT RUN",
        )
    )
    return tuple(descriptors)
