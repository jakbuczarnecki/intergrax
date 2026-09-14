# © Artur Czarnecki. All rights reserved.

"""Explicit default catalog composition."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.catalog import QualificationCatalog
from testing_support.execution_qualification.catalog.profile_builders import (
    NPSC5E_FINAL_PROFILE_ID,
    NPSC5E_R1_PROFILE_ID,
    NPSC5E_R2_PROFILE_ID,
    NPSC5E_R3_PROFILE_ID,
    NPSC5F_FINAL_PROFILE_ID,
    NPSC5F_R1_PROFILE_ID,
    NPSC5F_R2_PROFILE_ID,
    NPSC5F_R3_PROFILE_ID,
    NPSC5F_R4_PROFILE_ID,
    PROFILE_BUILDERS,
)
from testing_support.execution_qualification.catalog.validation import (
    validate_catalog_profile_ids,
)


def build_default_qualification_catalog() -> QualificationCatalog:
    validate_catalog_profile_ids()
    profile_ids = (
        NPSC5E_R1_PROFILE_ID,
        NPSC5E_R2_PROFILE_ID,
        NPSC5E_R3_PROFILE_ID,
        NPSC5E_FINAL_PROFILE_ID,
        NPSC5F_R1_PROFILE_ID,
        NPSC5F_R2_PROFILE_ID,
        NPSC5F_R3_PROFILE_ID,
        NPSC5F_R4_PROFILE_ID,
        NPSC5F_FINAL_PROFILE_ID,
    )
    return QualificationCatalog(
        profile_ids=profile_ids,
        profile_builders=PROFILE_BUILDERS,
    )
