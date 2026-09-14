# © Artur Czarnecki. All rights reserved.

"""Global semantic parity certification matrix (legacy reference vs canonical profiles)."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5E_FINAL_MANDATORY,
    NPSC5E_R1_FINAL_MANDATORY,
    NPSC5E_R2_FINAL_MANDATORY,
    NPSC5E_R3_FINAL_MANDATORY,
    NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
    NPSC5F_R1_FINAL_MANDATORY,
    NPSC5F_R2_FINAL_MANDATORY,
    NPSC5F_R3_FINAL_MANDATORY,
    NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
)
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
    NPSC5F_R1_ROOT_GATE_ID,
)
from testing_support.execution_qualification.semantic_parity.models import (
    QualificationSemanticParityCase,
)

GLOBAL_SEMANTIC_PARITY_MATRIX: tuple[QualificationSemanticParityCase, ...] = (
    QualificationSemanticParityCase(
        profile_id=NPSC5F_R1_PROFILE_ID,
        legacy_semantic_source=NPSC5F_R1_FINAL_MANDATORY,
        canonical_profile_id=NPSC5F_R1_PROFILE_ID,
        expected_root_ids=(NPSC5F_R1_ROOT_GATE_ID,),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5E_R3_PROFILE_ID,
        legacy_semantic_source=NPSC5E_R3_FINAL_MANDATORY,
        canonical_profile_id=NPSC5E_R3_PROFILE_ID,
        expected_root_ids=("npsc5e-r3.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5E_R2_PROFILE_ID,
        legacy_semantic_source=NPSC5E_R2_FINAL_MANDATORY,
        canonical_profile_id=NPSC5E_R2_PROFILE_ID,
        expected_root_ids=("npsc5e-r2.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5E_R1_PROFILE_ID,
        legacy_semantic_source=NPSC5E_R1_FINAL_MANDATORY,
        canonical_profile_id=NPSC5E_R1_PROFILE_ID,
        expected_root_ids=("npsc5e-r1.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5E_FINAL_PROFILE_ID,
        legacy_semantic_source=NPSC5E_FINAL_MANDATORY,
        canonical_profile_id=NPSC5E_FINAL_PROFILE_ID,
        expected_root_ids=("npsc5e.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5F_R2_PROFILE_ID,
        legacy_semantic_source=NPSC5F_R2_FINAL_MANDATORY,
        canonical_profile_id=NPSC5F_R2_PROFILE_ID,
        expected_root_ids=("npsc5f-r2.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5F_R3_PROFILE_ID,
        legacy_semantic_source=NPSC5F_R3_FINAL_MANDATORY,
        canonical_profile_id=NPSC5F_R3_PROFILE_ID,
        expected_root_ids=("npsc5f-r3.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5F_R4_PROFILE_ID,
        legacy_semantic_source=NPSC5F_R4_MANDATORY_REGRESSION_SUITES,
        canonical_profile_id=NPSC5F_R4_PROFILE_ID,
        expected_root_ids=("npsc5f-r4.final",),
    ),
    QualificationSemanticParityCase(
        profile_id=NPSC5F_FINAL_PROFILE_ID,
        legacy_semantic_source=NPSC5F_FINAL_MANDATORY_REGRESSION_SUITES,
        canonical_profile_id=NPSC5F_FINAL_PROFILE_ID,
        expected_root_ids=("npsc5f.final",),
    ),
)
