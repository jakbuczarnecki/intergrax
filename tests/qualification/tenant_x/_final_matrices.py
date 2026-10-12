# © Artur Czarnecki. All rights reserved.

"""TENANT-X-FINAL mandatory adversarial matrix (F1–F10)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._final_types import (
    FinalAdversarialClass,
    FinalAdversarialMatrixRow,
    FinalAdversarialOutcome,
    FrzTenId,
    TenantXWave,
)

FINAL_ADVERSARIAL_MATRIX: Final[tuple[FinalAdversarialMatrixRow, ...]] = (
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F1_AUTHORITY_ISOLATION,
        scenario="tenant A authority → tenant B target",
        expected_outcome=FinalAdversarialOutcome.DENIED,
        primary_evidence_owner=TenantXWave.R1,
        accepted_source_waves=(TenantXWave.R1, TenantXWave.R4),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r1_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r1_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_01, FrzTenId.FRZ_TEN_03, FrzTenId.FRZ_TEN_09),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F2_STATE_ISOLATION,
        scenario="tenant A request → tenant B durable/runtime state",
        expected_outcome=FinalAdversarialOutcome.DENIED,
        primary_evidence_owner=TenantXWave.R3,
        accepted_source_waves=(TenantXWave.R3,),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r3_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r3_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_04,),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F3_PROVIDER_CONFIG_ISOLATION,
        scenario="tenant A provider/config/profile binding → tenant B execution",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R2,
        accepted_source_waves=(TenantXWave.R2,),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r2_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r2_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_05, FrzTenId.FRZ_TEN_11),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F4_EVIDENCE_ISOLATION,
        scenario="tenant A evidence → tenant B assessment / attribution",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R4,
        accepted_source_waves=(TenantXWave.R4, TenantXWave.R2),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r4_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r4_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_07,),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F5_CREDENTIAL_ISOLATION,
        scenario="tenant A credential reference → tenant B provider invocation",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R2,
        accepted_source_waves=(TenantXWave.R2,),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r2_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r2_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_06,),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F6_RECOVERY_ISOLATION,
        scenario="tenant A checkpoint/state → tenant B recovery/resume",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R3,
        accepted_source_waves=(TenantXWave.R3,),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r3_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r3_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_08,),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F7_CHILD_ASYNC_ISOLATION,
        scenario="tenant A parent/background/queue/async work → tenant B child execution",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R4,
        accepted_source_waves=(TenantXWave.R4, TenantXWave.R1),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r4_current_head.py",
        executable_evidence="tests.qualification.tenant_x._r4_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_02, FrzTenId.FRZ_TEN_03),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F8_MISSING_TENANT_FAIL_CLOSED,
        scenario="missing tenant on required scope",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.FINAL,
        accepted_source_waves=(TenantXWave.R1, TenantXWave.R2, TenantXWave.R3, TenantXWave.R4),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_final_fast.py",
        executable_evidence="tests.qualification.tenant_x._final_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_10,),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE,
        scenario="extension/provider may not replace/widen/drop tenant scope",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R2,
        accepted_source_waves=(TenantXWave.R2, TenantXWave.R4),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r2_fast.py",
        executable_evidence="tests.qualification.tenant_x._r2_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_11,),
    ),
    FinalAdversarialMatrixRow(
        adversarial_class=FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION,
        scenario="tenant identity consistent request → execution → evidence → event/trace",
        expected_outcome=FinalAdversarialOutcome.REJECTED,
        primary_evidence_owner=TenantXWave.R4,
        accepted_source_waves=(TenantXWave.R4,),
        current_head_validation="tests/qualification/tenant_x/test_tenant_x_r4_fast.py",
        executable_evidence="tests.qualification.tenant_x._r4_adversarial",
        frz_ten_criteria=(FrzTenId.FRZ_TEN_02, FrzTenId.FRZ_TEN_07),
    ),
)
