# © Artur Czarnecki. All rights reserved.

"""Ideal Harness error taxonomy — maps runtime codes to families and recovery (IDEAL-22.1/22.2)."""

from __future__ import annotations

from enum import Enum

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode


class HarnessErrorFamily(str, Enum):
    """Aligned with IDEAL_HARNESS_AI_ARCHITECTURE §8.1."""

    USER = "user_error"
    POLICY = "policy_error"
    DEPENDENCY = "dependency_error"
    RUNTIME = "runtime_error"
    QUALITY = "quality_error"


class HarnessRecoveryStrategy(str, Enum):
    """Default recovery posture per family."""

    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_PROVIDER = "fallback_provider"
    DEGRADE_PARTIAL = "degrade_partial"
    ESCALATE_HITL = "escalate_hitl"


_FAMILY_BY_CODE: dict[RuntimeErrorCode, HarnessErrorFamily] = {
    RuntimeErrorCode.USER_ERROR: HarnessErrorFamily.USER,
    RuntimeErrorCode.VALIDATION_ERROR: HarnessErrorFamily.USER,
    RuntimeErrorCode.PERMISSION_ERROR: HarnessErrorFamily.POLICY,
    RuntimeErrorCode.POLICY_ERROR: HarnessErrorFamily.POLICY,
    RuntimeErrorCode.DEPENDENCY_ERROR: HarnessErrorFamily.DEPENDENCY,
    RuntimeErrorCode.TIMEOUT: HarnessErrorFamily.DEPENDENCY,
    RuntimeErrorCode.LLM_ERROR: HarnessErrorFamily.DEPENDENCY,
    RuntimeErrorCode.TOOL_ERROR: HarnessErrorFamily.DEPENDENCY,
    RuntimeErrorCode.RUNTIME_ERROR: HarnessErrorFamily.RUNTIME,
    RuntimeErrorCode.INTERNAL_ERROR: HarnessErrorFamily.RUNTIME,
    RuntimeErrorCode.RUNTIME_ERROR: HarnessErrorFamily.RUNTIME,
    RuntimeErrorCode.QUALITY_ERROR: HarnessErrorFamily.QUALITY,
}

_RECOVERY_BY_FAMILY: dict[HarnessErrorFamily, HarnessRecoveryStrategy] = {
    HarnessErrorFamily.USER: HarnessRecoveryStrategy.FAIL_FAST,
    HarnessErrorFamily.POLICY: HarnessRecoveryStrategy.ESCALATE_HITL,
    HarnessErrorFamily.DEPENDENCY: HarnessRecoveryStrategy.RETRY_WITH_BACKOFF,
    HarnessErrorFamily.RUNTIME: HarnessRecoveryStrategy.RETRY_WITH_BACKOFF,
    HarnessErrorFamily.QUALITY: HarnessRecoveryStrategy.DEGRADE_PARTIAL,
}


def family_for_code(code: RuntimeErrorCode) -> HarnessErrorFamily:
    return _FAMILY_BY_CODE.get(code, HarnessErrorFamily.RUNTIME)


def recovery_for_code(code: RuntimeErrorCode) -> HarnessRecoveryStrategy:
    return _RECOVERY_BY_FAMILY[family_for_code(code)]


def is_quality_failure(code: RuntimeErrorCode) -> bool:
    return family_for_code(code) is HarnessErrorFamily.QUALITY


def is_dependency_failure(code: RuntimeErrorCode) -> bool:
    return family_for_code(code) is HarnessErrorFamily.DEPENDENCY
