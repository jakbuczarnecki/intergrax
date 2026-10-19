# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider faults mapped into the existing continuity denial codes.

No new platform denial code is introduced. Vendor status values stay here.
"""

from __future__ import annotations

from enum import StrEnum

from intergrax.contracts.governance.history_continuity import ContinuityDenialCode


class ProviderFault(StrEnum):
    REVISION_CONFLICT = "revision_conflict"
    FENCE_CONFLICT = "fence_conflict"
    DIGEST_MISMATCH = "digest_mismatch"
    APPEND_POSITION_CONFLICT = "append_position_conflict"
    OUTCOME_UNKNOWN = "outcome_unknown"
    UNAVAILABLE = "unavailable"
    PREFIX_MISSING = "prefix_missing"
    CORRUPT = "corrupt"
    ALREADY_INITIALIZED = "already_initialized"
    NOT_INITIALIZED = "not_initialized"
    PENDING = "pending"
    SCOPE_MISMATCH = "scope_mismatch"


def map_provider_fault(fault: ProviderFault) -> ContinuityDenialCode:
    """Translate one provider fault into the existing contract denial."""

    if fault is ProviderFault.REVISION_CONFLICT:
        return ContinuityDenialCode.REVISION_STALE
    if fault is ProviderFault.FENCE_CONFLICT:
        return ContinuityDenialCode.FENCE_STALE
    if fault is ProviderFault.DIGEST_MISMATCH:
        return ContinuityDenialCode.WITNESS_MISMATCH
    if fault is ProviderFault.APPEND_POSITION_CONFLICT:
        return ContinuityDenialCode.WITNESS_MISMATCH
    if fault is ProviderFault.OUTCOME_UNKNOWN:
        return ContinuityDenialCode.TRANSITION_UNRESOLVED
    if fault is ProviderFault.UNAVAILABLE:
        return ContinuityDenialCode.RECONCILIATION_UNAVAILABLE
    if fault is ProviderFault.PREFIX_MISSING:
        return ContinuityDenialCode.WITNESS_MISMATCH
    if fault is ProviderFault.CORRUPT:
        return ContinuityDenialCode.WITNESS_MISMATCH
    if fault is ProviderFault.ALREADY_INITIALIZED:
        return ContinuityDenialCode.ALREADY_INITIALIZED
    if fault is ProviderFault.NOT_INITIALIZED:
        return ContinuityDenialCode.NOT_INITIALIZED
    if fault is ProviderFault.PENDING:
        return ContinuityDenialCode.TRANSITION_UNRESOLVED
    if fault is ProviderFault.SCOPE_MISMATCH:
        return ContinuityDenialCode.SCOPE_MISMATCH
    raise TypeError("unsupported provider fault")


def map_transport_exception(exc: BaseException) -> ProviderFault:
    """Map a vendor exception by its type name. This does not import a vendor SDK."""

    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in {"DeadlineExceeded", "ServiceUnavailable", "InternalServerError"}:
        return ProviderFault.OUTCOME_UNKNOWN
    if name == "HttpResponseError" and status == 412:
        return ProviderFault.APPEND_POSITION_CONFLICT
    if name in {"FailedPrecondition", "Aborted"}:
        return ProviderFault.REVISION_CONFLICT
    return ProviderFault.UNAVAILABLE
