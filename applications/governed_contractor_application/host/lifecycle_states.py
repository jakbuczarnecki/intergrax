# © Artur Czarnecki. All rights reserved.

"""Host lifecycle states for governed external work (PC-5)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.contracts.external_work import ExternalWorkStatus


class GovernedExternalWorkHostState(StrEnum):
    REQUESTED = "REQUESTED"
    CREATE_POLICY_DENIED = "CREATE_POLICY_DENIED"
    CREATE_IN_PROGRESS = "CREATE_IN_PROGRESS"
    QUOTE_RECEIVED = "QUOTE_RECEIVED"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    AWAITING_PAYMENT = "AWAITING_PAYMENT"
    ACCEPT_POLICY_DENIED = "ACCEPT_POLICY_DENIED"
    EXECUTION_IN_PROGRESS = "EXECUTION_IN_PROGRESS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_SUCCEEDED_ATTESTATION_PENDING = "EXECUTION_SUCCEEDED_ATTESTATION_PENDING"
    EXECUTION_SUCCEEDED_ATTESTATION_FAILED = "EXECUTION_SUCCEEDED_ATTESTATION_FAILED"
    EXECUTION_SUCCEEDED_ATTESTED = "EXECUTION_SUCCEEDED_ATTESTED"
    CANCELLED = "CANCELLED"


def map_provider_status_to_host_state(
    status: ExternalWorkStatus | None,
    *,
    after_create: bool = False,
) -> GovernedExternalWorkHostState | None:
    """Normalize provider status → host state (provider-neutral mapping)."""
    if status is None:
        return None
    if status in {
        ExternalWorkStatus.QUOTE_AVAILABLE,
        ExternalWorkStatus.WAITING_FOR_ACCEPTANCE,
        ExternalWorkStatus.QUOTE_PENDING,
    }:
        return GovernedExternalWorkHostState.QUOTE_RECEIVED
    if status is ExternalWorkStatus.WAITING_FOR_HUMAN:
        return GovernedExternalWorkHostState.AWAITING_HUMAN
    if status is ExternalWorkStatus.CANCELLED:
        return GovernedExternalWorkHostState.CANCELLED
    if status in {
        ExternalWorkStatus.FAILED,
        ExternalWorkStatus.EXPIRED,
    }:
        return GovernedExternalWorkHostState.EXECUTION_FAILED
    if status in {
        ExternalWorkStatus.COMPLETED,
        ExternalWorkStatus.ACCEPTED,
        ExternalWorkStatus.EXECUTING,
    }:
        return GovernedExternalWorkHostState.EXECUTION_IN_PROGRESS
    if after_create and status in {
        ExternalWorkStatus.CREATED,
        ExternalWorkStatus.INITIALIZING,
    }:
        return GovernedExternalWorkHostState.CREATE_IN_PROGRESS
    return None
