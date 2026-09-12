# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action safety — proposals never self-execute (PREVENTIVE R7)."""

from __future__ import annotations

from typing import Iterable

from intergrax.contracts.external_operations.safety import assert_no_secrets_in_audit_payload


def assert_proposal_has_no_execution_surface(proposal: object) -> None:
    if hasattr(proposal, "execute"):
        raise TypeError("PreventiveActionProposal must not expose execute()")


def assert_no_secrets_in_preventive_action_audit(parts: Iterable[str]) -> None:
    assert_no_secrets_in_audit_payload(parts)


__all__ = [
    "assert_no_secrets_in_preventive_action_audit",
    "assert_proposal_has_no_execution_surface",
]
