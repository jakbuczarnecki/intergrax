# © Artur Czarnecki. All rights reserved.

"""Approval domain — MP-4D authority integration over MP-4C contracts."""

from intergrax.approval.authority_context import ApprovalAuthorityContextFactory
from intergrax.approval.errors import (
    ApprovalAuthorityContextError,
    ApprovalAuthorizationDenied,
    ApprovalAuthorizationError,
    ApprovalError,
)
from intergrax.approval.service import (
    TRUSTED_OPERATION_APPROVAL_ACTION,
    TRUSTED_OPERATION_APPROVAL_CREATE,
    ApprovalService,
)

__all__ = [
    "ApprovalAuthorityContextError",
    "ApprovalAuthorityContextFactory",
    "ApprovalAuthorizationDenied",
    "ApprovalAuthorizationError",
    "ApprovalError",
    "ApprovalService",
    "TRUSTED_OPERATION_APPROVAL_ACTION",
    "TRUSTED_OPERATION_APPROVAL_CREATE",
]
