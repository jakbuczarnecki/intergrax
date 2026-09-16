# © Artur Czarnecki. All rights reserved.

"""Shared RequestIdentity fixtures for user-profile projection lifecycle tests."""

from __future__ import annotations

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity


def memory_test_identity(
    *,
    tenant_id: str = "tenant-test",
    user_id: str = "u1",
    principal_type: PrincipalType = PrincipalType.USER,
    auth_subject: str | None = None,
) -> RequestIdentity:
    subject = auth_subject if auth_subject is not None else user_id
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=principal_type,
        auth_subject=subject,
    )
