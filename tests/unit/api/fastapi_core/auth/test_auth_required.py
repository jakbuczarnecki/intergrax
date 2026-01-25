# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.fastapi_core.auth.authorization import AuthRequired
from intergrax.fastapi_core.auth.context import AuthContext
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.errors.auth import NotAuthenticatedError


def test_auth_required_allows_authenticated_request() -> None:
    context = RequestContext(
        request_id="req-1",
        tenant_id="tenant-1",
        path="/test",
        method="POST",
        user_id="user-1",
        auth=AuthContext(
            is_authenticated=True,
            tenant_id="tenant-1",
            user_id="user-1",
            scopes=(),
        ),
    )

    guard = AuthRequired()

    # Should not raise
    guard(context)


def test_auth_required_blocks_unauthenticated_request() -> None:
    context = RequestContext(
        request_id="req-1",
        path="/test",
        method="POST",
        tenant_id=None,
        user_id=None,
        auth=AuthContext(
            is_authenticated=False,
            tenant_id=None,
            user_id=None,
            scopes=(),
        ),
    )

    guard = AuthRequired()

    with pytest.raises(NotAuthenticatedError):
        guard(context)
