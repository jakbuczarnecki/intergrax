# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.identity.contracts import (
    IdentityGetUserInput,
    IdentityGetUserOutput,
    IdentityListTenantsInput,
    IdentityListTenantsOutput,
    IdentityVerifyTokenInput,
    IdentityVerifyTokenOutput,
)
from intergrax.tools.providers.identity.service import identity_get_user, identity_list_tenants, identity_verify_token


class IdentityVerifyTokenHandler(ServiceToolHandler[IdentityVerifyTokenInput, IdentityVerifyTokenOutput]):
    _service = identity_verify_token


class IdentityGetUserHandler(ServiceToolHandler[IdentityGetUserInput, IdentityGetUserOutput]):
    _service = identity_get_user


class IdentityListTenantsHandler(ServiceToolHandler[IdentityListTenantsInput, IdentityListTenantsOutput]):
    _service = identity_list_tenants
