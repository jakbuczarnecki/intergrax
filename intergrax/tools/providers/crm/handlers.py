# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.crm.contracts import (
    CrmGetAccountInput,
    CrmGetAccountOutput,
    CrmListContactsInput,
    CrmListContactsOutput,
    CrmListTicketsInput,
    CrmListTicketsOutput,
)
from intergrax.tools.providers.crm.service import crm_get_account, crm_list_contacts, crm_list_tickets


class CrmGetAccountHandler(ServiceToolHandler[CrmGetAccountInput, CrmGetAccountOutput]):
    _service = crm_get_account


class CrmListContactsHandler(ServiceToolHandler[CrmListContactsInput, CrmListContactsOutput]):
    _service = crm_list_contacts


class CrmListTicketsHandler(ServiceToolHandler[CrmListTicketsInput, CrmListTicketsOutput]):
    _service = crm_list_tickets
