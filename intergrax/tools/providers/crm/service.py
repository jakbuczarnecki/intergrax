# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.crm import CrmBackend
from intergrax.tools.providers.crm.contracts import (
    CrmAccountOutput,
    CrmContactOutput,
    CrmGetAccountInput,
    CrmGetAccountOutput,
    CrmListContactsInput,
    CrmListContactsOutput,
    CrmListTicketsInput,
    CrmListTicketsOutput,
    CrmTicketOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

CRM_GET_ACCOUNT_TOOL_ID = "crm.get_account"
CRM_LIST_CONTACTS_TOOL_ID = "crm.list_contacts"
CRM_LIST_TICKETS_TOOL_ID = "crm.list_tickets"


def _require_crm(ctx: ToolWiringContext) -> CrmBackend:
    backend = ctx.crm_backend
    if backend is None:
        raise RuntimeError("crm_backend_not_configured")
    return backend


def crm_get_account(ctx: ToolWiringContext, params: CrmGetAccountInput) -> CrmGetAccountOutput:
    account = _require_crm(ctx).get_account(params.account_id.strip())
    return CrmGetAccountOutput(
        account=CrmAccountOutput(
            account_id=account.account_id,
            name=account.name,
            industry=account.industry,
            metadata=dict(account.metadata),
        )
    )


def crm_list_contacts(ctx: ToolWiringContext, params: CrmListContactsInput) -> CrmListContactsOutput:
    contacts = [
        CrmContactOutput(
            contact_id=item.contact_id,
            email=item.email,
            name=item.name,
            account_id=item.account_id or "",
        )
        for item in _require_crm(ctx).list_contacts(account_id=params.account_id.strip(), limit=params.limit)
    ]
    return CrmListContactsOutput(account_id=params.account_id.strip(), contacts=contacts, total=len(contacts))


def crm_list_tickets(ctx: ToolWiringContext, params: CrmListTicketsInput) -> CrmListTicketsOutput:
    tickets = [
        CrmTicketOutput(
            ticket_id=item.ticket_id,
            subject=item.subject,
            status=item.status,
            account_id=item.account_id or "",
        )
        for item in _require_crm(ctx).list_tickets(account_id=params.account_id.strip(), limit=params.limit)
    ]
    return CrmListTicketsOutput(account_id=params.account_id.strip(), tickets=tickets, total=len(tickets))
