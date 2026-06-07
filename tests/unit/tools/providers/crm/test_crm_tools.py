# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.crm import CrmAccount, CrmContact, CrmTicket
from intergrax.tools.providers.crm.contracts import CrmGetAccountInput, CrmListContactsInput, CrmListTicketsInput
from intergrax.tools.providers.crm.service import crm_get_account, crm_list_contacts, crm_list_tickets
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeCrmBackend:
    def get_account(self, account_id: str) -> CrmAccount:
        return CrmAccount(account_id=account_id, name="Acme", industry="software")

    def list_contacts(self, *, account_id: str, limit: int = 50) -> list[CrmContact]:
        return [CrmContact(contact_id="c-1", email="ops@acme.test", name="Ops", account_id=account_id)]

    def list_tickets(self, *, account_id: str, limit: int = 50) -> list[CrmTicket]:
        return [CrmTicket(ticket_id="t-1", subject="Help", status="open", account_id=account_id)]


def test_crm_tools_return_context() -> None:
    ctx = ToolWiringContext(crm_backend=FakeCrmBackend())
    account = crm_get_account(ctx, CrmGetAccountInput(account_id="a-1"))
    assert account.account.name == "Acme"
    contacts = crm_list_contacts(ctx, CrmListContactsInput(account_id="a-1"))
    assert contacts.total == 1
    tickets = crm_list_tickets(ctx, CrmListTicketsInput(account_id="a-1"))
    assert tickets.total == 1
