# © Artur Czarnecki. All rights reserved.

"""Provider-neutral commercial money representation (GEC-1 / platform).

Distinct from LLM token cost rollups (``AgentRunCost`` uses float USD proxies).
Commercial amounts MUST use ``Decimal`` — never binary floating point.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class MoneyAmount(BaseModel):
    """Immutable non-negative monetary amount with ISO 4217 alphabetic currency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amount: Decimal = Field(description="Non-negative decimal amount (no binary float).")
    currency: str = Field(
        min_length=3,
        max_length=3,
        description="ISO 4217 alphabetic currency code (uppercase).",
    )

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_decimal(cls, value: object) -> Decimal:
        if isinstance(value, Decimal):
            amount = value
        elif isinstance(value, bool):
            raise ValueError("amount must be a Decimal-compatible numeric value")
        elif isinstance(value, int):
            amount = Decimal(value)
        elif isinstance(value, str):
            try:
                amount = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError("amount must be a valid decimal string") from exc
        elif isinstance(value, float):
            raise ValueError("amount must not use binary floating point; use Decimal or str")
        else:
            raise ValueError("amount must be a Decimal-compatible numeric value")
        if amount.is_nan() or amount.is_infinite():
            raise ValueError("amount must be a finite decimal")
        if amount < 0:
            raise ValueError("amount must be non-negative")
        return amount

    @field_validator("currency")
    @classmethod
    def _validate_currency(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not _CURRENCY_RE.match(normalized):
            raise ValueError("currency must be a 3-letter ISO 4217 alphabetic code")
        return normalized
