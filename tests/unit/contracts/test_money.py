# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.money import MoneyAmount

_MONEY_PATH = Path("intergrax/contracts/money.py")


@pytest.mark.unit
@pytest.mark.gate
def test_money_amount_valid_construction_and_roundtrip() -> None:
    money = MoneyAmount(amount=Decimal("12.50"), currency="usd")
    assert money.amount == Decimal("12.50")
    assert money.currency == "USD"
    restored = MoneyAmount.model_validate_json(money.model_dump_json())
    assert restored == money


@pytest.mark.unit
@pytest.mark.gate
def test_money_amount_rejects_float() -> None:
    with pytest.raises(ValidationError, match="binary floating point"):
        MoneyAmount(amount=1.25, currency="USD")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.gate
def test_money_amount_rejects_negative() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        MoneyAmount(amount=Decimal("-0.01"), currency="USD")


@pytest.mark.unit
@pytest.mark.gate
def test_money_amount_rejects_invalid_currency() -> None:
    with pytest.raises(ValidationError, match="ISO 4217"):
        MoneyAmount(amount=Decimal("1"), currency="US1")


@pytest.mark.unit
@pytest.mark.gate
def test_money_amount_is_immutable() -> None:
    money = MoneyAmount(amount=Decimal("1"), currency="EUR")
    with pytest.raises(ValidationError):
        money.amount = Decimal("2")  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.gate
def test_money_module_has_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    source = _MONEY_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source
