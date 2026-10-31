# © Artur Czarnecki. All rights reserved.

"""Normative test bindings for INTERNAL / FIXED terminal RPL (no intergrax imports)."""

from __future__ import annotations

from typing import Final

from tests.qualification.ebh_7._ebh7_terminal_operator_authorization import (
    fixed_substrate_rpl_concern_ids,
    internal_rpl_concern_ids,
)

INTERNAL_RPL_CONCERNS: Final[frozenset[str]] = internal_rpl_concern_ids()
FIXED_RPL_CONCERNS: Final[frozenset[str]] = fixed_substrate_rpl_concern_ids()

_TERMINAL_RPL_MODULE: Final[str] = "tests/qualification/ebh_7/test_ebh_7_terminal_rpl_gates.py"


def internal_rpl_positive_node(concern_id: str) -> str:
    return f"{_TERMINAL_RPL_MODULE}::test_ebh7_internal_rpl_positive[{concern_id}]"


def internal_rpl_negative_node(concern_id: str) -> str:
    return f"{_TERMINAL_RPL_MODULE}::test_ebh7_internal_rpl_negative[{concern_id}]"


def fixed_rpl_positive_node(concern_id: str) -> str:
    return f"{_TERMINAL_RPL_MODULE}::test_ebh7_fixed_substrate_rpl_positive[{concern_id}]"


def fixed_rpl_negative_node(concern_id: str) -> str:
    return f"{_TERMINAL_RPL_MODULE}::test_ebh7_fixed_substrate_rpl_negative[{concern_id}]"
