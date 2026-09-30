# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sanctioned bootstrap protocol conformance checks (EBH-3-R2)."""

from __future__ import annotations

from intergrax.integrations._shared.conformance import (
    assert_conditional_document_store,
    assert_secrets_store,
)

__all__ = [
    "assert_conditional_document_store",
    "assert_secrets_store",
]
