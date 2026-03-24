# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Structured client-facing legal response after :class:`LegalFinalizeAnswerStep` draft.

Platform implementations map draft → body / uncertainty / disclaimer and optionally
set :attr:`format_version` for API consumers (``route.extra``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LegalShapedClientResponse(BaseModel):
    """
    Product-layer envelope: separate narrative body from mandatory disclaimers and
    uncertainty callouts. ``compose_legal_client_answer_text`` builds ``RuntimeAnswer.answer``.
    """

    body: str = Field(description="Primary answer text shown to the end user.")
    uncertainty_summary: str = Field(
        default="",
        description="Optional short block on limits of analysis / open questions.",
    )
    disclaimer_block: str = Field(
        default="",
        description="Mandatory legal/product disclaimer copy (not legal advice, etc.).",
    )
    format_version: str = Field(
        default="legal_client_response.v1",
        description="Version tag for host APIs; echoed in RouteInfo.extra when applied.",
    )


def compose_legal_client_answer_text(shaped: LegalShapedClientResponse) -> str:
    """Join non-empty parts with blank lines for ``RuntimeAnswer.answer``."""
    parts: list[str] = []
    for block in (
        shaped.body.strip(),
        shaped.uncertainty_summary.strip(),
        shaped.disclaimer_block.strip(),
    ):
        if block:
            parts.append(block)
    return "\n\n".join(parts)
