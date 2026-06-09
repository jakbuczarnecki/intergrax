# © Artur Czarnecki. All rights reserved.

"""AWS Bedrock Guardrails vendor boundary (boto3 allowed here only)."""

from __future__ import annotations

from typing import Any


def bedrock_apply_guardrail(text: str, *, policy_id: str, mode: str) -> dict[str, Any]:
    import boto3

    client = boto3.client("bedrock")
    response = client.apply_guardrail(
        guardrailIdentifier=policy_id,
        guardrailVersion="DRAFT",
        source="INPUT" if mode == "input" else "OUTPUT",
        content=[{"text": {"text": text}}],
    )
    return dict(response)
