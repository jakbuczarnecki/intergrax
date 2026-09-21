# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraft-owned artifact and execution target reference parsing (UCA-6C-R)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

_CODECRAFT_ARTIFACT_PREFIX = "codecraft:artifact:"
_CODECRAFT_EXECUTION_TARGET_PREFIX = "codecraft:execution-target:"


def artifact_reference_for_craft(craft_id: str) -> str:
    cid = require_non_empty_text(craft_id, label="craft_id")
    return f"{_CODECRAFT_ARTIFACT_PREFIX}{cid}"


def parse_codecraft_artifact_reference(reference: str) -> str | None:
    ref = reference.strip()
    if not ref.startswith(_CODECRAFT_ARTIFACT_PREFIX):
        return None
    craft_id = ref[len(_CODECRAFT_ARTIFACT_PREFIX) :].strip()
    if not craft_id:
        return None
    return craft_id


def execution_target_reference_for_craft(craft_id: str) -> str:
    cid = require_non_empty_text(craft_id, label="craft_id")
    return f"{_CODECRAFT_EXECUTION_TARGET_PREFIX}{cid}"


def parse_codecraft_execution_target_reference(reference: str) -> str | None:
    ref = reference.strip()
    if not ref.startswith(_CODECRAFT_EXECUTION_TARGET_PREFIX):
        return None
    craft_id = ref[len(_CODECRAFT_EXECUTION_TARGET_PREFIX) :].strip()
    if not craft_id:
        return None
    return craft_id


__all__ = [
    "artifact_reference_for_craft",
    "execution_target_reference_for_craft",
    "parse_codecraft_artifact_reference",
    "parse_codecraft_execution_target_reference",
]
