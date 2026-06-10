# © Artur Czarnecki. All rights reserved.

"""Prompt diff / compare helpers for managed prompts (AUDIT-IDEAL-17.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.prompts.schema.prompt_schema import LocalizedPromptDocument


@dataclass(frozen=True, slots=True)
class PromptFieldDiff:
    field_name: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class PromptCompareResult:
    prompt_id: str
    left_version: int
    right_version: int
    changed_fields: tuple[PromptFieldDiff, ...]
    identical: bool


def compare_prompt_documents(
    left: LocalizedPromptDocument,
    right: LocalizedPromptDocument,
    *,
    locale: str = "en",
) -> PromptCompareResult:
    """Compare two prompt document versions for a single locale."""
    if left.id != right.id:
        raise ValueError("prompt ids must match for compare")

    left_content = left.locales.get(locale) or next(iter(left.locales.values()), None)
    right_content = right.locales.get(locale) or next(iter(right.locales.values()), None)
    if left_content is None or right_content is None:
        raise ValueError("both documents must expose at least one locale")

    diffs: list[PromptFieldDiff] = []
    for field_name in ("system", "developer", "user_template"):
        before = getattr(left_content, field_name) or ""
        after = getattr(right_content, field_name) or ""
        if before != after:
            diffs.append(PromptFieldDiff(field_name=field_name, before=before, after=after))

    if left.meta.owner_team != right.meta.owner_team:
        diffs.append(
            PromptFieldDiff(
                field_name="meta.owner_team",
                before=left.meta.owner_team,
                after=right.meta.owner_team,
            )
        )

    return PromptCompareResult(
        prompt_id=left.id,
        left_version=left.version,
        right_version=right.version,
        changed_fields=tuple(diffs),
        identical=not diffs,
    )
