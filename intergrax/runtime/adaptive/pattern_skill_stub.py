# © Artur Czarnecki. All rights reserved.

"""Optional skill manifest stub generator for mined patterns (Phase W-ADAPT-6.4)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.contracts import ProcessPatternAction, ProcessPatternProposal


class SkillStubDraft(BaseModel):
    """Draft skill manifest for human review — not auto-registered."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    draft_id: str
    pattern_id: str
    suggested_action: ProcessPatternAction
    skill_id: str
    description: str
    tool_ids: list[str] = Field(default_factory=list)
    evidence_run_ids: list[str] = Field(default_factory=list)
    merge_instructions: str = (
        "Review this draft and run `python -m intergrax.scaffold new-skill` manually."
    )


def build_skill_stub_draft(proposal: ProcessPatternProposal) -> SkillStubDraft | None:
    """Build a skill stub only for CREATE_SKILL_DRAFT suggestions."""
    if proposal.suggested_action != ProcessPatternAction.CREATE_SKILL_DRAFT:
        return None
    tool_ids = _extract_tool_ids(proposal.description)
    skill_id = f"mined.{proposal.pattern_id}"
    return SkillStubDraft(
        draft_id=f"stub_{proposal.pattern_id}",
        pattern_id=proposal.pattern_id,
        suggested_action=proposal.suggested_action,
        skill_id=skill_id,
        description=proposal.description,
        tool_ids=tool_ids,
        evidence_run_ids=list(proposal.evidence_run_ids),
    )


def write_skill_stub_draft(
    proposal: ProcessPatternProposal,
    *,
    output_dir: Path,
) -> Path | None:
    """Write skill stub JSON to output directory; returns path when written."""
    draft = build_skill_stub_draft(proposal)
    if draft is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{draft.draft_id}.json"
    output_path.write_text(json.dumps(draft.model_dump(mode="json"), indent=2), encoding="utf-8")
    return output_path


def _extract_tool_ids(description: str) -> list[str]:
    marker = "tools="
    if marker not in description:
        return []
    fragment = description.split(marker, maxsplit=1)[1]
    raw = fragment.split(" ", maxsplit=1)[0]
    if raw == "no-tool":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
