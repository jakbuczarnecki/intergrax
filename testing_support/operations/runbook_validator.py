# © Artur Czarnecki. All rights reserved.

"""Parse and validate Execution Engine production runbook markdown (EE-B4-C)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from testing_support.operations.forbidden_patterns import FORBIDDEN_RUNBOOK_PATTERNS
from testing_support.operations.incident_taxonomy import REQUIRED_RUNBOOK_IDS
from testing_support.operations.runbook_contract import (
    RUNBOOK_REQUIRED_SECTIONS,
    RunbookSectionId,
)

_RUNBOOK_HEADER = re.compile(r"^##\s+(RB-\d{2})\s+", re.MULTILINE)
_SECTION_HEADER = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class RunbookBlock:
    runbook_id: str
    body: str


@dataclass(frozen=True, slots=True)
class RunbookValidationResult:
    runbook_id: str
    missing_sections: tuple[str, ...]
    forbidden_hits: tuple[str, ...]


def split_runbook_blocks(markdown: str) -> dict[str, RunbookBlock]:
    matches = list(_RUNBOOK_HEADER.finditer(markdown))
    blocks: dict[str, RunbookBlock] = {}
    for index, match in enumerate(matches):
        runbook_id = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        blocks[runbook_id] = RunbookBlock(
            runbook_id=runbook_id, body=markdown[start:end]
        )
    return blocks


def sections_present_in_block(body: str) -> set[str]:
    found: set[str] = set()
    for section_match in _SECTION_HEADER.finditer(body):
        title = section_match.group(1).strip()
        found.add(title)
    return found


def missing_required_sections(body: str) -> tuple[str, ...]:
    present = sections_present_in_block(body)
    required_titles = {section.value for section in RUNBOOK_REQUIRED_SECTIONS}
    missing = sorted(required_titles - present)
    return tuple(missing)


_ACTION_SECTIONS_FOR_FORBIDDEN_SCAN: tuple[RunbookSectionId, ...] = (
    RunbookSectionId.IMMEDIATE_SAFE_ACTION,
    RunbookSectionId.DIAGNOSIS,
    RunbookSectionId.RECOVERY_PATH,
)


def _lines_without_prohibition_context(section_text: str) -> str:
    kept: list[str] = []
    for line in section_text.splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("do not"):
            continue
        if normalized.startswith("- do not"):
            continue
        if normalized.startswith("* do not"):
            continue
        kept.append(line)
    return "\n".join(kept)


def find_forbidden_patterns(text: str) -> tuple[str, ...]:
    """Scan operator-action sections only (Do NOT may list prohibitions verbatim)."""
    hits: list[str] = []
    for section in _ACTION_SECTIONS_FOR_FORBIDDEN_SCAN:
        section_text = runbook_section_text(text, section)
        if not section_text:
            continue
        scan_text = _lines_without_prohibition_context(section_text)
        for entry in FORBIDDEN_RUNBOOK_PATTERNS:
            if entry.regex.search(scan_text):
                hits.append(entry.pattern_id)
    return tuple(sorted(set(hits)))


def validate_runbook_document(path: Path) -> tuple[RunbookValidationResult, ...]:
    markdown = path.read_text(encoding="utf-8")
    blocks = split_runbook_blocks(markdown)
    results: list[RunbookValidationResult] = []
    for runbook_id in REQUIRED_RUNBOOK_IDS:
        block = blocks.get(runbook_id)
        if block is None:
            results.append(
                RunbookValidationResult(
                    runbook_id=runbook_id,
                    missing_sections=tuple(s.value for s in RUNBOOK_REQUIRED_SECTIONS),
                    forbidden_hits=(),
                )
            )
            continue
        missing = missing_required_sections(block.body)
        forbidden = find_forbidden_patterns(block.body)
        results.append(
            RunbookValidationResult(
                runbook_id=runbook_id,
                missing_sections=missing,
                forbidden_hits=forbidden,
            )
        )
    return tuple(results)


def runbook_section_text(body: str, section: RunbookSectionId) -> str:
    pattern = re.compile(
        rf"^###\s+{re.escape(section.value)}\s*\n(.*?)(?=^###\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def get_runbook_body(markdown: str, runbook_id: str) -> str:
    blocks = split_runbook_blocks(markdown)
    block = blocks.get(runbook_id)
    return block.body if block else ""
