# © Artur Czarnecki. All rights reserved.

"""Golden prompt catalog content checks (FAUDIT-PE.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field


class PromptGoldenExpectation(BaseModel):
    prompt_id: str
    version: int
    content_sha256: str


class PromptGoldenCatalogReport(BaseModel):
    schema_version: str = "1.0.0"
    checked: int = 0
    passed: bool = True
    failures: list[str] = Field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_prompt_golden_catalog(
    *,
    catalog_dir: Path,
    expectations: tuple[PromptGoldenExpectation, ...],
) -> PromptGoldenCatalogReport:
    failures: list[str] = []
    for item in expectations:
        prompt_path = catalog_dir / item.prompt_id / f"v{item.version}.yaml"
        if not prompt_path.is_file():
            failures.append(f"missing prompt file: {prompt_path.as_posix()}")
            continue
        digest = _sha256(prompt_path.read_text(encoding="utf-8"))
        if digest != item.content_sha256:
            failures.append(
                f"content hash mismatch for {item.prompt_id} v{item.version}"
            )
    return PromptGoldenCatalogReport(
        checked=len(expectations),
        passed=not failures,
        failures=failures,
    )
